import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Hide INFO logs
import tensorflow as tf
import time
import numpy as np
import glob
from pathlib import Path
from tqdm import tqdm
import argparse
import sys
try:
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
except ImportError: print("ERROR: scikit-learn not found. Run: pip install scikit-learn"); exit(1)
try:
    import matplotlib.pyplot as plt
    PLOTTING_ENABLED = True
except ImportError:
    print("Warning: matplotlib not found. Reliability diagram plotting disabled.")
    PLOTTING_ENABLED = False
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, Layer, Softmax

# =============================================================================
# --- Configuration ---
# =============================================================================
parser = argparse.ArgumentParser(
    description="Run inference and calibration analysis using preprocessed NPZ data.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--model_path", type=str, default='./BestTestingDataset.keras',
                    help="Path to the saved Keras model (.keras).")
parser.add_argument("--data_dir", type=str, default='processed_testing_data',
                    help="Directory containing preprocessed NPZ test files.")
parser.add_argument("--temperature", type=float, default=1.4199,
                    help="Temperature for scaling logits. Set to optimal value from calibration.")

args = parser.parse_args()

MODEL_SAVE_PATH = args.model_path
PROCESSED_TEST_DATA_DIR = args.data_dir
OPTIMAL_TEMPERATURE = args.temperature
APPLY_TEMP_SCALING = True
BATCH_SIZE = 2048
FFT_LENGTH = 2000
NUM_CLASSES = 10
NUM_BINS_ECE = 15

print("=" * 50)
print(" Starting Inference & Calibration Analysis (NPZ Data)")
print("=" * 50)
print(f"TensorFlow Version: {tf.__version__}")
print(f"Model Path: {MODEL_SAVE_PATH}")
print(f"Test Data Path (Processed NPZ): {PROCESSED_TEST_DATA_DIR}")
print(f"Apply Temperature Scaling: {APPLY_TEMP_SCALING}")
if APPLY_TEMP_SCALING: print(f"Using Temperature (T): {OPTIMAL_TEMPERATURE:.4f}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"FFT Length: {FFT_LENGTH}")
print(f"Num Classes: {NUM_CLASSES}")
print(f"ECE Bins: {NUM_BINS_ECE}")
print("-" * 50)

# =============================================================================
# --- Data Loading (NPZ) Function ---
# =============================================================================

def load_npz_data(npz_path_tensor):
    def _load_npz(npz_path_bytes):
        npz_path = npz_path_bytes.decode('utf-8');
        try:
            with np.load(npz_path) as data: features = data['features'].astype(np.float32); labels = data['labels'].astype(np.int32);
            if features.shape[0] == 0 or labels.shape[0] == 0 or features.shape[0] != labels.shape[0]: print(f"Warning: Empty/mismatched data in {npz_path}", file=sys.stderr); return np.zeros((0, FFT_LENGTH, 1), dtype=np.float32), np.zeros((0,), dtype=np.int32);
            if features.shape[1] != FFT_LENGTH: print(f"Warning: Feature length mismatch in {npz_path}. Expected {FFT_LENGTH}, got {features.shape[1]}", file=sys.stderr); return np.zeros((0, FFT_LENGTH, 1), dtype=np.float32), np.zeros((0,), dtype=np.int32)
            return features, labels
        except Exception as e: print(f"Error loading NPZ {npz_path}: {e}", file=sys.stderr); return np.zeros((0, FFT_LENGTH, 1), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    features, labels = tf.numpy_function(_load_npz, [npz_path_tensor], (tf.float32, tf.int32));
    features.set_shape([None, FFT_LENGTH, 1]); labels.set_shape([None]); return features, labels


# =============================================================================
# --- Inference Dataset Creation (Using NPZ) ---
# =============================================================================

def create_npz_dataset(processed_dir, batch_size, cache_dataset=True):
    npz_pattern = os.path.join(processed_dir, "processed_*.npz")
    npz_file_paths = tf.io.gfile.glob(npz_pattern)
    if not npz_file_paths:
        raise FileNotFoundError(f"No processed .npz files found: {npz_pattern}")
    print(f"Found {len(npz_file_paths)} processed NPZ files.")
    dataset = tf.data.Dataset.from_tensor_slices(npz_file_paths)
    dataset = dataset.map(load_npz_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.filter(lambda features, labels: tf.shape(features)[0] > 0)
    dataset = dataset.unbatch()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    print("Inference dataset pipeline created.")
    return dataset

# =============================================================================
# --- Calibration Analysis Functions (Unchanged) ---
# =============================================================================

def calculate_ece(y_true, y_pred_probs, n_bins=10):
    pred_classes = np.argmax(y_pred_probs, axis=1); confidences = np.max(y_pred_probs, axis=1); accuracies = (pred_classes == y_true).astype(np.float32); ece = 0.0; bin_lowers = np.linspace(0.0, 1.0, n_bins + 1)[:-1]; bin_uppers = np.linspace(0.0, 1.0, n_bins + 1)[1:]; bin_accuracies = np.zeros(n_bins); bin_confidences = np.zeros(n_bins); bin_counts = np.zeros(n_bins);
    for i in range(n_bins): in_bin = (confidences > bin_lowers[i]) & (confidences <= bin_uppers[i]); bin_counts[i] = np.sum(in_bin);
    if bin_counts[i] > 0: bin_accuracies[i] = np.mean(accuracies[in_bin]); bin_confidences[i] = np.mean(confidences[in_bin]); ece += bin_counts[i] * np.abs(bin_accuracies[i] - bin_confidences[i]);
    total_samples = np.sum(bin_counts);
    if total_samples == 0: return 0.0, bin_accuracies, bin_confidences, bin_counts;
    ece = ece / total_samples; return ece, bin_accuracies, bin_confidences, bin_counts

def plot_reliability_diagram(bin_accuracies, bin_confidences, bin_counts, n_bins, title="Reliability Diagram"):
    if not PLOTTING_ENABLED: print("Plotting disabled."); return;
    bin_lowers = np.linspace(0.0, 1.0, n_bins + 1)[:-1]; bin_centers = bin_lowers + (1.0 / (2 * n_bins)); plt.figure(figsize=(6, 6)); plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration'); valid_bins = bin_counts > 0; plt.plot(bin_confidences[valid_bins], bin_accuracies[valid_bins], marker='o', linestyle='-', label='Model Calibration'); plt.xlabel("Average Confidence in Bin"); plt.ylabel("Accuracy in Bin"); plt.title(title); plt.legend(); plt.grid(True, alpha=0.5); plt.xlim(0, 1); plt.ylim(0, 1); save_path = f"{title.replace(' ', '_').lower()}.png"; plt.savefig(save_path); print(f"Reliability diagram saved to {save_path}"); plt.close();


# =============================================================================
# --- Main Inference and Calibration Logic ---
# =============================================================================

if __name__ == "__main__":
    main_start_time = time.time()
    # --- Load Trained Model ---
    print(f"\nLoading trained Keras model from: {MODEL_SAVE_PATH}")
    try:
        trained_model = load_model(MODEL_SAVE_PATH, compile=False)
        print("Trained model loaded successfully.")
    except Exception as e:
        print(f"\nFATAL ERROR loading model: {e}"); import traceback; traceback.print_exc(); exit(1)

    # --- Create Logit Model (Temp Scaling & ECE) ---
    print("\nCreating logit model (output before final softmax)...")
    logit_model = None
    try:
        final_layer = trained_model.layers[-1]
        if isinstance(final_layer, Dense) and hasattr(final_layer, 'activation') and final_layer.activation == tf.keras.activations.softmax:
            original_weights = final_layer.get_weights(); config = final_layer.get_config(); config['activation'] = 'linear'; config['dtype'] = 'float32'; logit_input_tensor = trained_model.layers[-2].output; new_logit_layer = Dense.from_config(config)(logit_input_tensor); logit_model = Model(inputs=trained_model.input, outputs=new_logit_layer); logit_model.layers[-1].set_weights(original_weights); print("Logit model created via replacement.")
        elif isinstance(final_layer, Dense): print(f"Assuming final layer '{final_layer.name}' is logits."); logit_model = Model(inputs=trained_model.input, outputs=final_layer.output)
        elif isinstance(final_layer, Softmax): logit_model = Model(inputs=trained_model.input, outputs=trained_model.layers[-2].output); print("Using output of layer before final Softmax layer.")
        else: raise TypeError(f"Cannot determine logit layer from final layer type: {type(final_layer)}")
        if not logit_model: raise RuntimeError("Logit model creation failed.")
        print(f"Logit model output shape: {logit_model.output.shape}")
    except Exception as e: print(f"FATAL Error creating logit model: {e}"); logit_model = None

    # --- Create Inference Dataset ---
    print(f"\nCreating Inference Dataset from NPZ files in: {PROCESSED_TEST_DATA_DIR}")
    try:
        inference_dataset = create_npz_dataset(
            PROCESSED_TEST_DATA_DIR,
            BATCH_SIZE
        )
    except Exception as e: print(f"\nFATAL ERROR creating inference dataset: {e}"); exit(1)

    # --- Perform Inference ---
    print("\nStarting inference...")
    all_true_labels = []
    all_predicted_classes_unscaled = []
    all_predicted_classes_scaled = []
    all_probs_unscaled = []
    all_probs_scaled = []
    batch_num = 0
    start_inference_time = time.time()

    if logit_model is None:
        print("WARNING: Logit model not available. Cannot perform scaling or ECE. Running basic prediction.")
        APPLY_TEMP_SCALING = False

    for x_batch, y_batch in tqdm(inference_dataset, desc="Inference Batches"):
        batch_num += 1
        try:
            if logit_model:
                batch_logits = logit_model.predict_on_batch(x_batch)
                batch_probs_unscaled = tf.nn.softmax(batch_logits, axis=-1).numpy()
                batch_pred_unscaled = np.argmax(batch_probs_unscaled, axis=1)
                scaled_logits = batch_logits / OPTIMAL_TEMPERATURE
                batch_probs_scaled = tf.nn.softmax(scaled_logits, axis=-1).numpy()
                batch_pred_scaled = np.argmax(batch_probs_scaled, axis=1)

                all_probs_unscaled.append(batch_probs_unscaled)
                all_probs_scaled.append(batch_probs_scaled)
                all_predicted_classes_unscaled.extend(batch_pred_unscaled)
                all_predicted_classes_scaled.extend(batch_pred_scaled)
            else:
                 final_batch_probs = trained_model.predict_on_batch(x_batch)
                 batch_pred_classes = np.argmax(final_batch_probs, axis=1)
                 all_probs_unscaled.append(final_batch_probs)
                 all_probs_scaled.append(final_batch_probs)
                 all_predicted_classes_unscaled.extend(batch_pred_classes)
                 all_predicted_classes_scaled.extend(batch_pred_classes)

            all_true_labels.extend(y_batch.numpy())

        except Exception as e:
             print(f"\nERROR DURING INFERENCE ON BATCH {batch_num}: {e}")

    end_inference_time = time.time()
    print(f"\nInference finished in {end_inference_time - start_inference_time:.2f} seconds.")

    if not all_true_labels: print("\nError: No samples processed."); exit(1)

    y_true = np.array(all_true_labels)
    y_pred_unscaled = np.array(all_predicted_classes_unscaled)
    y_pred_scaled = np.array(all_predicted_classes_scaled)
    probs_unscaled_np = np.concatenate(all_probs_unscaled, axis=0) if all_probs_unscaled else np.array([])
    probs_scaled_np = np.concatenate(all_probs_scaled, axis=0) if all_probs_scaled else np.array([])

    min_len = len(y_true)
    if len(y_pred_unscaled) != min_len or len(y_pred_scaled) != min_len or \
       (len(probs_unscaled_np) > 0 and probs_unscaled_np.shape[0] != min_len) or \
       (len(probs_scaled_np) > 0 and probs_scaled_np.shape[0] != min_len):
        print("\nWarning: Length mismatch detected between labels and predictions/probabilities.")
        min_len = min(len(y_true), len(y_pred_unscaled), len(y_pred_scaled),
                      len(probs_unscaled_np) if len(probs_unscaled_np)>0 else min_len,
                      len(probs_scaled_np) if len(probs_scaled_np)>0 else min_len)
        y_true = y_true[:min_len]
        y_pred_unscaled = y_pred_unscaled[:min_len]
        y_pred_scaled = y_pred_scaled[:min_len]
        if len(probs_unscaled_np)>0: probs_unscaled_np = probs_unscaled_np[:min_len]
        if len(probs_scaled_np)>0: probs_scaled_np = probs_scaled_np[:min_len]

    if min_len == 0: print("No valid results available for metrics."); exit(1)


    # --- Calculate and Display Standard Metrics ---
    print("\n" + "-" * 50)
    print(" Standard Metrics (Unscaled Model, T=1.0)")
    print("-" * 50)
    accuracy_unscaled = accuracy_score(y_true, y_pred_unscaled)
    conf_matrix_unscaled = confusion_matrix(y_true, y_pred_unscaled)
    unique_labels_unscaled = np.unique(np.concatenate((y_true, y_pred_unscaled)))
    class_report_unscaled = classification_report(y_true, y_pred_unscaled, labels=unique_labels_unscaled, zero_division=0, digits=4)
    print(f"Overall Test Accuracy (Unscaled): {accuracy_unscaled:.4f}")
    print("\nConfusion Matrix (Unscaled):")
    print(conf_matrix_unscaled)
    print("\nClassification Report (Unscaled):")
    print(class_report_unscaled)

    if APPLY_TEMP_SCALING and logit_model is not None:
        print("\n" + "-" * 50)
        print(f" Standard Metrics (Scaled Model, T={OPTIMAL_TEMPERATURE:.4f})")
        print("-" * 50)
        accuracy_scaled = accuracy_score(y_true, y_pred_scaled)
        conf_matrix_scaled = confusion_matrix(y_true, y_pred_scaled)
        unique_labels_scaled = np.unique(np.concatenate((y_true, y_pred_scaled)))
        class_report_scaled = classification_report(y_true, y_pred_scaled, labels=unique_labels_scaled, zero_division=0, digits=4)
        print(f"Overall Test Accuracy (Scaled): {accuracy_scaled:.4f}")
        print("\nConfusion Matrix (Scaled):")
        print(conf_matrix_scaled)
        print("\nClassification Report (Scaled):")
        print(class_report_scaled)
    print("-" * 50)

    # --- Calibration Analysis ---
    print("\nCalculating Calibration Metrics...")
    if logit_model and len(probs_unscaled_np) > 0 and len(probs_scaled_np) > 0:
        ece_unscaled, acc_unscaled, conf_unscaled, counts_unscaled = calculate_ece(y_true, probs_unscaled_np, n_bins=NUM_BINS_ECE)
        print(f"\nCalibration BEFORE Scaling (T=1.0):")
        print(f"  Expected Calibration Error (ECE): {ece_unscaled:.5f}")
        if PLOTTING_ENABLED: plot_reliability_diagram(acc_unscaled, conf_unscaled, counts_unscaled, NUM_BINS_ECE, title="Reliability Diagram (Unscaled T=1.0)")

        if APPLY_TEMP_SCALING:
            ece_scaled, acc_scaled, conf_scaled, counts_scaled = calculate_ece(y_true, probs_scaled_np, n_bins=NUM_BINS_ECE)
            print(f"\nCalibration AFTER Scaling (T={OPTIMAL_TEMPERATURE:.4f}):")
            print(f"  Expected Calibration Error (ECE): {ece_scaled:.5f}")
            if PLOTTING_ENABLED: plot_reliability_diagram(acc_scaled, conf_scaled, counts_scaled, NUM_BINS_ECE, title=f"Reliability Diagram (Scaled T={OPTIMAL_TEMPERATURE:.4f})")

            if ece_scaled < ece_unscaled: print("\n-> Temperature scaling IMPROVED calibration (lower ECE).")
            elif ece_scaled > ece_unscaled: print("\n-> Temperature scaling WORSENED calibration (higher ECE).")
            else: print("\n-> Temperature scaling had no significant effect on ECE.")
        else:
            print("\nTemperature Scaling was disabled, only showing unscaled calibration.")
    else:
        print("\nSkipping calibration analysis: Logit model unavailable or no probabilities collected.")
    print("-" * 50)

    # --- Final Summary ---
    main_end_time = time.time()
    print("\n" + "=" * 40)
    print(f" Script Finished in {main_end_time - main_start_time:.2f} seconds. ")
    print("=" * 40)
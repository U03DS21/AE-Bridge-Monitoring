import os
import tensorflow as tf
import time
import numpy as np
import glob
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import argparse
import sys
import scipy.optimize
import scipy.special
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, MaxPooling1D,
    GlobalAveragePooling1D, Dense, Dropout, Add, LayerNormalization,
    MultiHeadAttention, SeparableConv1D
)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras import mixed_precision
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
try:
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    import matplotlib.pyplot as plt
    PLOTTING_ENABLED = True
except ImportError:
    print("Warning: scikit-learn or matplotlib not found. Final test report/plotting disabled.")
    PLOTTING_ENABLED = False
try:
    import coremltools as ct
    COREML_ENABLED = False # Default to false, this is only for experimentation
except ImportError:
    print("Warning: coremltools not found. Core ML conversion will be skipped.")
    COREML_ENABLED = False; ct = None

# =============================================================================
# --- Configuration ---
# =============================================================================

PROCESSED_TRAIN_DATA_DIR = 'processed_training_data'
PROCESSED_VAL_DATA_DIR   = 'processed_validation_data'
PROCESSED_TEST_DATA_DIR  = 'processed_testing_data'

# Dataset Parameters
NUM_CLASSES = 10
FFT_LENGTH = 2000

# Training Hyperparameters
EPOCHS = 1600
BATCH_SIZE = 256
DROPOUT_RATE = 0.3
CLIP_GRADIENTS = True
EARLY_STOPPING_PATIENCE = 200

# Optimization & Output Toggles
ENABLE_MIXED_PRECISION = True
ENABLE_CACHING = True
ENABLE_PROFILING = True
PROFILING_EPOCH = 20
TENSORBOARD_LOG_DIR = './logs/npz_train_val_logs_cnn'
PLOT_MODEL_FILE = './model_architecture_cnn.png'
BEST_MODEL_SAVE_PATH = './best_model_cnn.keras'
COREML_MODEL_SAVE_PATH = './best_model_cnn.mlpackage'

# Temperature Scaling
PROCESSED_CALIBRATION_DATA_DIR = PROCESSED_TEST_DATA_DIR
CALIBRATION_BATCH_SIZE = BATCH_SIZE
APPLY_TEMP_SCALING = True
TEMP_SCALE_OPTIMAL_T = 1.0
TEMP_SCALE_NUM_BINS_ECE = 15
NUM_BINS_ECE = 15
TEMP_SCALE_INITIAL_T = [1.5]
TEMP_SCALE_BOUNDS = [(0.1, 10.0)]

# =============================================================================
# --- Model Definition ---
# =============================================================================

def create_pure_cnn_model_v2(input_shape, num_classes,
                             cnn_filters=[64, 128, 256],
                             cnn_kernels=[9, 5, 3],
                             cnn_pools=[2, 2, 2],
                             dense_units=128,
                             dropout_rate=0.3,
                             transformer_heads=4,
                             transformer_ff_dim=None,
                             use_separable_conv=True,
                             bottleneck_ratio=0.5):

    if not (len(cnn_filters) == len(cnn_kernels) == len(cnn_pools)):
        raise ValueError("Length of cnn_filters, cnn_kernels, and cnn_pools must match.")

    if transformer_ff_dim is None:
        transformer_ff_dim = dense_units * 4 if dense_units > 0 else 0

    inputs = Input(shape=input_shape, name='input_layer', dtype=tf.float32)
    x = inputs
    print(f"Building CNN-Transformer V2 with {len(cnn_filters)} CNN blocks.")
    print(f"Using Dropout rate: {dropout_rate}")
    print(f"Intermediate Dense units: {dense_units}")
    if dense_units > 0:
        print(f"Transformer: heads={transformer_heads}, ff_dim={transformer_ff_dim}")
    print(f"CNN Config: SeparableConv = {use_separable_conv}, Bottleneck Ratio = {bottleneck_ratio}")

    # --- CNN Blocks ---
    for i, (filters, kernel, pool_size) in enumerate(zip(cnn_filters, cnn_kernels, cnn_pools)):
        block_num = i + 1
        block_input = x

        # --- Bottleneck Layer (1x1 Conv) ---
        apply_bottleneck = bottleneck_ratio is not None and bottleneck_ratio < 1.0 and i > 0
        if apply_bottleneck:
            bottleneck_filters = max(8, int(filters * bottleneck_ratio))
            print(f"  Block {block_num}: Applying bottleneck {x.shape[-1]} -> {bottleneck_filters} channels.")
            x = Conv1D(filters=bottleneck_filters,
                       kernel_size=1,
                       padding='same',
                       use_bias=False,
                       kernel_initializer='he_normal',
                       name=f'conv1d_bottleneck_{block_num}')(x)
            x = BatchNormalization(name=f'batchnorm_bottleneck_{block_num}')(x)
            x = Activation('gelu', name=f'activation_bottleneck_{block_num}')(x)
            input_channels_main_conv = bottleneck_filters
        else:
            input_channels_main_conv = x.shape[-1]
            print(f"  Block {block_num}: No bottleneck applied.")


        # --- Main Convolution (Standard or Separable) ---
        if use_separable_conv:
            print(f"    Using SeparableConv1D: kernel={kernel}, filters={filters}")
            x = SeparableConv1D(
                filters=filters,
                kernel_size=kernel,
                padding='same',
                use_bias=False,
                depthwise_initializer='he_normal',
                pointwise_initializer='he_normal',
                name=f'separable_conv1d_{block_num}'
            )(x)
        else:
            print(f"    Using Conv1D: kernel={kernel}, in_channels={input_channels_main_conv}, out_channels={filters}")
            x = Conv1D(
                filters=filters,
                kernel_size=kernel,
                strides=1,
                padding='same',
                use_bias=False,
                kernel_initializer='he_normal',
                name=f'conv1d_{block_num}'
            )(x)

        x = BatchNormalization(name=f'batchnorm_{block_num}')(x)
        x = Activation('gelu', name=f'activation_{block_num}')(x)

        # --- Pooling ---
        if pool_size > 1:
            print(f"    Applying MaxPooling1D: pool_size={pool_size}")
            x = MaxPooling1D(pool_size=pool_size, strides=pool_size,
                             padding='valid', name=f'maxpool_{block_num}')(x)

        # --- Dropout ---
        if dropout_rate > 0.0:
            x = Dropout(dropout_rate, name=f'cnn_dropout_{block_num}')(x)

    # --- Intermediate Dense Layer & Transformer Block ---
    if dense_units > 0:
        print(f"\nApplying Intermediate Dense layer ({dense_units} units) and Transformer block.")
        x = Dense(dense_units,
                  kernel_initializer='he_normal',
                  name='intermediate_dense')(x)

        # --- Multi-Head Self-Attention ---
        x_norm1 = LayerNormalization(epsilon=1e-6, name='transformer_mha_prenorm')(x)
        attn_output = MultiHeadAttention(
            num_heads=transformer_heads, key_dim=dense_units, dropout=dropout_rate,
            name='transformer_mha'
        )(query=x_norm1, value=x_norm1, key=x_norm1)
        x = Add(name='transformer_mha_add')([x, attn_output])

        # --- Feed-Forward Network ---
        x_norm2 = LayerNormalization(epsilon=1e-6, name='transformer_ffn_prenorm')(x)
        ffn_output = Dense(transformer_ff_dim, activation='gelu', name='transformer_ffn_dense1')(x_norm2)
        if dropout_rate > 0.0:
            ffn_output = Dropout(dropout_rate, name='transformer_ffn_dropout1')(ffn_output)
        ffn_output = Dense(dense_units, name='transformer_ffn_dense2')(ffn_output)
        x = Add(name='transformer_ffn_add')([x, ffn_output])

    else:
        print("\nSkipping Intermediate Dense and Transformer block as dense_units <= 0")

    # --- Final Global Pooling and Softmax ---
    print("\nApplying Global Average Pooling and Final Dense layer.")
    x = GlobalAveragePooling1D(name='global_avg_pool')(x)
    outputs = Dense(num_classes, activation='softmax', dtype=tf.float32, name='output_dense')(x)

    model = Model(inputs=inputs, outputs=outputs, name='cnn_transformer_model_v2')
    return model

# =============================================================================
# --- Data Loading (NPZ) and Dataset Creation ---
# =============================================================================

def load_npz_data(npz_path_tensor):
    def _load_npz(npz_path_bytes):
        npz_path = npz_path_bytes.decode('utf-8')
        try:
            with np.load(npz_path) as data:
                features = data['features'].astype(np.float32)
                labels = data['labels'].astype(np.int32)
                if features.shape[0] == 0 or labels.shape[0] == 0 or features.shape[0] != labels.shape[0]:
                     return np.zeros((0, FFT_LENGTH, 1), dtype=np.float32), np.zeros((0,), dtype=np.int32)
                if features.shape[1] != FFT_LENGTH:
                    print(f"ERROR: Feature length mismatch in {npz_path}. Expected {FFT_LENGTH}, got {features.shape[1]}", file=sys.stderr)
                    return np.zeros((0, FFT_LENGTH, 1), dtype=np.float32), np.zeros((0,), dtype=np.int32)
                return features, labels
        except FileNotFoundError:
             print(f"Error: NPZ file not found: {npz_path}", file=sys.stderr)
             return np.zeros((0, FFT_LENGTH, 1), dtype=np.float32), np.zeros((0,), dtype=np.int32)
        except Exception as e:
             print(f"Error loading NPZ file {npz_path}: {type(e).__name__} - {e}", file=sys.stderr)
             return np.zeros((0, FFT_LENGTH, 1), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    features, labels = tf.numpy_function(
        _load_npz,
        [npz_path_tensor],
        (tf.float32, tf.int32)
    )
    features.set_shape([None, FFT_LENGTH, 1])
    labels.set_shape([None])
    return features, labels

def create_npz_dataset(processed_dir, batch_size, is_training, cache_dataset=True, file_limit=None):
    purpose = "training" if is_training else "evaluation/calibration/testing"
    npz_pattern = os.path.join(processed_dir, "processed_*.npz")
    npz_file_paths = sorted(tf.io.gfile.glob(npz_pattern))
    if not npz_file_paths:
        raise FileNotFoundError(f"No processed .npz files found for {purpose}: {npz_pattern}")

    if file_limit and file_limit < len(npz_file_paths):
        print(f"Limiting dataset to first {file_limit} files for {purpose}.")
        npz_file_paths = npz_file_paths[:file_limit]

    print(f"Found {len(npz_file_paths)} processed NPZ files for {purpose}.")

    dataset = tf.data.Dataset.from_tensor_slices(npz_file_paths)
    dataset = dataset.map(load_npz_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.filter(lambda features, labels: tf.shape(features)[0] > 0)
    dataset = dataset.unbatch()

    if cache_dataset:
        print(f"Caching {purpose} dataset...")
        dataset = dataset.cache()

    if is_training:
        shuffle_buffer = 20000
        print(f"Shuffling training samples with buffer size: {shuffle_buffer}")
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
        print("Repeating training dataset for epochs...")
        dataset = dataset.repeat()
    else:
        print(f"Dataset for {purpose} configured for single pass (no shuffle/repeat).")

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    print(f"{purpose.capitalize()} dataset pipeline created.")
    return dataset

# =============================================================================
# --- Training Function ---
# =============================================================================

def train_model(model, train_dataset, validation_dataset, total_train_samples, validation_steps, callbacks=None):
    print("\nCalculating training steps per epoch...")
    if total_train_samples <= 0: raise ValueError(f"Total train samples must be positive.")
    steps_per_epoch = max(1, total_train_samples // BATCH_SIZE)
    if total_train_samples % BATCH_SIZE != 0: steps_per_epoch += 1
    print(f"Using total training samples = {total_train_samples}")
    print(f"Calculated training steps per epoch: {steps_per_epoch}")

    if validation_steps <= 0: raise ValueError("Validation steps must be positive.")
    print(f"Using validation steps per epoch: {validation_steps}")

    decay_steps = 53890
    initial_learning_rate = 0.0
    warmup_steps = 510
    target_learning_rate = 2e-3
    lr_schedule = CosineDecay(
        initial_learning_rate,
        decay_steps,
        alpha=0.0,
        name="CosineDecay",
        warmup_target=target_learning_rate,
        warmup_steps=warmup_steps,
    )

    optimizer_base = AdamW(learning_rate=lr_schedule, weight_decay=1e-5,
                           beta_1=0.9, beta_2=0.99, epsilon=1e-7,
                           clipnorm=2.0 if CLIP_GRADIENTS else None)

    if ENABLE_MIXED_PRECISION:
        print("Wrapping optimizer with LossScaleOptimizer for Mixed Precision.")
        optimizer = mixed_precision.LossScaleOptimizer(optimizer_base)
    else:
        print("Using base AdamW optimizer.")
        optimizer = optimizer_base

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("\nStarting model.fit with validation...")
    start_fit_time = time.time()
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_dataset,
        validation_steps=validation_steps,
        verbose=1,
        callbacks=callbacks
    )
    end_fit_time = time.time()
    print(f"model.fit duration: {end_fit_time - start_fit_time:.2f} seconds")
    print("Model training finished (or stopped early by EarlyStopping).")
    return history, model

# =============================================================================
# --- Temperature Scaling & Calibration Analysis Functions ---
# =============================================================================

def calculate_nll_numpy(T, logits, labels_onehot):
    T_val = T[0]
    if T_val <= 1e-6:
        return np.inf
    try:
        logits_np = logits.astype(np.float64)
        labels_onehot_np = labels_onehot.astype(np.float64)
        T_np = np.float64(T_val)

        scaled_logits_np = logits_np / T_np
        log_probs_np = scipy.special.log_softmax(scaled_logits_np, axis=-1)

        nll_per_sample = -np.sum(labels_onehot_np * log_probs_np, axis=1)
        mean_nll = np.mean(nll_per_sample)

        if not np.isfinite(mean_nll):
            return np.finfo(np.float64).max
        return mean_nll
    except Exception as e:
        print(f"ERROR in calculate_nll_numpy (T={T_val:.4f}): {type(e).__name__} - {e}", file=sys.stderr)
        return np.finfo(np.float64).max

def calculate_ece(y_true, y_pred_probs, n_bins=10):
    if y_true.shape[0] == 0 or y_pred_probs.shape[0] == 0:
        print("Warning: calculate_ece received empty arrays.")
        return 0.0, np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    if y_true.shape[0] != y_pred_probs.shape[0]:
        raise ValueError("y_true and y_pred_probs must have the same number of samples.")

    pred_classes = np.argmax(y_pred_probs, axis=1)
    confidences = np.max(y_pred_probs, axis=1)
    accuracies = (pred_classes == y_true).astype(np.float32)

    ece = 0.0
    bin_lowers = np.linspace(0.0, 1.0, n_bins + 1)[:-1]
    bin_uppers = np.linspace(0.0, 1.0, n_bins + 1)[1:]

    bin_accuracies = np.zeros(n_bins, dtype=np.float64)
    bin_confidences = np.zeros(n_bins, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    for i in range(n_bins):
        if i == n_bins - 1:
             in_bin = (confidences > bin_lowers[i]) & (confidences <= bin_uppers[i] + 1e-9)
        else:
             in_bin = (confidences > bin_lowers[i]) & (confidences <= bin_uppers[i])

        bin_counts[i] = np.sum(in_bin)

        if bin_counts[i] > 0:
            bin_accuracies[i] = np.mean(accuracies[in_bin])
            bin_confidences[i] = np.mean(confidences[in_bin])
            ece += bin_counts[i] * np.abs(bin_accuracies[i] - bin_confidences[i])

    total_samples = np.sum(bin_counts)
    if total_samples == 0:
        print("Warning: No samples found in any bin for ECE calculation.")
        return 0.0, bin_accuracies, bin_confidences, bin_counts

    ece = ece / total_samples
    return ece, bin_accuracies, bin_confidences, bin_counts


def plot_reliability_diagram(bin_accuracies, bin_confidences, bin_counts, n_bins, title="Reliability Diagram"):
    if not PLOTTING_ENABLED:
        print("Plotting disabled (matplotlib/sklearn not found).")
        return
    plot_dir = Path("./plots_cnn")
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_path = plot_dir / f"{title.replace(' ', '_').lower()}.png"

    # --- Plotting code ---
    bin_lowers = np.linspace(0.0, 1.0, n_bins + 1)[:-1]
    bin_centers = bin_lowers + (1.0 / (2 * n_bins))

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')

    valid_bins = bin_counts > 0
    if np.any(valid_bins):
        plt.plot(bin_confidences[valid_bins], bin_accuracies[valid_bins], marker='o', linestyle='-', color='blue', label='Model Calibration')
    else:
        print(f"Warning: No valid bins with data to plot in '{title}'.")

    plt.xlabel("Average Confidence in Bin")
    plt.ylabel("Accuracy in Bin")
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()

    try:
        plt.savefig(save_path)
        print(f"Reliability diagram saved to {save_path}")
    except Exception as e:
        print(f"Error saving plot {save_path}: {e}")
    finally:
        plt.close()


# =============================================================================
# --- Main Execution Block ---
# =============================================================================

if __name__ == "__main__":
    main_start_time = time.time()
    print("=" * 60)
    print("  Training (CNN-Transformer), Validation, Testing & Temp Scaling (NPZ)")
    print("  (No ModelCheckpoint, Saves Best Model After Training)")
    print("=" * 60)

    # --- Set Mixed Precision Policy ---
    if ENABLE_MIXED_PRECISION:
        print(f"TensorFlow Version: {tf.__version__}")
        print("Enabling Mixed Precision (mixed_float16)...")
        try:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            print(f"Compute dtype: {policy.compute_dtype}, Variable dtype: {policy.variable_dtype}")
        except Exception as e:
            print(f"Error setting Mixed Precision policy: {e}. Continuing without.")
            ENABLE_MIXED_PRECISION = False
    else:
        print("Mixed Precision DISABLED.")

    # --- Count Samples (Train & Validation) ---
    def count_npz_samples(data_dir):
        count = 0
        pattern = os.path.join(data_dir, "processed_*.npz")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No NPZ files found in {data_dir}. Cannot count samples.")
        print(f"Found {len(files)} NPZ files in {data_dir}. Counting samples...")
        for f_path in tqdm(files, desc=f"Counting Samples in {os.path.basename(data_dir)}", unit="file", leave=False):
            try:
                with np.load(f_path) as data:
                    if 'labels' in data and data['labels'].ndim > 0:
                        count += len(data['labels'])
                    else:
                        print(f"\nWarning: 'labels' key missing or empty in {f_path}", file=sys.stderr)
            except Exception as e:
                print(f"\nWarning: Could not read/count samples in {f_path}: {e}", file=sys.stderr)
        if count <= 0:
            raise ValueError(f"No valid samples counted in {data_dir}. Check NPZ files.")
        print(f"Found {count} total samples in {data_dir}.")
        return count

    try:
        total_train_samples = count_npz_samples(PROCESSED_TRAIN_DATA_DIR)
        total_val_samples = count_npz_samples(PROCESSED_VAL_DATA_DIR)
    except Exception as e:
        print(f"\nFATAL ERROR counting samples: {e}. Please check data directories and NPZ file integrity.")
        exit(1)

    validation_steps = max(1, total_val_samples // BATCH_SIZE)
    if total_val_samples % BATCH_SIZE != 0: validation_steps += 1

    # --- Create Datasets ---
    print(f"\nCreating Datasets (Batch Size: {BATCH_SIZE})...")
    try:
        train_dataset = create_npz_dataset(PROCESSED_TRAIN_DATA_DIR, BATCH_SIZE, is_training=True, cache_dataset=ENABLE_CACHING)
        val_dataset = create_npz_dataset(PROCESSED_VAL_DATA_DIR, BATCH_SIZE, is_training=False, cache_dataset=ENABLE_CACHING)
        print("Training and Validation dataset pipelines created successfully.")
    except Exception as e:
        print(f"\nFATAL ERROR creating datasets: {e}")
        exit(1)

    # --- Set Input Shape ---
    input_shape = (FFT_LENGTH, 1)
    print(f"\nUsing fixed Input shape = {input_shape}")

    # --- Model Creation ---
    print("\nCreating Pure 1D CNN Model V2...")
    try:
        model = create_pure_cnn_model_v2(
            input_shape=input_shape,
            num_classes=NUM_CLASSES,
        )
        model.summary(line_length=120)
        PLOT_MODEL_FILE = './model_architecture_cnn_v2.png'
        try: plot_model(model, to_file=PLOT_MODEL_FILE, show_shapes=True); print(f"Model plot saved to {PLOT_MODEL_FILE}")
        except Exception as plot_e: print(f"Could not plot model: {plot_e}")
    except Exception as model_e:
        print(f"\nFATAL ERROR during model creation: {model_e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # --- Setup Callbacks ---
    print("\nSetting up Training Callbacks (TensorBoard, EarlyStopping)...")
    callbacks = []
    base_log_dir = Path(TENSORBOARD_LOG_DIR)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_log_dir = base_log_dir / f"run_{timestamp}_cnn"
    unique_log_dir.mkdir(parents=True, exist_ok=True)
    print(f"TensorBoard log directory for this run: {unique_log_dir.resolve()}")
    base_log_dir_for_cmd = base_log_dir

    tensorboard_callback = TensorBoard(log_dir=str(unique_log_dir), histogram_freq=1, write_graph=True)
    callbacks.append(tensorboard_callback)

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )
    callbacks.append(early_stopping_callback)
    print(f"Enabled EarlyStopping: monitor='val_acuraccy', patience={EARLY_STOPPING_PATIENCE}, restore_best_weights=True")

    if ENABLE_PROFILING:
        print(f"*** Enabling Profiler for epoch {PROFILING_EPOCH} ***")
        prof_steps_per_epoch = max(1, total_train_samples // BATCH_SIZE)
        if total_train_samples % BATCH_SIZE != 0: prof_steps_per_epoch += 1
        profile_start_step = (PROFILING_EPOCH - 1) * prof_steps_per_epoch + 1
        profile_end_step = min(profile_start_step + 49, PROFILING_EPOCH * prof_steps_per_epoch)
        if profile_start_step <= profile_end_step and PROFILING_EPOCH >= 1:
            profile_batch_tuple = (profile_start_step, profile_end_step)
            print(f"Profiler capturing steps: {profile_batch_tuple}")
            profiler_callback = TensorBoard(log_dir=str(unique_log_dir / "profile"), profile_batch=profile_batch_tuple)
            callbacks.append(profiler_callback)
            print(f"View logs & profiles: tensorboard --logdir {base_log_dir_for_cmd.resolve()}")
        else:
            print("Warning: Invalid profile range calculated. Profiling disabled.")
            print(f"View logs with: tensorboard --logdir {base_log_dir_for_cmd.resolve()}")
    else:
        print(f"View logs with: tensorboard --logdir {base_log_dir_for_cmd.resolve()}")


    # --- Training ---
    print("\nStarting Training with Validation...")
    start_run_time = time.time()
    history, model_with_best_weights = train_model(
        model,
        train_dataset,
        validation_dataset=val_dataset,
        total_train_samples=total_train_samples,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    end_run_time = time.time()
    print(f"\nTotal Training duration: {end_run_time - start_run_time:.2f} seconds")

    # --- Save the Best Model ---
    print(f"\nTraining complete. Saving the best model (weights restored by EarlyStopping) to: {BEST_MODEL_SAVE_PATH}")
    try:
        save_model(model_with_best_weights, BEST_MODEL_SAVE_PATH)
        print("Best model saved successfully.")
        model_to_evaluate = model_with_best_weights
    except Exception as e:
        print(f"\nERROR saving the final best model: {e}")
        print("Proceeding with the model currently in memory, but it might not be saved.")
        model_to_evaluate = model_with_best_weights


    # --- Temperature Scaling Calibration ---
    print("\n" + "=" * 40)
    print(" Starting Temperature Scaling Calibration (using Best Trained Model)")
    print("=" * 40)
    optimal_T_found = 1.0
    try:
        print(f"Creating Calibration Dataset from: {PROCESSED_CALIBRATION_DATA_DIR}")
        cal_dataset_for_scaling = create_npz_dataset(
            PROCESSED_CALIBRATION_DATA_DIR,
            CALIBRATION_BATCH_SIZE,
            is_training=False,
            cache_dataset=ENABLE_CACHING
        )

        print("Creating logit model from the best trained model...")
        logit_model_for_scaling = None
        try:
            final_layer_cal = model_to_evaluate.layers[-1]
            if isinstance(final_layer_cal, Dense) and final_layer_cal.activation == tf.keras.activations.softmax:
                pre_softmax_output = model_to_evaluate.layers[-2].output
                config = final_layer_cal.get_config()
                config['activation'] = 'linear'
                config['dtype'] = 'float32'
                new_logit_layer_output = Dense.from_config(config)(pre_softmax_output)
                logit_model_for_scaling = Model(inputs=model_to_evaluate.input, outputs=new_logit_layer_output, name=f"{model_to_evaluate.name}_logits")
                logit_model_for_scaling.layers[-1].set_weights(final_layer_cal.get_weights())
                print("Logit model for scaling created by replacing final activation.")
            else:
                 print(f"Warning: Could not automatically extract logits. Last layer is {type(final_layer_cal)} with activation {getattr(final_layer_cal, 'activation', 'N/A')}.")
                 logit_model_for_scaling = None

            if not logit_model_for_scaling:
                 raise RuntimeError("Logit model creation failed.")

        except Exception as e:
            print(f"Error creating logit model for scaling: {e}. Check model structure.")
            logit_model_for_scaling = None

        if logit_model_for_scaling:
             print("Calculating logits on calibration data...")
             all_cal_logits, all_cal_labels = [], []
             for x_batch, y_batch in tqdm(cal_dataset_for_scaling, desc="Predicting Calib Set (Logits)", unit="batch", leave=False):
                  batch_logits = logit_model_for_scaling.predict_on_batch(x_batch)
                  all_cal_logits.append(batch_logits)
                  all_cal_labels.append(y_batch.numpy())

             if not all_cal_logits:
                  raise ValueError("No logits were calculated. Calibration dataset might be empty or prediction failed.")

             cal_logits_np = np.concatenate(all_cal_logits, axis=0)
             cal_labels_np = np.concatenate(all_cal_labels, axis=0)
             cal_labels_onehot_np = tf.one_hot(cal_labels_np.astype(np.int32), depth=NUM_CLASSES).numpy()
             print(f"Finished calculating calibration logits. Shape: {cal_logits_np.shape}")

             print("\nOptimizing Temperature (T) using SciPy L-BFGS-B...")
             optimization_result = scipy.optimize.minimize(
                 calculate_nll_numpy,
                 TEMP_SCALE_INITIAL_T,
                 args=(cal_logits_np, cal_labels_onehot_np),
                 method='L-BFGS-B',
                 bounds=TEMP_SCALE_BOUNDS,
                 options={'disp': False, 'ftol': 1e-9, 'gtol': 1e-7}
             )

             if optimization_result.success:
                 optimal_T_found = optimization_result.x[0]
                 min_nll = optimization_result.fun
                 print(f"Optimization Successful! Optimal T = {optimal_T_found:.4f}, Min NLL = {min_nll:.4f}")
             else:
                 print(f"Warning: Temperature optimization failed: {optimization_result.message}.")
                 print("Using default T=1.0 for subsequent steps.")
                 optimal_T_found = 1.0

             TEMP_SCALE_OPTIMAL_T = optimal_T_found
             print(f"\nTemperature Scaling calibration complete. Optimal T set to: {TEMP_SCALE_OPTIMAL_T:.4f}")

        else:
             print("Skipping temperature scaling optimization as logit model creation failed.")
             TEMP_SCALE_OPTIMAL_T = 1.0

    except FileNotFoundError as e:
        print(f"\nError finding calibration data: {e}. Skipping temperature scaling.")
        TEMP_SCALE_OPTIMAL_T = 1.0
    except Exception as e:
        print(f"\nAn error occurred during temperature scaling: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()
        print("Using default T=1.0 due to error.")
        TEMP_SCALE_OPTIMAL_T = 1.0
    print("=" * 40)


    # --- Final Testing ---
    print("\n" + "=" * 40)
    print(" Starting Final Evaluation on Test Set")
    print("=" * 40)
    try:
        print(f"Creating Test Dataset from: {PROCESSED_TEST_DATA_DIR}")
        test_dataset = create_npz_dataset(
            PROCESSED_TEST_DATA_DIR,
            BATCH_SIZE,
            is_training=False,
            cache_dataset=ENABLE_CACHING
        )
        print("Test dataset pipeline created.")

        print("\nEvaluating best model on test set...")
        if not model_to_evaluate.optimizer:
             print("Re-compiling model for evaluation (optimizer was missing)...")
             model_to_evaluate.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        test_loss, test_accuracy = model_to_evaluate.evaluate(test_dataset, verbose=1)
        print(f"\nTest Loss (Unscaled): {test_loss:.4f}")
        print(f"Test Accuracy (Unscaled): {test_accuracy:.4f}")

        # --- Test Report & Calibration Analysis on Test Set ---
        if PLOTTING_ENABLED:
             print("\nGenerating detailed test report and ECE analysis...")
             all_test_labels = []
             all_test_logits = []
             eval_logit_model = None

             # --- Get Logits from the best model for test predictions ---
             try:
                 print("Creating logit model from best model for test predictions...")
                 final_layer_eval = model_to_evaluate.layers[-1]
                 if isinstance(final_layer_eval, Dense) and final_layer_eval.activation == tf.keras.activations.softmax:
                     pre_softmax_output = model_to_evaluate.layers[-2].output
                     config = final_layer_eval.get_config(); config['activation'] = 'linear'; config['dtype'] = 'float32'
                     new_logit_layer_output = Dense.from_config(config)(pre_softmax_output)
                     eval_logit_model = Model(inputs=model_to_evaluate.input, outputs=new_logit_layer_output, name=f"{model_to_evaluate.name}_test_logits")
                     eval_logit_model.layers[-1].set_weights(final_layer_eval.get_weights())
                     print("Logit model for test evaluation created.")
                 else:
                     print(f"Warning: Could not automatically extract logits for test evaluation (last layer type: {type(final_layer_eval)}). ECE/scaling might be inaccurate.")
                     eval_logit_model = None

             except Exception as e:
                 print(f"Warning: Could not create logit model for test predictions: {e}. Skipping scaled analysis.")
                 eval_logit_model = None

             # --- Predict on test set ---
             temp_to_apply = TEMP_SCALE_OPTIMAL_T if APPLY_TEMP_SCALING else 1.0
             print(f"Applying Temperature T={temp_to_apply:.4f} for test set ECE/Reliability plot.")

             all_test_probs_scaled = []
             predict_model = eval_logit_model if eval_logit_model else model_to_evaluate
             if not predict_model:
                 print("Error: No model available for test prediction.")
                 raise RuntimeError("Cannot proceed with test prediction.")

             print(f"Predicting using: {'Logit Model' if eval_logit_model else 'Original Model (Softmax Output)'}")

             for x_batch, y_batch in tqdm(test_dataset, desc="Predicting Test Set", unit="batch", leave=False):
                  batch_preds = predict_model.predict_on_batch(x_batch)

                  if eval_logit_model:
                       scaled_logits = batch_preds / temp_to_apply
                       batch_probs = tf.nn.softmax(scaled_logits, axis=-1).numpy()
                  else:
                       if temp_to_apply != 1.0:
                           print("Warning: Cannot apply temperature scaling as logits are unavailable. Using T=1.0 for ECE.", file=sys.stderr)
                       batch_probs = batch_preds

                  all_test_labels.extend(y_batch.numpy())
                  all_test_probs_scaled.append(batch_probs)

             # --- Post-process predictions ---
             if not all_test_labels:
                 print("Error: No test labels collected after prediction.")
             elif not all_test_probs_scaled:
                 print("Error: No test probabilities collected after prediction.")
             else:
                 y_true_test = np.array(all_test_labels)
                 y_probs_test_scaled = np.concatenate(all_test_probs_scaled, axis=0)
                 y_pred_test = np.argmax(y_probs_test_scaled, axis=1)

                 # --- Generate Reports ---
                 min_len_test = len(y_true_test)
                 if len(y_pred_test) != min_len_test: y_pred_test = y_pred_test[:min_len_test]
                 if len(y_probs_test_scaled) != min_len_test: y_probs_test_scaled = y_probs_test_scaled[:min_len_test]

                 if min_len_test > 0:
                      print("\n--- Test Set Performance ---")
                      print("\nTest Confusion Matrix:")
                      cm = confusion_matrix(y_true_test, y_pred_test)
                      print(cm)

                      print("\nTest Classification Report:")
                      report = classification_report(y_true_test, y_pred_test, zero_division=0, digits=4)
                      print(report)

                      print("\n--- Test Set Calibration Analysis ---")
                      ece_test, acc_bins_test, conf_bins_test, counts_bins_test = calculate_ece(
                          y_true_test, y_probs_test_scaled, n_bins=NUM_BINS_ECE
                      )
                      print(f"  Expected Calibration Error (ECE) on Test Set (T={temp_to_apply:.4f}): {ece_test:.5f}")

                      plot_reliability_diagram(
                          acc_bins_test, conf_bins_test, counts_bins_test,
                          NUM_BINS_ECE,
                          title=f"Reliability Diagram Test Set (T={temp_to_apply:.4f})"
                      )
                 else:
                      print("No valid test predictions available to generate detailed report.")
        else:
             print("\nSkipping detailed test report and ECE analysis (sklearn/matplotlib not found).")

    except FileNotFoundError as e:
        print(f"\nError loading test data: {e}. Skipping test evaluation.")
    except Exception as e:
        print(f"\nAn error occurred during final testing: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()
    print("=" * 40)

    # --- Core ML Conversion ---
    if COREML_ENABLED and ct is not None:
        print("\nStarting Core ML Model Conversion (Hybrid V3)...")
        try:
             print(f"Loading model from {BEST_MODEL_SAVE_PATH} for CoreML conversion...")
             loaded_model_for_coreml = load_model(
                 BEST_MODEL_SAVE_PATH
             )
             print("Model loaded successfully from disk.")
             keras_model_to_convert = loaded_model_for_coreml

             coreml_input = ct.TensorType(name="input_layer", shape=(1, FFT_LENGTH, 1))

             print(f"Calling ct.convert with source='tensorflow' on loaded model...")
             mlmodel = ct.convert(
                 keras_model_to_convert,
                 inputs=[coreml_input],
                 compute_units=ct.ComputeUnit.ALL,
                 source="tensorflow"
             )
             mlmodel.save(COREML_MODEL_SAVE_PATH)
             print(f"Core ML model saved successfully to: {COREML_MODEL_SAVE_PATH}")

        except FileNotFoundError:
            print(f"Error during CoreML conversion: Could not find saved model file at {BEST_MODEL_SAVE_PATH}")
        except NameError as ne:
             print(f"Error during CoreML conversion (likely missing custom object definition): {ne}")
        except Exception as e:
            print(f"Error during Core ML conversion: {type(e).__name__} - {e}")
            import traceback
            traceback.print_exc()
    elif COREML_ENABLED and ct is None:
         print("\nSkipping Core ML Conversion: coremltools library loaded but 'ct' object is None.")
    else:
         print("\nSkipping Core ML Conversion (COREML_ENABLED is False or coremltools not found).")


    # --- Final Summary ---
    main_end_time = time.time()
    print("\n" + "=" * 50)
    print(f" Script Finished in {main_end_time - main_start_time:.2f} seconds. ")
    print(f" Best Model saved to: {BEST_MODEL_SAVE_PATH}")
    print(f" Optimal Temperature calculated (T): {TEMP_SCALE_OPTIMAL_T:.4f}")
    print(f" TensorBoard Logs: tensorboard --logdir {Path(TENSORBOARD_LOG_DIR).resolve()}")
    print("=" * 50)
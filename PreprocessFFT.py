import os
import glob
import time
import numpy as np
import pandas as pd
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
import joblib
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse
from pathlib import Path

# --- Configuration ---
FFT_FEATURES = True
FFT_LENGTH = 2000
APPLY_LOG_TRANSFORM = True

# --- FFT Function ---
def apply_fft(data):
    if data.shape[0] == 0:
        return np.array([], dtype=np.float32)
    fft_values = fft(data, axis=0)
    fft_len_half = (data.shape[0] + 1) // 2
    magnitudes = np.abs(fft_values)[:fft_len_half]
    return magnitudes.astype(np.float32)

# --- Function to Pad/Truncate FFT Features ---
def adjust_fft_length(fft_mag, target_len):
    current_len = fft_mag.shape[0]
    if current_len == target_len:
        return fft_mag
    elif current_len > target_len:
        return fft_mag[:target_len]
    else:
        padding = np.zeros((target_len - current_len,), dtype=np.float32)
        return np.concatenate([fft_mag, padding], axis=0)

# --- Function to Fit and Save Scaler on FFT Magnitudes ---
def fit_and_save_scaler_on_fft(input_dir, scaler_path, fft_length, apply_log=True):
    print(f"Starting Scaler Fitting on FFT Magnitudes from directory: {input_dir}")
    print(f"Target FFT Length: {fft_length}, Apply log1p transform: {apply_log}")
    scaler = StandardScaler()
    file_paths = sorted(list(Path(input_dir).glob("*.csv")))
    if not file_paths:
        raise ValueError(f"No CSV files found in directory: {input_dir}")

    print(f"Found {len(file_paths)} files for scaler fitting.")
    scaler_fit_start_time = time.time()
    processed_files_count = 0
    processed_cables_count = 0

    for file_path in tqdm(file_paths, desc="Fitting Scaler on FFT"):
        try:
            df_train = pd.read_csv(file_path)
            if df_train.shape[0] < 2: continue
            data_train = df_train.iloc[:-1, :].values.astype(np.float32)
            if data_train.ndim != 2 or data_train.shape[0] < 1: continue

            file_processed_flag = False
            for i in range(data_train.shape[1]):
                cable_series = data_train[:, i]
                if cable_series.shape[0] < 2: continue

                diff_series = np.diff(cable_series, prepend=cable_series[0])
                if diff_series.shape[0] == 0: continue

                fft_mag = apply_fft(diff_series)
                if fft_mag.shape[0] == 0: continue

                features = adjust_fft_length(fft_mag, fft_length)
                if features.shape[0] != fft_length: continue

                if apply_log:
                    features = np.log1p(features)

                scaler.partial_fit(features.reshape(-1, 1))
                processed_cables_count += 1
                file_processed_flag = True

            if file_processed_flag:
                processed_files_count += 1

        except pd.errors.EmptyDataError:
            print(f"\nWarning: Empty CSV file encountered during scaler fitting: {file_path}")
        except FileNotFoundError:
            print(f"\nWarning: File not found during scaler fitting: {file_path}")
        except Exception as e:
            print(f"\nWarning: Could not process {file_path} for scaler fitting: {type(e).__name__} - {e}")

    scaler_fit_end_time = time.time()

    if processed_files_count == 0 or processed_cables_count == 0:
         raise RuntimeError("Scaler could not be fitted. No valid files or cables processed.")
    if not hasattr(scaler, 'mean_') or not hasattr(scaler, 'scale_') or scaler.mean_ is None or scaler.scale_ is None:
         raise RuntimeError("Scaler fitting failed (attributes mean_ or scale_ missing or None).")

    print(f"\nScaler fitted on FFT magnitudes from {processed_files_count} files ({processed_cables_count} cables total) in {scaler_fit_end_time - scaler_fit_start_time:.2f} seconds.")
    print(f"Scaler mean: {scaler.mean_[0]:.4f}, scale: {scaler.scale_[0]:.4f}")

    print(f"Saving fitted scaler to: {scaler_path}")
    joblib.dump(scaler, scaler_path)
    print("Scaler saved successfully.")
    return scaler

# --- Function to Process a Single File ---
def process_single_file(args):
    input_file_path, output_dir_path, scaler, fft_length, apply_log = args
    base_filename = input_file_path.stem 
    output_npz_path = output_dir_path / f"processed_{base_filename}.npz"

    try:
        df = pd.read_csv(input_file_path)
        if df.shape[0] <= 2: return None

        labels = df.iloc[-1, :].values.astype(np.int32)
        data = df.iloc[:-1, :].values.astype(np.float32)
        if data.shape[0] < 1 or data.ndim != 2: return None

        num_cables = data.shape[1]
        processed_cable_data = []

        for i in range(num_cables):
            cable_series = data[:, i]
            if cable_series.shape[0] < 2: continue

            diff_series = np.diff(cable_series, prepend=cable_series[0])
            if diff_series.shape[0] == 0: continue

            fft_mag = apply_fft(diff_series)
            if fft_mag.shape[0] == 0: continue

            features = adjust_fft_length(fft_mag, fft_length)
            if features.shape[0] != fft_length: continue

            if apply_log:
                features = np.log1p(features)

            try:
                scaled_features = scaler.transform(features.reshape(-1, 1)).flatten()
            except NotFittedError:
                 print(f"ERROR: Scaler not fitted when processing {input_file_path.name}, cable {i}!")
                 return None
            except ValueError as ve:
                 print(f"ERROR: ValueError during transform for {input_file_path.name}, cable {i}: {ve}")
                 return None

            processed_cable_data.append(scaled_features.reshape(-1, 1))

        if not processed_cable_data:
            print(f"Warning: No valid cables processed for {input_file_path.name}")
            return None

        final_features = np.stack(processed_cable_data, axis=0).astype(np.float32)
        final_labels = labels

        np.savez_compressed(output_npz_path, features=final_features, labels=final_labels)
        return str(output_npz_path)

    except pd.errors.EmptyDataError:
        print(f"Warning: Empty data error for {input_file_path.name}")
        return None
    except FileNotFoundError:
        print(f"Warning: File not found {input_file_path.name}")
        return None
    except Exception as e:
        print(f"ERROR processing {input_file_path.name}: {type(e).__name__} - {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Offline Preprocessing Script (v2 - Scale FFT Mag)")
    parser.add_argument("--input_dir", type=str, default="training_data", help="Directory containing raw CSV files.")
    parser.add_argument("--output_dir", type=str, default="processed_data_fftscaled", help="Directory to save processed .npz files.")
    parser.add_argument("--scaler_file", type=str, default="fft_scaler.joblib", help="Path to save/load the fitted scaler (for FFT magnitudes).")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of worker processes for parallel processing.")
    parser.add_argument("--no_log_transform", action="store_true", help="Disable log1p transform before scaling FFT magnitudes.")

    args = parser.parse_args()

    INPUT_DIR = Path(args.input_dir)
    OUTPUT_DIR = Path(args.output_dir)
    SCALER_PATH = Path(args.scaler_file)
    MAX_WORKERS = args.workers
    APPLY_LOG_TRANSFORM = not args.no_log_transform

    print("=" * 60)
    print(" Starting Offline Preprocessing (v2 - Scaling FFT Magnitudes) ")
    print("=" * 60)
    print(f"Input directory:        {INPUT_DIR.resolve()}")
    print(f"Output directory:       {OUTPUT_DIR.resolve()}")
    print(f"Scaler path:            {SCALER_PATH.resolve()}")
    print(f"FFT Features:           {FFT_FEATURES}")
    print(f"FFT Length:             {FFT_LENGTH}")
    print(f"Apply log1p transform:  {APPLY_LOG_TRANSFORM}")
    print(f"Max Workers:            {MAX_WORKERS}")
    print("-" * 60)

    # --- Fit and Save Scaler on FFT Magnitudes ---
    if not SCALER_PATH.exists():
        try:
            fitted_scaler = fit_and_save_scaler_on_fft(INPUT_DIR, SCALER_PATH, FFT_LENGTH, APPLY_LOG_TRANSFORM)
        except Exception as e:
            print(f"\nFATAL ERROR during scaler fitting: {type(e).__name__} - {e}")
            exit(1)
    else:
        print(f"Loading existing FFT magnitude scaler from: {SCALER_PATH}")
        try:
            fitted_scaler = joblib.load(SCALER_PATH)
            print("Scaler loaded successfully.")
            if hasattr(fitted_scaler, 'mean_') and hasattr(fitted_scaler, 'scale_'):
                 print(f"(Scaler mean: {fitted_scaler.mean_[0]:.4f}, scale: {fitted_scaler.scale_[0]:.4f})")
            else:
                 print("Warning: Loaded scaler seems not fitted correctly. Refitting might be needed.")
                 print("Exiting due to potentially invalid loaded scaler.")
                 exit(1)
        except Exception as e:
            print(f"\nFATAL ERROR loading scaler: {type(e).__name__} - {e}")
            exit(1)


    # --- Prepare for Parallel Processing ---
    all_files = sorted(list(INPUT_DIR.glob("*.csv")))
    if not all_files:
        print(f"Error: No CSV files found in {INPUT_DIR} for processing.")
        exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ensured output directory exists: {OUTPUT_DIR}")

    tasks = [(file_path, OUTPUT_DIR, fitted_scaler, FFT_LENGTH, APPLY_LOG_TRANSFORM) for file_path in all_files]

    # --- Process Files in Parallel ---
    print(f"\nStarting parallel processing of {len(all_files)} files using up to {MAX_WORKERS} workers...")
    start_proc_time = time.time()
    success_count = 0
    fail_count = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_file, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Files"):
            result = future.result()
            if result is not None:
                success_count += 1
            else:
                fail_count += 1

    end_proc_time = time.time()
    print("\n" + "=" * 60)
    print(" Offline Preprocessing Complete ")
    print("=" * 60)
    print(f"Successfully processed: {success_count} files")
    print(f"Failed/Skipped:       {fail_count} files")
    print(f"Total processing time: {end_proc_time - start_proc_time:.2f} seconds")
    print(f"Processed data saved in: {OUTPUT_DIR.resolve()}")
    print("=" * 60)
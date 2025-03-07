import numpy as np
import mne
import matplotlib.pyplot as plt
from pathlib import Path
from utils.file_mgt import get_random_eeg_file_paths

# Step 1: Get all .fif file paths
paths = get_random_eeg_file_paths("fif", 5000)

# Storage lists
avg_amplitudes_per_file = []  # Stores avg amplitude per channel for each file
peak_amplitudes_per_file = []  # Stores max amplitude per channel for each file
retained_epochs = []  # Number of retained epochs per file
removed_epochs = []   # Number of removed epochs per file
channel_names = None  # Stores EEG channel names (initialized later)

# Set the epoch rejection threshold (110 μV → converted to Volts for MNE)
rejection_threshold = 110e-6  # MNE works in Volts

# Step 2: Process each file
for path in paths:
    try:
        # Load epochs
        epochs = mne.read_epochs(path, preload=True)

        if channel_names is None:
            channel_names = epochs.ch_names  # Store channel names from first file
        
        # Define rejection criteria
        reject_criteria = dict(eeg=rejection_threshold)  # Reject epochs > 110 μV

        # Store total epochs before rejection
        total_epochs_before = len(epochs)

        # Apply rejection
        epochs.drop_bad(reject=reject_criteria)

        # ✅ NEW CHECK: Skip processing if all epochs were removed
        total_epochs_after = len(epochs)
        epochs_removed = total_epochs_before - total_epochs_after

        if total_epochs_after == 0:
            print(f"⚠️ All epochs dropped for {path}. Skipping file.")
            removed_epochs.append(epochs_removed)
            retained_epochs.append(0)
            continue

        # Store retained and removed epochs
        retained_epochs.append(total_epochs_after)
        removed_epochs.append(epochs_removed)

        # Compute mean absolute amplitude per channel across all epochs and time points
        data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
        avg_amplitudes = np.mean(np.abs(data), axis=(0, 2))  # Shape: (n_channels,)
        avg_amplitudes_per_file.append(avg_amplitudes)

        # Compute peak (max) absolute amplitude per channel
        peak_amplitudes = np.max(np.abs(data), axis=(0, 2))  # Max across epochs and time
        peak_amplitudes_per_file.append(peak_amplitudes)

        # Save cleaned epochs
        cleaned_path = str(path).replace(".fif", "_clean-epo.fif")
        epochs.save(cleaned_path, overwrite=True)

    except Exception as e:
        print(f"⚠️ Failed to process {path}: {e}")

# Step 3: Convert to NumPy array for easier computation
avg_amplitudes_per_file = np.array(avg_amplitudes_per_file)  # Shape: (n_files, n_channels)
peak_amplitudes_per_file = np.array(peak_amplitudes_per_file)  # Shape: (n_files, n_channels)

# Step 4: Compute overall statistics
overall_avg_amplitudes = np.mean(avg_amplitudes_per_file, axis=0)  # Mean amplitude per channel
overall_peak_amplitudes = np.max(peak_amplitudes_per_file, axis=0)  # Peak amplitude per channel
peak_to_avg_ratio = overall_peak_amplitudes / overall_avg_amplitudes  # Ratio of peak to average

# Step 5: Compute multiple adaptive thresholds (95th-99th percentiles)
percentiles = [95, 96, 97, 98, 99]
adaptive_thresholds = {p: np.percentile(overall_avg_amplitudes, p) for p in percentiles}
adaptive_peak_thresholds = {p: np.percentile(overall_peak_amplitudes, p) for p in percentiles}

# Step 6: Compute standard deviation per channel
std_dev_per_channel = np.std(avg_amplitudes_per_file, axis=0)

# Step 7: Find outliers (Channels with extremely high peaks)
outlier_threshold = np.percentile(overall_peak_amplitudes, 99)  # Using 99th percentile as a cutoff
outliers = [ch for ch, peak in zip(channel_names, overall_peak_amplitudes) if peak > outlier_threshold]

# Step 8: Find the top 10 most variable channels (highest standard deviation)
top_variable_channels = sorted(zip(channel_names, std_dev_per_channel), key=lambda x: x[1], reverse=True)[:10]

# Step 9: Calculate percentage of time above certain thresholds
thresholds = [50e-6, 100e-6, 200e-6, 300e-6]  # 50, 100, 200, 300 μV
percentage_above_threshold = {
    t: np.mean(overall_peak_amplitudes > t) * 100 for t in thresholds
}

# Step 10: Compute rejection statistics
total_epochs = np.array(retained_epochs) + np.array(removed_epochs)
rejection_rate = (np.sum(removed_epochs) / np.sum(total_epochs)) * 100  # Percentage of epochs removed

# Step 11: Save Summary to File
summary_path = "eeg_analysis_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n**EEG Channel Amplitude Analysis** 📊\n")
    f.write(f"Files Processed: {len(paths)}\n")
    f.write(f"Total Epochs Retained: {np.sum(retained_epochs)}\n")
    f.write(f"Total Epochs Rejected: {np.sum(removed_epochs)}\n")
    f.write(f"Rejection Rate: {rejection_rate:.2f}% (Epochs exceeding {rejection_threshold * 1e6:.0f} μV were removed)\n\n")

    f.write("Adaptive Thresholds for Mean Amplitude:\n")
    for p, val in adaptive_thresholds.items():
        f.write(f"   - {p}th percentile: {val * 1e6:.2f} μV\n")

    f.write("\nAdaptive Thresholds for Peak Amplitude:\n")
    for p, val in adaptive_peak_thresholds.items():
        f.write(f"   - {p}th percentile: {val * 1e6:.2f} μV\n")

    f.write("\nPeak-to-Average Ratio per Channel (Top 10 highest ratios):\n")
    top_ratios = sorted(zip(channel_names, peak_to_avg_ratio), key=lambda x: x[1], reverse=True)[:10]
    for ch, ratio in top_ratios:
        f.write(f"   - {ch}: {ratio:.2f}x higher than average\n")

    f.write("\nChannels with Extreme Peaks (Above 99th Percentile):\n")
    f.write(f"   - {outliers}\n")

    f.write("\nTop 10 Most Variable Channels (Highest Standard Deviation):\n")
    for ch, std in top_variable_channels:
        f.write(f"   - {ch}: {std * 1e6:.2f} μV\n")

    f.write("\nPercentage of Time EEG Amplitude Exceeds Thresholds:\n")
    for t, pct in percentage_above_threshold.items():
        f.write(f"   - Above {t * 1e6:.0f} μV: {pct:.2f}% of the time\n")

# Print Summary Location
print(f"\n📄 EEG analysis summary saved to: {summary_path}")

# Step 12: Plot Histogram of Peak Amplitudes
plt.figure(figsize=(8, 5))
plt.hist(overall_peak_amplitudes * 1e6, bins=30, color='red', alpha=0.7, edgecolor='black')
plt.axvline(100, color='blue', linestyle="--", label="100 μV")
plt.axvline(200, color='orange', linestyle="--", label="200 μV")
plt.axvline(300, color='red', linestyle="--", label="300 μV")
plt.xlabel("Peak Amplitude (μV)")
plt.ylabel("Frequency")
plt.title("Distribution of EEG Peak Amplitudes")
plt.legend()
plt.grid()
plt.show()

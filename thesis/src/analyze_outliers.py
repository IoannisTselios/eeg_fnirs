import os
import pandas as pd
import mne
import matplotlib.pyplot as plt
from pathlib import Path

# Path to the root directory containing the results
root_dir = Path(r'L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\results\\run_20250404_1629')
fif_path = Path(r'L:\\LovbeskyttetMapper\\CONNECT-ME\\CONMED3\\Dataoptagelser\\NIRS-EEG')
# Output directories for CSV and plots
csv_output_dir = root_dir / "csv_outputs"
plot_output_dir = root_dir / "plots"
csv_output_dir.mkdir(parents=True, exist_ok=True)
plot_output_dir.mkdir(parents=True, exist_ok=True)

# List to collect data from all CSV files
all_outliers = []

# Define post-administration variants
post_admin_variants = {
    0: ["Baseline"],
    1: ["Post-administration 1", "Post administration 1"],
    2: ["Post-administration 2", "Post administration 2"]
}

# Recursively search for 'eeg_outliers_detail_df.csv' in all subdirectories
for root, dirs, files in os.walk(root_dir):
    if 'feature_extraction_files' in root and 'eeg_outliers_detail_df.csv' in files:
        file_path = Path(root) / 'eeg_outliers_detail_df.csv'
        df = pd.read_csv(file_path)
        df['source_file'] = str(file_path)
        all_outliers.append(df)

# Combine all outliers into a single DataFrame
if all_outliers:
    outliers_df = pd.concat(all_outliers, ignore_index=True)
else:
    outliers_df = pd.DataFrame()

if not outliers_df.empty:
    for _, row in outliers_df.iterrows():
        patient_id = str(row["id"]).strip()
        session = str(row["drug"]).strip()

        try:
            time_condition = int(row['time'])
        except ValueError:
            continue

        session_map = {"1": "Session 1", "2": "Session 2", "3": "Session 3"}
        session_folder = session_map.get(session, session)

        patient_path = fif_path / Path(patient_id) / Path(session_folder)
        outlier_folders = post_admin_variants.get(time_condition, ["UNKNOWN"])
        outlier_paths = [patient_path / folder for folder in outlier_folders]

        outlier_fif_files = []
        for outlier_path in outlier_paths:
            outlier_fif_files.extend(list(outlier_path.rglob("*.fif")))

        if not outlier_fif_files:
            continue

        outlier_fif = outlier_fif_files[0]

        try:
            # 👉 Load epochs
            epochs_outlier = mne.read_epochs(outlier_fif, preload=True)

            # 👉 Compute PSD (averaged across epochs)
            psd_data, freqs = epochs_outlier.compute_psd().get_data(return_freqs=True)

            # 👉 Average across epochs → shape becomes (n_channels, n_freqs)
            psd_mean = psd_data.mean(axis=0)  # shape: (n_channels, n_freqs)

            # 👉 Create DataFrame (transpose to match frequency order)
            psd_df = pd.DataFrame(psd_mean.T, index=freqs, columns=epochs_outlier.ch_names)

            # 👉 Total PSD Power (sum of power across all channels and freqs)
            total_psd_power = psd_mean.sum()

            # 👉 Power in 20–50 Hz band
            band_mask = (freqs >= 20) & (freqs <= 50)
            band_power = psd_mean[:, band_mask].sum()

            # 👉 Flagging based on thresholds
            flagged = []
            if band_power > 1e-10:
                flagged.append("High muscle artifact")
            if total_psd_power < 1e-12:
                flagged.append("Low total PSD power")

            # 👉 Add flag to DataFrame
            psd_df['Flag'] = ', '.join(flagged) if flagged else "OK"

            # 👉 Save PSD values to CSV
            psd_csv_path = csv_output_dir / f"EEG_PSD_Values_{patient_id}_{session}_Admin{time_condition}.csv"
            psd_df.to_csv(psd_csv_path)
            print(f"✅ Saved PSD data to {psd_csv_path}")

            # 👉 Save PSD plot
            psd_fig = epochs_outlier.plot_psd(show=False)
            psd_path = plot_output_dir / f"EEG_Outlier_PSD_{patient_id}_{session}_Admin{time_condition}.png"
            psd_fig.savefig(psd_path)
            plt.close()
            print(f"✅ Saved PSD plot to {psd_path}")

            # 👉 Log results
            print(f"Total PSD Power: {total_psd_power:.4e}")
            print(f"Power in 20–50 Hz: {band_power:.4e}")
            if flagged:
                print(f"🚨 Flagged: {', '.join(flagged)}")
            else:
                print("✅ No issues detected.")

        except Exception as e:
            print(f"❌ Error processing {patient_id}: {e}")

from pathlib import Path
import os
import pandas as pd
import mne
import matplotlib.pyplot as plt

# Define paths
root_path = Path("L:\\LovbeskyttetMapper\\CONNECT-ME\\CONMED3\\Dataoptagelser\\NIRS-EEG\\")
outlier_csv_path = "L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\feature_extraction_files\\eeg_outliers_detail_df.csv"
output_dir = Path("L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\plots\\EEG_Outlier_Segments")
output_dir.mkdir(parents=True, exist_ok=True)

# Load outlier details
df_outliers = pd.read_csv(outlier_csv_path)

# Process each outlier
for _, row in df_outliers.iterrows():
    patient_id = row["id"]
    session = row["drug"]  # e.g., "Session 1"
    time_condition = row["time"]  # 0 = Baseline, 1 = Post Admin 1, 2 = Post Admin 2
    feature = row["Feature"]

    # ---- Construct Paths ----
    patient_path = root_path / patient_id / session
    baseline_path = patient_path / "Baseline"
    
    # Handle both naming variations of "Post-administration"
    post_admin_variants = {
        0: ["Baseline"],
        1: ["Post-administration 1", "Post administration 1"],
        2: ["Post-administration 2", "Post administration 2"]
    }
    
    outlier_folders = post_admin_variants.get(time_condition, ["UNKNOWN"])
    outlier_paths = [patient_path / folder for folder in outlier_folders]

    # 🔍 Debugging: Print paths to check correctness
    print(f"🔍 Searching for FIF files in:")
    print(f"   - Baseline Path: {baseline_path}")
    print(f"   - Outlier Paths: {outlier_paths}")

    # Find `.fif` files in baseline & outlier directories
    baseline_fif_files = list(baseline_path.rglob("*.fif"))

    # Search for `.fif` files in both possible "Post-administration" folder variations
    outlier_fif_files = []
    for outlier_path in outlier_paths:
        outlier_fif_files.extend(list(outlier_path.rglob("*.fif")))

    # 🔍 Debugging: Print all `.fif` files found
    print(f"   - Baseline FIFs: {baseline_fif_files}")
    print(f"   - Outlier FIFs: {outlier_fif_files}")

    if not baseline_fif_files or not outlier_fif_files:
        print(f"⚠️ No `.fif` files found for {patient_id} in {session}, skipping.")
        continue

    # Select first available `.fif` file
    baseline_fif = baseline_fif_files[0]
    outlier_fif = outlier_fif_files[0]

    print(f"📌 Processing Patient {patient_id}, {session}, Post Admin {time_condition} ({feature})...")

    try:
        # Load EEG data
        epochs_baseline = mne.read_epochs(baseline_fif, preload=True)
        epochs_outlier = mne.read_epochs(outlier_fif, preload=True)

        # # Convert outlier time to seconds
        # sfreq = epochs_outlier.info["sfreq"]
        # outlier_time = row["time"]  # This should be in seconds already

        # # Define segment duration (±2.5s around the outlier)
        # segment_start = max(outlier_time - 2.5, 0)
        # segment_end = min(outlier_time + 2.5, epochs_outlier.times[-1])

        # # Crop EEG to focus on outlier segment
        # epochs_outlier.crop(tmin=segment_start, tmax=segment_end)
        # epochs_baseline.crop(tmin=segment_start, tmax=segment_end)

        # ---- Plot EEG Signals ----
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        # Plot Baseline EEG
        epochs_baseline.average().plot(spatial_colors=True, show=False, axes=axes[0])
        axes[0].set_title(f"Baseline EEG - Patient {patient_id}, {session}")

        # Plot Outlier EEG
        epochs_outlier.average().plot(spatial_colors=True, show=False, axes=axes[1])
        axes[1].set_title(f"Outlier EEG - Patient {patient_id}, {session}, Post Admin {time_condition}")

        # Save figure
        save_path = output_dir / f"EEG_Outlier_vs_Baseline_{patient_id}_{session}_Admin{time_condition}.png"
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"✅ Saved EEG outlier segment for {patient_id} at {save_path}")

    except Exception as e:
        print(f"❌ Error processing files for {patient_id}: {e}")
        continue

print("📊 All EEG outlier segments processed and saved.")

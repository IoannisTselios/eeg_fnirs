import logging
from pathlib import Path

from matplotlib import pyplot as plt
import mne
import pandas as pd


df_outliers = pd.read_csv("L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\results\\run_20250305_1241\\feature_extraction_files\\eeg_outliers_detail_df.csv")
        
# Ensure `root_path` and `output_dir` are `Path` objects
root_path = Path("L:\\LovbeskyttetMapper\\CONNECT-ME\\CONMED3\\Dataoptagelser\\NIRS-EEG\\")

for _, row in df_outliers.iterrows():
    patient_id = str(row["id"]).strip()  # Convert to string & remove spaces
    session = str(row["drug"]).strip()  # Convert to string & remove spaces
    time_condition = row["time"]
    feature = row["Feature"]

    # ✅ Ensure patient_id and session are `Path` objects
    patient_path = root_path / Path(patient_id) / Path(session)

    post_admin_variants = {
        0: ["Baseline"],
        1: ["Post-administration 1", "Post administration 1"],
        2: ["Post-administration 2", "Post administration 2"]
    }
    outlier_folders = post_admin_variants.get(time_condition, ["UNKNOWN"])
    outlier_paths = [patient_path / Path(folder) for folder in outlier_folders]  # Ensure these are Paths

    # 🔍 Find `.fif` files
    outlier_fif_files = []
    for outlier_path in outlier_paths:
        outlier_fif_files.extend(list(outlier_path.rglob("*.fif")))

    outlier_fif = outlier_fif_files[0]

    try:
        epochs_outlier = mne.read_epochs(outlier_fif, preload=True)
        epochs_outlier.plot_psd()
        plt.show(block=True)


    except Exception as e:
        logging.error(f"❌ Error processing files for {patient_id}: {e}")
        continue
import logging
import os
from pathlib import Path
import pandas as pd
import mne
import matplotlib.pyplot as plt

# ---- EEG Outlier Analysis ----
class EEGOutlierAnalyzer:

    def __init__(self, root_path, outlier_csv_path, output_dir):
        self.root_path = root_path
        self.outlier_csv_path = outlier_csv_path
        self.output_dir = output_dir
        logging.info("EEG Outlier Analyzer Initialized")

    def process_outliers(self):
        df_outliers = pd.read_csv(self.outlier_csv_path)
        
        # Ensure `root_path` and `output_dir` are `Path` objects
        root_path = Path(self.root_path)
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

        for _, row in df_outliers.iterrows():
            patient_id = str(row["id"]).strip()  # Convert to string & remove spaces
            session = str(row["drug"]).strip()  # Convert to string & remove spaces
            time_condition = row["time"]
            feature = row["Feature"]

            # ✅ Ensure patient_id and session are `Path` objects
            patient_path = root_path / Path(patient_id) / Path(session)
            baseline_path = patient_path / Path("Baseline")

            post_admin_variants = {
                0: ["Baseline"],
                1: ["Post-administration 1", "Post administration 1"],
                2: ["Post-administration 2", "Post administration 2"]
            }
            outlier_folders = post_admin_variants.get(time_condition, ["UNKNOWN"])
            outlier_paths = [patient_path / Path(folder) for folder in outlier_folders]  # Ensure these are Paths

            # 🔍 Find `.fif` files
            baseline_fif_files = list(baseline_path.rglob("*.fif"))
            outlier_fif_files = []
            for outlier_path in outlier_paths:
                outlier_fif_files.extend(list(outlier_path.rglob("*.fif")))

            if not baseline_fif_files or not outlier_fif_files:
                logging.warning(f"⚠️ No `.fif` files found for {patient_id} in {session}, skipping.")
                continue

            baseline_fif = baseline_fif_files[0]
            outlier_fif = outlier_fif_files[0]

            try:
                epochs_baseline = mne.read_epochs(baseline_fif, preload=True)
                epochs_outlier = mne.read_epochs(outlier_fif, preload=True)

                fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

                epochs_baseline.average().plot(spatial_colors=True, show=False, axes=axes[0])
                axes[0].set_title(f"Baseline EEG - Patient {patient_id}, {session}")

                epochs_outlier.average().plot(spatial_colors=True, show=False, axes=axes[1])
                axes[1].set_title(f"Outlier EEG - Patient {patient_id}, {session}, Post Admin {time_condition}")

                # ✅ Ensure `save_path` is a `Path` object
                save_path = output_dir / f"EEG_Outlier_vs_Baseline_{patient_id}_{session}_Admin{time_condition}.png"
                plt.tight_layout()
                plt.savefig(save_path)
                plt.close()

                logging.info(f"✅ Saved EEG outlier segment for {patient_id} at {save_path}")
            except Exception as e:
                logging.error(f"❌ Error processing files for {patient_id}: {e}")
                continue



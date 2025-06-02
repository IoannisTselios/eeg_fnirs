import logging
from pathlib import Path
import pandas as pd
import mne
import matplotlib.pyplot as plt

# ---- EEG Outlier Analysis for Dataset 2 ----
class EEGOutlierAnalyzer:

    def __init__(self, root_path, outlier_csv_path, output_dir):
        self.root_path = Path(root_path)
        self.outlier_csv_path = Path(outlier_csv_path)
        self.output_dir = Path(output_dir)
        logging.info("EEG Outlier Analyzer Initialized for Dataset 2")

    def process_outliers(self):
        df_outliers = pd.read_csv(self.outlier_csv_path)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for _, row in df_outliers.iterrows():
            patient_id = str(row["id"]).strip()
            feature = row["Feature"]

            # Find the .fif file directly
            fif_file = list((self.root_path).rglob(f"{patient_id}.fif"))
            if not fif_file:
                logging.warning(f"⚠️ No `.fif` file found for {patient_id}, skipping.")
                continue

            fif_file = fif_file[0]  # Take the first match
            try:
                epochs = mne.read_epochs(fif_file, preload=True)

                # Average the epochs
                fig = epochs.average().plot(spatial_colors=True, show=False)
                plt.title(f"EEG Average - Patient {patient_id} (Outlier Feature: {feature})")

                save_path = self.output_dir / f"EEG_Outlier_{patient_id}_{feature}.png"
                fig.savefig(save_path)
                plt.close()

                logging.info(f"✅ Saved EEG average for outlier patient {patient_id} at {save_path}")

            except Exception as e:
                logging.error(f"❌ Error processing {patient_id}: {e}")
                continue

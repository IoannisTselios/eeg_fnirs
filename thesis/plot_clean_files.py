import glob
import logging
import os
from pathlib import Path
import mne
from tqdm import tqdm
import pandas as pd

logging.basicConfig(force=True, format="%(levelname)s - %(name)s - %(message)s", level=logging.INFO)
mne.set_log_level("WARNING")

def resolve_outlier_file_paths(df, extension=".fif"):
    time_mapping = {
        "0": ["Baseline"],
        "1": ["Post administration 1", "Post-administration 1"],
        "2": ["Post administration 2", "Post-administration 2"]
    }

    base_dir = Path("L:/LovbeskyttetMapper/CONNECT-ME/Ioannis/thesis_code/results/run_20250403_1902/preprocessed_files")
    paths = []

    for idx, row in df.iterrows():
        patient_id = str(row["id"])
        drug = str(row["drug"])
        time_key = str(row["time"])
        matched_file = None

        for time_variant in time_mapping.get(time_key, []):
            file_name = f"{patient_id}_{drug}_{time_variant}_preprocessed_raw.fif"
            full_path = base_dir / file_name
            if full_path.exists():
                matched_file = full_path
                break

        paths.append(str(matched_file) if matched_file else None)

    df["resolved_file_path"] = paths
    return df

# ✅ Load the metadata CSV (or replace with your own DataFrame)
df = pd.read_csv("L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\results\\run_20250403_1902\\feature_extraction_files\eeg_outliers_detail_df.csv")  # change to your actual CSV
df = resolve_outlier_file_paths(df)

# ✅ Loop through resolved file paths
for file in tqdm(df["resolved_file_path"].dropna(), desc="Processing Resolved Files"):
    try:
        logging.info(f"🔎 Loading file: {file}")
        raw = mne.io.read_raw_fif(file, preload=True)

        # 🧠 Visualize
        raw.plot_psd(fmin=1, fmax=40, spatial_colors=True)
        raw.plot(block=True)

    except ValueError as e:
        logging.error(f"❌ Value error while processing {file}: {e}")
        continue
    except Exception as e:
        logging.error(f"❌ Failed to process {file}: {e}")
        continue

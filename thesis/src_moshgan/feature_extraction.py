import os
import glob
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import mne
from tqdm import tqdm
from scipy.signal import welch

class EEGFeatureExtractor:
    def __init__(self, SOURCE_FOLDER):
        self.SOURCE_FOLDER = SOURCE_FOLDER
        logging.info("EEG Preprocessor Initialized")

    def compute_bandpower(self, data, sf, band):
        freqs, psd = welch(data, sf, nperseg=2048)
        idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
        return np.trapz(psd[:, idx_band], freqs[idx_band], axis=1)

    def extract_features(self, feature_output_dir):
        mne.set_log_level("WARNING")
        logging.basicConfig(force=True, format='%(levelname)s - %(name)s - %(message)s', level=logging.INFO)

        fif_files = glob.glob(os.path.join(self.SOURCE_FOLDER, "*.fif"))
        if not fif_files:
            logging.error("❌ No FIF files found.")
            return

        bands = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30)
        }

        feature_rows = []

        for file in tqdm(fif_files, desc="Processing Files"):
            logging.info(f"Current file is {file}")
            file_name = Path(file).stem
            epochs = mne.read_epochs(file, preload=True)
            print(f"Event IDs for {file_name}: {list(epochs.event_id.keys())}")

            sfreq = epochs.info["sfreq"]
            ch_names = epochs.info["ch_names"]

            # Segment epochs into resting and stimulus based on annotations
            rest_epochs = []
            stim_epochs = []

            for annot in epochs.event_id.keys():
                if "rest" in annot.lower():
                    rest_epochs.append(epochs[annot])
                else:
                    stim_epochs.append(epochs[annot])

            # === Remove empty epochs before concatenation ===
            stim_epochs = [ep for ep in stim_epochs if len(ep) > 0]

            if stim_epochs:
                stim_epochs = mne.concatenate_epochs(stim_epochs)
            else:
                # === Use raw from the Epochs object ===
                try:
                    raw = epochs._raw
                    print(f"\n📉 Plotting raw data for: {file}")
                    raw.plot(n_channels=32, duration=30.0, scalings='auto', title=f"RAW: {Path(file).name}")
                except Exception as e:
                    logging.error(f"❌ Failed to plot raw from Epochs for {file}: {e}")
                
                logging.warning(f"⚠️ No non-empty stimulus epochs in {file}. Skipping.")
                continue

            # Combine all resting epochs
            rest_epochs = [ep for ep in rest_epochs if len(ep) > 0]
            if rest_epochs:
                rest_epochs = mne.concatenate_epochs(rest_epochs)
            else:
                rest_epochs = None

            for label, epoch_set in [("resting", rest_epochs), ("stimulus", stim_epochs)]:
                if epoch_set is None:
                    continue

                data = epoch_set.get_data()  # shape (n_epochs, n_channels, n_times)

                for ep_idx in range(data.shape[0]):
                    row = {"id": file_name, "type": label, "epoch": ep_idx}
                    for band_name, band_range in bands.items():
                        bp = self.compute_bandpower(data[ep_idx], sfreq, band_range)
                        for ch_idx, ch in enumerate(ch_names):
                            row[f"{band_name}_{ch}"] = bp[ch_idx]
                    feature_rows.append(row)

        df = pd.DataFrame(feature_rows)
        Path(feature_output_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(feature_output_dir) / "eeg_features.csv", index=False)
        return df

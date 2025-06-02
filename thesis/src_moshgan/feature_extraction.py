import os
import glob
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import mne
from tqdm import tqdm
from scipy.integrate import simpson
from scipy.signal import welch

class EEGFeatureExtractor:
    def __init__(self, SOURCE_FOLDER):
        self.SOURCE_FOLDER = SOURCE_FOLDER
        logging.info("EEG Preprocessor Initialized")

    def bandpower_mne(self, epochs_data, sf, bands):
        bandpower_dict = {}
        for low, high, label in bands:
            band_power = []
            for channel_data in epochs_data:
                band_power.append(self.bandpower(channel_data, sf, [low, high]))
            bandpower_dict[label] = np.mean(band_power)
        return bandpower_dict

    def bandpower(self, data, sf, band, window_sec=None):
        band = np.asarray(band, dtype=float)
        low, high = band

        if window_sec is None:
            window_sec = 4

        nperseg = min(int(window_sec * sf), len(data))
        noverlap = nperseg // 2

        freqs, psd = welch(data, sf, nperseg=nperseg, noverlap=noverlap, detrend=False, scaling='density')
        freq_res = np.mean(np.diff(freqs))
        idx_band = np.logical_and(freqs >= low, freqs <= high)

        if not np.any(idx_band):
            raise ValueError(f"No frequency components found in the range {low}-{high} Hz.")

        return simpson(psd[idx_band], dx=freq_res)

    def compute_brain_wave_band_power(self, epochs: mne.Epochs):
        delta_power = 0
        theta_power = 0
        epochs_data = epochs.get_data(copy=False)

        for epoch_id in range(epochs_data.shape[0]):
            df = self.bandpower_mne(epochs_data[epoch_id] * 1e6,
                                    sf=float(epochs._raw_sfreq[0]),
                                    bands=[(0.5, 4, "Delta"), (4, 8, "Theta")])
            delta_power += np.mean(df['Delta'])
            theta_power += np.mean(df['Theta'])

        delta_power /= epochs_data.shape[0]
        theta_power /= epochs_data.shape[0]

        return (delta_power, theta_power)

    def extract_features(self, feature_output_dir):
        mne.set_log_level("WARNING")
        logging.basicConfig(force=True, format='%(levelname)s - %(name)s - %(message)s', level=logging.INFO)

        fif_files = glob.glob(os.path.join(self.SOURCE_FOLDER, "*.fif"))
        if not fif_files:
            logging.error("❌ No FIF files found.")
            return

        feature_list = []

        for file in tqdm(fif_files, desc="Processing Files"):
            logging.info("Current file is {}".format(file))
            features = []
            file_name = Path(file).stem
            features.append(file_name)

            epochs = mne.read_epochs(file)

            try:
                epochs_familiar = epochs["Familiar voice"]
                fam_count = epochs_familiar.selection.shape[0]
            except KeyError:
                epochs_familiar, fam_count = None, 0

            try:
                epochs_medical = epochs["Medical voice"]
                med_count = epochs_medical.selection.shape[0]
            except KeyError:
                epochs_medical, med_count = None, 0

            del epochs

            if fam_count == 0 and med_count == 0:
                continue

            delta_fam, theta_fam = (np.nan, np.nan)
            delta_med, theta_med = (np.nan, np.nan)

            if fam_count > 0:
                delta_fam, theta_fam = self.compute_brain_wave_band_power(epochs_familiar)

            if med_count > 0:
                delta_med, theta_med = self.compute_brain_wave_band_power(epochs_medical)

            delta_diff = delta_fam - delta_med if not np.isnan(delta_fam) and not np.isnan(delta_med) else np.nan
            theta_diff = theta_fam - theta_med if not np.isnan(theta_fam) and not np.isnan(theta_med) else np.nan

            features += [delta_fam, theta_fam, delta_med, theta_med, delta_diff, theta_diff]
            feature_list.append(features)

        df = pd.DataFrame(feature_list, columns=[
            'id', 'delta_familiar', 'theta_familiar',
            'delta_medical', 'theta_medical',
            'delta_diff', 'theta_diff'
        ])
        df.to_csv(Path(feature_output_dir) / "eeg_features.csv", index=False)
        return df

# Dummy path to show usage; replace with actual path when using the class
# extractor = EEGFeatureExtractor("path_to_fif_files")
# extractor.extract_features("output_dir")


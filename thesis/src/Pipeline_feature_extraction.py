import logging
import pandas as pd
import mne
import numpy as np
from tqdm import tqdm
from scipy.integrate import simpson
from scipy.signal import welch
from utils.file_mgt import get_random_eeg_file_paths

# ---- EEG Feature Extraction ----
class EEGFeatureExtractor:
    
    def bandpower_mne(self, epochs_data, sf, bands, ch_names=None):
        bandpower_dict = {}

        # Loop over the frequency bands and compute the bandpower for each band
        for low, high, label in bands:
            band_power = []

            # Loop over each channel
            for channel_data in epochs_data:
                # Compute the bandpower for this channel
                band_power.append(self.bandpower(channel_data, sf, [low, high], window_sec=None))

            # Store the average power for the band
            bandpower_dict[label] = np.mean(band_power)

        return bandpower_dict

    def bandpower(self, data, sf, band, window_sec=None):
        """Compute the average power of the signal in a specific frequency band using Welch's method, 
        ensuring it matches YASA's implementation exactly."""

        band = np.asarray(band, dtype=float)
        low, high = band

        if window_sec is None:
            window_sec = 4  # Ensures a consistent window length

        if not isinstance(window_sec, (int, float)) or window_sec <= 0:
            raise ValueError(f"Invalid window_sec value: {window_sec}")

        nperseg = min(int(window_sec * sf), len(data))  # Match YASA's segment length
        noverlap = nperseg // 2  # Match YASA's default overlap

        # Compute the Power Spectral Density (PSD) using Welch’s method
        freqs, psd = welch(data, sf, nperseg=nperseg, noverlap=noverlap, detrend=False, scaling='density')

        # Compute frequency resolution exactly as YASA does
        freq_res = np.mean(np.diff(freqs))  # Mean frequency resolution

        # Find the indices of the frequencies within the desired band
        idx_band = np.logical_and(freqs >= low, freqs <= high)

        if not np.any(idx_band):  # Ensure we have valid frequency indices
            raise ValueError(f"No frequency components found in the range {low}-{high} Hz.")

        # Compute band power using TRAPEZOIDAL rule (to match YASA)
        bp = simpson(psd[idx_band], dx=freq_res)

        return bp

    def compute_brain_wave_band_power(self, epochs: mne.Epochs) -> tuple[float, float]:
        """
        Computes the relative band power averaged across channels and epochs, for Delta, Theta, and Alpha frequency bands.
        """
        delta_power = 0
        theta_power = 0

        epochs_data = epochs.get_data()

        for epoch_id in range(epochs_data.shape[0]):
            
            df = self.bandpower_mne(epochs_data[epoch_id] * 1e6,
                            sf = float(epochs._raw_sfreq[0]),
                            bands = [(0.5, 4, "Delta"), (4, 8, "Theta")],
                            ch_names = epochs.ch_names
                            )

            delta_power += np.mean(df['Delta'])
            theta_power += np.mean(df['Theta'])
            del df

        delta_power /= epochs_data.shape[0]
        theta_power /= epochs_data.shape[0]

        return (delta_power, theta_power)

    def extract_features(self, feature_output_dir):
        mne.set_log_level("WARNING")
        logging.basicConfig(force=True, format='%(levelname)s - %(name)s - %(message)s', level=logging.INFO)

        paths = list()
        paths = get_random_eeg_file_paths("fif", 5000)

        feature_list = []

        for path in tqdm(paths):
            logging.info("Current file is {}".format(path))

            try:
                # ---- Recording info ----
                patient_id = path.parts[6]
                drug = path.parts[7]
                time = 0 if path.parts[8][-1] == "e" else int(path.parts[8][-1])

                # ---- Load all epochs ----
                epochs = mne.read_epochs(path, preload=True)

                # ---- Get data for each epoch ----
                sfreq = float(epochs.info["sfreq"])
                epochs_data = epochs.get_data() * 1e6  # Convert to uV

                # ---- Define fronto-central channels ----
                fc_channels = ["AFF1h", "AF7", "AFF5h", "AFF6h", "AF8", "FC5", "FC3", "FCC3h", "FCC4h", "FFC2h", "FCC2h", "CCP3h", "CCP4h"]

                # ---- Loop over all epochs ----
                for epoch_idx, epoch_data in enumerate(epochs_data):
                    features = [
                        patient_id,
                        drug,
                        time,
                        epoch_idx
                    ]

                    # Compute delta & theta power per channel
                    theta_vals = []
                    for ch_name in fc_channels:
                        if ch_name not in epochs.ch_names:
                            features += [np.nan, np.nan]
                            continue

                        ch_idx = epochs.ch_names.index(ch_name)
                        ch_data = epoch_data[ch_idx]

                        delta_power = self.bandpower(ch_data, sfreq, [0.5, 4], window_sec=4)
                        theta_power = self.bandpower(ch_data, sfreq, [4, 8], window_sec=4)

                        features += [delta_power, theta_power]
                        theta_vals.append(theta_power)  # Collect theta for FC mean

                    # --- ADD delta_overall and theta_fc_mean ---
                    # delta_overall: mean across ALL channels in the epoch
                    delta_vals_all = []
                    for ch_idx in range(epoch_data.shape[0]):
                        ch_data = epoch_data[ch_idx]
                        delta = self.bandpower(ch_data, sfreq, [0.5, 4], window_sec=4)
                        delta_vals_all.append(delta)
                    delta_overall = np.nanmean(delta_vals_all)

                    # theta_fc_mean: mean theta of FC channels only (already collected above)
                    theta_fc_mean = np.nanmean(theta_vals) if theta_vals else np.nan

                    features += [delta_overall, theta_fc_mean]  # <--- ADDED

                    feature_list.append(features)

            except Exception as e:
                logging.error(f"⚠️ Failed to process file {path.name}: {e}")
                continue

        # ---- Save to CSV ----
        columns = ["id", "drug", "time", "epoch_idx"]
        for ch in fc_channels:
            columns.append(f"delta_{ch}")
            columns.append(f"theta_{ch}")
        columns.extend(["delta_overall", "theta_fc_mean"])
        df = pd.DataFrame(feature_list, columns=columns)
        df.to_csv(feature_output_dir / "eeg_features.csv", index=False)

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

        epochs_data = epochs.get_data(copy=False)

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
            features = []

            # ---- Recording info ----
            features.append(path.parts[6]) # Patient ID
            features.append(path.parts[7]) # Drug
            features.append(0 if path.parts[8][-1] == "e" else int(path.parts[8][-1])) # Time of recording

            # ---- Epoch ----

            # ---- Epoch Extraction: Include all epochs after first target annotation ----
            # ---- Epoch Extraction: Include all epochs after first target annotation ----
            epochs = mne.read_epochs(path)

            # Find first annotation (either moderate or hard)
            first_annot_idx = None
            for idx, annot in enumerate(epochs.event_id.keys()):
                if "Mental arithmetics moderate" in annot or "Mental arithmetics hard" in annot:
                    first_annot_idx = idx
                    break

            if first_annot_idx is None:
                logging.warning(f"⚠️ No 'Mental arithmetics' annotations in {path}. Skipping.")
                continue

            # Get time index of first relevant event
            first_event_time = epochs.events[first_annot_idx, 0] / epochs.info['sfreq']

            # Select epochs occurring after this time
            onsets = np.array([e[0] / epochs.info['sfreq'] for e in epochs.events])
            valid_epochs_mask = onsets >= first_event_time
            epochs_selected = epochs[valid_epochs_mask]

            if len(epochs_selected) == 0:
                logging.warning(f"⚠️ No epochs after first relevant annotation in {path}. Skipping.")
                continue

            # ---- Crop to first 25 seconds of each epoch
            epochs_selected.crop(tmin=0, tmax=25)

            # ---- Get data for each epoch
            sfreq = float(epochs_selected.info["sfreq"])
            epochs_data = epochs_selected.get_data(copy=False) * 1e6  # Convert to uV

            # ---- Define fronto-central channels ----
            fc_channels = ["AFF1h", "AF7", "AFF5h", "AFF6h", "AF8", "FC5", "FC3", "FCC3h", "FCC4h", "FFC2h", "FCC2h", "CCP3h", "CCP4h"]

            # ---- Loop over epochs to compute delta & theta per channel ----
            for epoch_idx, epoch_data in enumerate(epochs_data):
                features = [
                    path.parts[6],  # Patient ID
                    path.parts[7],  # Drug
                    0 if path.parts[8][-1] == "e" else int(path.parts[8][-1]),  # Time
                    epoch_idx  # Epoch index
                ]
                
                # Compute delta & theta power per fronto-central channel
                for ch_name in fc_channels:
                    if ch_name not in epochs_selected.ch_names:
                        # If channel not present, store NaN
                        features += [np.nan, np.nan]
                        continue

                    ch_idx = epochs_selected.ch_names.index(ch_name)
                    ch_data = epoch_data[ch_idx]

                    delta_power = self.bandpower(ch_data, sfreq, [0.5, 4], window_sec=4)
                    theta_power = self.bandpower(ch_data, sfreq, [4, 8], window_sec=4)

                    features += [delta_power, theta_power]

                feature_list.append(features)
        
        # ---- Save to file ----

        columns = ["id", "drug", "time", "epoch_idx"]
        for ch in fc_channels:
            columns.append(f"delta_{ch}")
            columns.append(f"theta_{ch}")

        df = pd.DataFrame(feature_list, columns=columns)
        df.to_csv(feature_output_dir / "eeg_features.csv", index=False)




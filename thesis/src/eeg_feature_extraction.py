import logging
import os
import time

import mne
from tqdm import tqdm
import pandas as pd
# from yasa import bandpower
import numpy as np
# import antropy
from scipy.integrate import simpson
from scipy.signal import welch

from utils.file_mgt import *


"""Loads cleaned and epoched EEG data from .fif file.
Extracts features and saves them in a .csv file.
"""

def bandpower_mne(epochs_data, sf, bands, ch_names=None):
    bandpower_dict = {}

    # Loop over the frequency bands and compute the bandpower for each band
    for low, high, label in bands:
        band_power = []

        # Loop over each channel
        for channel_data in epochs_data:
            # Compute the bandpower for this channel
            band_power.append(bandpower(channel_data, sf, [low, high], window_sec=None))

        # Store the average power for the band
        bandpower_dict[label] = np.mean(band_power)

    return bandpower_dict

def bandpower(data, sf, band, window_sec=None):
    """Compute the average power of the signal in a specific frequency band using Welch's method, 
    ensuring it matches YASA's implementation exactly."""

    band = np.asarray(band, dtype=float)
    low, high = band

    if window_sec is None:
        window_sec = 4  # Ensures a consistent window length

    if not isinstance(window_sec, (int, float)) or window_sec <= 0:
        raise ValueError(f"Invalid window_sec value: {window_sec}")

    nperseg = int(window_sec * sf)  # Match YASA's segment length
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

def compute_brain_wave_band_power(epochs: mne.Epochs) -> tuple[float, float, float]:
    """
    Computes the relative band power averaged across channels and epochs, for Delta, Theta, and Alpha frequency bands.
    """
    delta_power = 0
    theta_power = 0
    alpha_power = 0

    epochs_data = epochs.get_data(copy=False)

    for epoch_id in range(epochs_data.shape[0]):
        
        df = bandpower_mne(epochs_data[epoch_id] * 1e6,
                        sf = float(epochs._raw_sfreq[0]),
                        bands = [(0.5, 4, "Delta"), (4, 8, "Theta"), (8, 13, "Alpha")],
                        ch_names = epochs.ch_names
                        )

        delta_power += np.mean(df['Delta'])
        theta_power += np.mean(df['Theta'])
        alpha_power += np.mean(df['Alpha'])
        del df

    delta_power /= epochs_data.shape[0]
    theta_power /= epochs_data.shape[0]
    alpha_power /= epochs_data.shape[0]

    return (delta_power, theta_power, alpha_power)


def compute_entropies(epochs: mne.Epochs) -> tuple[float, float]:
    """
    Computes the permutation and spectral entropies averaged across channels and across epochs.
    """
    pe = 0
    se = 0
    epochs_data = epochs.get_data(copy=False)
    
    for epoch_id in range(epochs_data.shape[0]):
        for channel_id in range(epochs_data.shape[1]):
            x = epochs_data[epoch_id][channel_id]
            pe += antropy.perm_entropy(x, normalize=True)
            se += antropy.spectral_entropy(x, sf = epochs.info['sfreq'], method = 'welch', normalize = True)

    pe /= epochs_data.shape[1]
    se /= epochs_data.shape[1]

    pe /= epochs_data.shape[0]
    se /= epochs_data.shape[0]

    return (pe, se)


def main():

    # mne.set_config("MNE_BROWSER_BACKEND", "qt")
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

        epochs = mne.read_epochs(path)
        
        # ---- Split by event type ----

        # epochs_audio = epochs['Audio']
        # audio_event_count = epochs_audio.selection.shape[0]
        epochs_arithmetics_moderate = epochs['Mental arithmetics moderate']
        arithmetics_moderate_event_count = epochs_arithmetics_moderate.selection.shape[0]
        epochs_arithmetics_hard = epochs['Mental arithmetics hard']
        arithmetics_hard_event_count = epochs_arithmetics_hard.selection.shape[0]
        del epochs
        # epochs_audio.crop(tmin=0, tmax=10)
        epochs_arithmetics_moderate.crop(tmin=0, tmax=25)
        epochs_arithmetics_hard.crop(tmin=0, tmax=25)

        # ---- Brain wave band power ----

        # powers_audio = compute_brain_wave_band_power(epochs_audio)
        powers_arithmetics_moderate = compute_brain_wave_band_power(epochs_arithmetics_moderate)
        powers_arithmetics_hard = compute_brain_wave_band_power(epochs_arithmetics_hard)

        powers = []

        # Weighted average (by event count)
        for i in range(3):
            temp_power = np.average([powers_arithmetics_moderate[i],
                                     powers_arithmetics_hard[i]],
                                     weights=[arithmetics_moderate_event_count, arithmetics_hard_event_count])
            powers.append(temp_power)

        # Alpha / delta
        powers.append(powers[2] / powers[0])

        features += powers

        # ---- Entropies ----

        # entropies_audio = compute_entropy_features(epochs_audio)
        # entropies_arithmetics_moderate = compute_entropies(epochs_arithmetics_moderate)
        # entropies_arithmetics_hard = compute_entropies(epochs_arithmetics_hard)

        # entropies = []

        # Weighted average (by event count)
        # for i in range(2):
            # temp_entropy = np.average([entropies_arithmetics_moderate[i],
                                     # entropies_arithmetics_hard[i]],
                                     # weights=[arithmetics_moderate_event_count, arithmetics_hard_event_count])
            # entropies.append(temp_entropy)

        # features += entropies

        # ---- Save to data structure ----

        # assert len(features) == 9
        feature_list.append(features)
    
    # ---- Save to file ----

    df = pd.DataFrame(feature_list, columns =['id', 'drug', 'time', 'delta', 'theta', 'alpha', 'ratio'])
    df.to_csv(os.path.join("L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\feature_extraction_files\\eeg_features.csv"), index = False)


if __name__ == "__main__":
    t = time.time()
    main()
    logging.info("Script run in {} s".format(time.time() - t))

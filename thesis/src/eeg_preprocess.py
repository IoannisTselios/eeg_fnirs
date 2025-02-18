import logging
import time
import numpy as np
import mne
import pyxdf
import matplotlib.pyplot as plt
from tqdm import tqdm
from autoreject import get_rejection_threshold
from pyprep import NoisyChannels
from mne.preprocessing import ICA
from utils.file_mgt import get_random_eeg_file_paths
from mne import Annotations


# ---- Function to Get Raw EEG Data from XDF ----
def get_raw_from_xdf(xdf_file_path: str, ref_electrode: str = "") -> mne.io.Raw:
    """This function loads an XDF EEG file, and returns the corresponding mne.io.Raw object.

    Parameters
    ----------
    xdf_file_path : str
        Path to the XDF file.
    ref_electrode : str
        If not empty, a referential montage with that electrode is used, otherwise an average montage is used.
    """
    streams, _ = pyxdf.load_xdf(xdf_file_path)

    # Find where the EEG data is located within the data structure
    assert len(streams) == 2, (
        "Unexpected XDF data structure : expecting 2 streams, got " + str(len(streams))
    )
    if streams[1]["time_series"].shape[0] > streams[0]["time_series"].shape[0]:
        stream_index = 1
        stream_index_markers = 0
    else:
        stream_index = 0
        stream_index_markers = 1

    # Count EEG channels and find the reference channel's index
    channels_info = streams[stream_index]["info"]["desc"][0]["channels"][0]["channel"]
    eeg_channel_count = 0
    ref_channel = -1
    for index, e in enumerate(channels_info):
        if e["type"][0] == "EEG":
            eeg_channel_count += 1
        if e["label"][0] == ref_electrode:
            ref_channel = index

    # Extract channels' info
    data = streams[stream_index]["time_series"].T
    # It is assumed that the EEG channels are the first ones
    data = data[:eeg_channel_count]
    # micro V to V and preamp gain ???
    data[:] *= 1e-6  # / 2
    sfreq = float(streams[stream_index]["info"]["nominal_srate"][0])
    channel_names = [
        e["label"][0]
        + (
            (" - " + ref_electrode)
            if (e["label"][0] != ref_electrode) and ref_electrode != ""
            else ""
        )
        for e in channels_info[:eeg_channel_count]
    ]

    # Data format check
    assert eeg_channel_count > 0, "No EEG channels were found."
    if ref_electrode != "":
        assert ref_channel > -1, "The specified reference electrode was not found."
    for e in channel_names:
        assert e[0] in ["F", "C", "T", "P", "O"], "The channel names are unexpected."
    assert sfreq > 0.0, "The sampling frequency is not a positive number."

    # Create the mne.io.Raw object
    info = mne.create_info(channel_names, sfreq, ["eeg"] * eeg_channel_count)
    raw = mne.io.RawArray(data, info, verbose=False)

    # Event annotations
    origin_time = streams[stream_index]["time_stamps"][0]
    markers_time_stamps = [
        e - origin_time for e in streams[stream_index_markers]["time_stamps"]
    ]
    markers_nb = len(markers_time_stamps)
    markers = Annotations(
        onset=markers_time_stamps,
        duration=[10] * 3 + [25] * 5 + [25] * 5,
        description=["Audio"] * 3
        + ["Mental arithmetics moderate"] * 5
        + ["Mental arithmetics hard"] * 5,
        ch_names=[channel_names] * markers_nb,
    )
    raw.set_annotations(markers)

    # Set the reference montage
    if ref_electrode != "":
        raw = raw.set_eeg_reference(ref_channels=[ref_electrode], verbose=False)
    else:
        raw = raw.set_eeg_reference(verbose=False)  # Use the average montage

    # Set the electrode positions
    channel_mapping = {"Fp1":"AFp1", "Fz": "AFF1h", "F3": "AF7", "F7": "AFF5h", "F9":"FT7", "FC5": "FC5", "FC1":"FC3", "C3":"FCC3h",
        "T7":"FFC1h", "CP5": "FCC1h", "CP1": "CCP3h", "Pz":"CCP1h", "P3":"CP1", "P7": "CP3", "P9":"CPP3h", "O1":"P1",
        "Oz":"AFp2", "O2":"AFF2h", "P10":"AF8", "P8":"AFF6h", "P4":"FT8", "CP2":"FC6", "CP6":"FC4", "T8":"FCC4h",
        "C4":"FFC2h", "Cz":"FCC2h", "FC2":"CCP4h", "FC6": "CCP2h", "F10":"CP2", "F8":"CP4", "F4":"CPP4h", "Fp2":"P2"}
    raw.rename_channels(channel_mapping)
    cap_montage = mne.channels.make_standard_montage("brainproducts-RNP-BA-128")
    raw.set_montage(cap_montage)

    return raw


# ---- Function to Apply Filtering and Handle Spikes ----
def filter_and_handle_spikes(raw: mne.io.Raw, spike_threshold=200e-6, filter_threshold=100e-6, max_consecutive_spikes=5):
    """Filters EEG data and removes/reduces sudden spikes."""
    
    print("🔄 Applying bandpass filter (1-40 Hz)...")
    raw.filter(l_freq=1, h_freq=40, verbose=False)

    print("🔄 Applying notch filter (50 Hz)...")
    raw.notch_filter(50, verbose=False)

    # Get EEG data
    raw_data = raw.get_data()
    peak_values = np.max(np.abs(raw_data), axis=1)

    # Identify extreme and moderate channels
    extreme_channels = np.where(peak_values > spike_threshold)[0]
    moderate_channels = np.where((peak_values > filter_threshold) & (peak_values < spike_threshold))[0]

    # Process extreme spikes: Interpolation or removal
    for ch_idx in extreme_channels:
        channel_data = raw_data[ch_idx]
        spike_indices = np.where(np.abs(channel_data) > spike_threshold)[0]

        if len(spike_indices) > 0:
            print(f"🚨 Handling spikes in channel: {raw.ch_names[ch_idx]} ({len(spike_indices)} occurrences)")
            spike_groups = np.split(spike_indices, np.where(np.diff(spike_indices) > 1)[0] + 1)

            for group in spike_groups:
                if len(group) > max_consecutive_spikes:
                    print(f"❌ Removing segment in channel {raw.ch_names[ch_idx]} (Too many consecutive spikes: {len(group)})")
                    channel_data[group] = np.nan
                else:
                    for spike_idx in group:
                        if spike_idx > 1 and spike_idx < len(channel_data) - 2:
                            channel_data[spike_idx] = np.median(channel_data[max(0, spike_idx-2):min(len(channel_data), spike_idx+3)])
                        else:
                            channel_data[spike_idx] = np.nan

            nan_indices = np.where(np.isnan(channel_data))[0]
            if len(nan_indices) > 0:
                print(f"🔄 Interpolating {len(nan_indices)} missing values in {raw.ch_names[ch_idx]}")
                valid_indices = np.where(~np.isnan(channel_data))[0]
                if len(valid_indices) > 0:
                    channel_data[nan_indices] = np.interp(nan_indices, valid_indices, channel_data[valid_indices])

    # Process moderate spikes: Apply additional filtering
    if len(moderate_channels) > 0:
        affected_moderate_channels = [raw.ch_names[i] for i in moderate_channels]
        print(f"⚠️ Applying additional filtering for moderate spikes in channels: {affected_moderate_channels}")
        raw.filter(l_freq=1, h_freq=30, picks=affected_moderate_channels, verbose=False)

    raw._data[:] = raw_data  # Update raw object
    print("✅ Spikes removed, data cleaned and interpolated.")
    return raw


# ---- Epoch Creation ----
def create_epochs(raw: mne.io.Raw):
    """Creates epochs from raw EEG data while ensuring valid events."""
    events, event_id = mne.events_from_annotations(raw)

    if len(events) == 0:
        print("❌ No valid EEG events found! Skipping epoch creation.")
        return None  

    # Define rejection criteria (increase threshold for fewer rejections)
    reject_criteria = dict(eeg=500e-6)

    epochs = mne.Epochs(
        raw, events, event_id=event_id, preload=True, tmin=-10, tmax=25, baseline=(None, 0), reject=reject_criteria
    )

    return epochs


# ---- Main Function ----
def main():
    logging.basicConfig(force=True, format="%(levelname)s - %(name)s - %(message)s")
    mne.set_log_level("WARNING")

    paths = get_random_eeg_file_paths("xdf", 5000)
    stats = {"bad_channels": [], "bad_epochs": [], "successes": 0}

    for path in tqdm(paths):
        try:
            raw = get_raw_from_xdf(path).load_data()
        except Exception as e:
            logging.error(f"❌ Failed to load {path}: {e}")
            continue

        # ---- Bad channels ---- 
        handler = NoisyChannels(raw)
        handler.find_bad_by_deviation()  # Detects high/low amplitude noise
        handler.find_bad_by_hfnoise()    # Detects high-frequency noise
        bad_channels = handler.get_bads()
        logging.info(f"Bad channels found: {bad_channels}")

        if bad_channels:
            raw.info["bads"] = bad_channels

            # ✅ Check for NaN before interpolation
            if not np.any(np.isnan(raw.get_data())):
                raw.interpolate_bads()
                raw.set_eeg_reference(ref_channels="average")
            else:
                logging.warning("⚠️ Skipping interpolation due to NaN values in raw data.")


        # Plot BEFORE cleaning
        raw.plot(block=True)

        # Apply filtering and handle spikes
        raw_new = filter_and_handle_spikes(raw)
        if raw_new is None:
            continue

        raw._data = raw.get_data()
        
        raw_new.plot(block=True)
        # Create epochs
        epochs = create_epochs(raw_new)
        if epochs is None or len(epochs.selection) == 0:
            logging.warning(f"⚠️ All epochs removed for {path}. Skipping...")
            continue

        # Save Cleaned Data
        fif_file_path = str(path).replace(".xdf", "_clean-epo.fif")
        epochs.save(fif_file_path, overwrite=True)
        stats["successes"] += 1

    logging.info(f"✅ Successfully cleaned {stats['successes']} files.")

if __name__ == "__main__":
    t_start = time.time()
    main()
    logging.info(f"🏁 Script completed in {time.time() - t_start:.2f} seconds.")

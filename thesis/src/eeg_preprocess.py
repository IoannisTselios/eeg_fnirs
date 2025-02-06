import logging
import time

import numpy as np
import mne
from mne import Annotations
import pyxdf
import matplotlib.pyplot as plt
from tqdm import tqdm
from autoreject import get_rejection_threshold
from pyprep import NoisyChannels
import pandas as pd

from utils.file_mgt import *


"""Loads and clean EEG data from .xdf file.
Clean data is saved as .fif file.
"""


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
    dfas = pd.DataFrame(streams[1]["time_series"])
    dfas.to_csv("L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\temp2.csv")
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


def filter_raw(raw: mne.io.Raw) -> None:
    """Filters a mne.io.Raw object : bandpass filter between 1 and 70 Hz, and notch filter at 50 Hz."""
    raw.filter(l_freq=1, h_freq=70, verbose=False)
    raw.notch_filter(50, verbose=False) # TODO : Remove this line if it turns out we can low pass at less than 50Hz
    return


def epochs_from_raw(raw: mne.io.Raw) -> mne.Epochs:
    """
    Returns a mne.Epochs object created from an annotated mne.io.Raw object.
    The length of the epochs is arbitrarily set, based on how the data was acquired.
    """
    events, events_id = mne.events_from_annotations(raw)
    return mne.Epochs(
        raw, events, event_id=events_id, preload=True, tmin=-10, tmax=25, baseline=(None, 0)
    )


def add_brain_wave_types_lines_on_pyplot_figure():
    """To be called after : fig = spectrum.plot()"""
    plt.axvline(x=0, color="b")
    # Delta
    plt.axvline(x=4, color="b")
    # Theta
    plt.axvline(x=8, color="b")
    # Alpha
    plt.axvline(x=13, color="b")
    # Beta
    plt.axvline(x=30, color="b")
    # Gamma


def main():

    logging.basicConfig(force=True, format='%(levelname)s - %(name)s - %(message)s')
    # mne.set_config("MNE_BROWSER_BACKEND", "qt")
    mne.set_log_level("WARNING")

    paths = list()
    paths = get_random_eeg_file_paths("xdf", 500)
    # paths = get_random_eeg_file_paths_one_session("xdf")
    # paths.append(r"data\raw\DRUG1\ID33\Baseline\sub-P001_ses-S001_task-Default_run-001_eeg_old387.xdf")

    stats = {"bad_channels":[], "bad_epochs":[], "successes":0}

    for path in tqdm(paths):

        # ---- Raw ----

        try:
            raw = get_raw_from_xdf(path).load_data()
        except Exception as e:
            logging.error(e)
            continue

        # ---- Bad channels ---- 

        handler = NoisyChannels(raw)
        handler.find_bad_by_deviation() # high/low overall amplitudes
        handler.find_bad_by_hfnoise() # high-frequency noise
        bad_channels = handler.get_bads()
        logging.info("Bad channels found by pyprep ({}) : {}".format(len(bad_channels), bad_channels))
        stats["bad_channels"].append(len(bad_channels))
        raw.info["bads"] = bad_channels
        if len(bad_channels) > 0 :
            raw.interpolate_bads()
            raw = raw.set_eeg_reference(ref_channels="average")

        # ---- Filtering ----

        filter_raw(raw)
        # raw.plot(block=True)

        # ---- Epoch ----

        epochs = epochs_from_raw(raw).load_data()
        del raw

        # ---- Bad epochs ----

        # ar = autoreject.AutoReject(n_interpolate=[1, 2, 3, 4], n_jobs=1, verbose=False)
        # ar.fit(epochs)
        # epochs = ar.transform(epochs)
        event_count = len(epochs.selection)
        reject = get_rejection_threshold(epochs, verbose=False)
        epochs.drop_bad(reject=reject)
        logging.info("{} epoch(s) were dropped by Autoreject".format(event_count - len(epochs.selection)))
        stats["bad_epochs"].append(event_count - len(epochs.selection))
        if (
            # set(epochs.selection).intersection([0, 1, 2]) == set() or # we won't use the passive paradigm
            set(epochs.selection).intersection([3, 4, 5, 6, 7]) == set() or
            set(epochs.selection).intersection([8, 9, 10, 11, 12]) == set()
        ):
            logging.warning("All epochs for one or more event type were dropped. Skipping to next recording.")
            continue

        # ---- Save as file ----

        file_name = "\\".join(str(path).split("\\")[:-1]) + "\\clean-epo.fif" # Same path, different file name
        epochs.save(file_name, overwrite=True)
        stats["successes"] += 1

    logging.info("Number of successfully cleaned recordings: {} ({}%)".format(stats["successes"], (stats["successes"]/len(paths)*100)))

    number = len([e for e in stats["bad_channels"] if e > 0])
    average = np.mean(np.array(stats["bad_channels"]))
    logging.info("Number of recordings with bad channels: {}".format(number))
    logging.info("Average number of bad channels (for all recordings): {}".format(average))

    number = len([e for e in stats["bad_epochs"] if e > 0])
    average = np.mean(np.array(stats["bad_epochs"]))
    logging.info("Number of recordings with bad epochs: {}".format(number))
    logging.info("Average number of bad epochs (for all recordings): {}".format(average))


if __name__ == "__main__":
    t = time.time()
    main()
    logging.info("Script run in {} s".format(time.time() - t))
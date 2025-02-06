import logging
import os
from itertools import compress

import mne
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from utils.file_mgt import get_random_eeg_file_paths


def compute_average_slope(epochs: mne.Epochs, show_plots : bool = False) -> float:
    """
    For each epoch and channel, fit a linear model to the signal.
    Returns the average slope.
    """
    X = epochs.times.reshape(-1, 1)
    epochs = epochs.drop_channels(ch_names = [e for e in epochs.ch_names if 'hbr' in e], on_missing='raise') # include only oxy-Hb (Hbo) channels
    slopes = []
    for event_id in range(epochs._data.shape[0]):
        for channel_id in range(epochs._data[event_id].shape[0]):
            y = epochs._data[event_id][channel_id]
            reg = LinearRegression().fit(X, y)
            slopes.append(reg.coef_[0])

            if show_plots:
                print("event number {}".format(event_id))
                print("channel number {}".format(channel_id))
                predictions = reg.predict(X)
                plt.plot(X, y)
                plt.plot(X, predictions, color='g')
                plt.xlabel("Time (s)")
                plt.ylabel("hbo (M?)")
                plt.show()

    return np.mean(np.array(slopes))


paths = get_random_eeg_file_paths("snirf", 500)
feature_list = []
stats = {"bad_channels":[], "bad_epochs":[], "successes":0}

for path in tqdm(paths):

    logging.info("Current file is {}".format(path))
    features = []

    # ---- Recording info ----

    features.append(path.parts[3].replace("ID", "")) # Patient ID
    features.append(int(path.parts[2][-1])) # Drug
    features.append(0 if path.parts[4][-1] == "e" else int(path.parts[4][-1])) # Time of recording

    # ---- Load ----

    raw_intensity = mne.io.read_raw_snirf(path)

    picks = mne.pick_types(raw_intensity.info, meg=False, fnirs=True)
    dists = mne.preprocessing.nirs.source_detector_distances(
        raw_intensity.info, picks=picks
    )
    raw_intensity.pick(picks[dists > 0.01])
    # raw_intensity.plot()

    # ---- Convert to optical density ----

    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
    del raw_intensity
    # raw_od.plot()

    # ---- Detect and mark bad channels based on SCI ----

    sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
    # fig, ax = plt.subplots(layout="constrained")
    # ax.hist(sci)
    # ax.set(xlabel="Scalp Coupling Index", ylabel="Count", xlim=[0, 1])
    # plt.show()
    raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < 0.5))
    stats["bad_channels"].append(len(raw_od.info["bads"]))
    # raw_od.interpolate_bads() # Need montage for this

    # ---- Convert to Hbo concentration ----

    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)
    del raw_od
    # raw_haemo.plot()

    # ---- Filter ----

    raw_haemo.filter(0.02, 0.7)
    # raw_haemo.plot(n_channels=len(raw_haemo.ch_names), duration=700, show_scrollbars=False)
    # raw_haemo.get_montage().plot()

    # ---- Epochs ----

    events, _ = mne.events_from_annotations(raw_haemo)
    # Have to manually adjust the events labels
    if events.shape[0] != 13:
        print("This recording is missing at least one event. Skipping to the next one.")
        continue
    events[0:3,2] = 0
    events[3:8,2] = 1
    events[8:13,2] = 2
    event_dict = {"Audio":0,
                  "Mental arithmetics moderate":1,
                  "Mental arithmetics hard":2
                  }
    # reject_criteria = dict(hbo=80e-6)

    epochs = mne.Epochs(
        raw_haemo,
        events,
        event_id=event_dict,
        tmin=0,
        tmax=25,
        # reject=reject_criteria,
        reject_by_annotation=True,
        proj=True,
        baseline=None, # baselining does not matter if we just use the slope as a feature
        preload=True,
        detrend=None
    )
    # epochs.plot(block=True)
    # epochs.plot_drop_log()
    del raw_haemo
        
    # ---- Split by event type ----

    # epochs_audio = epochs['Audio']
    # audio_event_count = epochs_audio.selection.shape[0]
    epochs_arithmetics_moderate = epochs['Mental arithmetics moderate']
    arithmetics_moderate_event_count = epochs_arithmetics_moderate.selection.shape[0]
    epochs_arithmetics_hard = epochs['Mental arithmetics hard']
    arithmetics_hard_event_count = epochs_arithmetics_hard.selection.shape[0]
    channel_nb = len([ch_name for ch_name in epochs.ch_names if 'hbo' in ch_name])
    del epochs
    # epochs_audio.crop(tmin=0, tmax=10)
    epochs_arithmetics_moderate.crop(tmin=0, tmax=25) # Discard the baseline period
    epochs_arithmetics_hard.crop(tmin=0, tmax=25)

    # ---- Compute features ----

    # Weighted average slope
    s_moderate = compute_average_slope(epochs_arithmetics_moderate)
    s_hard = compute_average_slope(epochs_arithmetics_hard)
    s = np.average([s_moderate, s_hard],
                   weights=[arithmetics_moderate_event_count, arithmetics_hard_event_count])
    features.append(s)

    # ---- Save to data structure ----

    assert len(features) == 4
    feature_list.append(features)
    stats["successes"] += 1

# ---- Combine and save to file ----

df = pd.DataFrame(feature_list, columns =['id', 'drug', 'time', 'fnirs_1'])
df.to_csv(os.path.join("data", "processed", "fnirs_features.csv"), index = False)

# ---- Display some stats ----

logging.info("Number of successfully cleaned recordings: {} ({}%)".format(stats["successes"], (stats["successes"]/len(paths)*100)))

number = len([e for e in stats["bad_channels"] if e > 0])
average = np.mean(np.array(stats["bad_channels"]))
logging.info("Number of recordings with bad channels: {}".format(number))
logging.info("Average number of bad channels (for all recordings): {}".format(average))

number = len([e for e in stats["bad_epochs"] if e > 0])
average = np.mean(np.array(stats["bad_epochs"]))
logging.info("Number of recordings with bad epochs: {}".format(number))
logging.info("Average number of bad epochs (for all recordings): {}".format(average))
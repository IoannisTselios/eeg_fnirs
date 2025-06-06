import logging
from pathlib import Path
import numpy as np
import mne
import pandas as pd
import pyxdf
from tqdm import tqdm
from pyprep import NoisyChannels
from utils.file_mgt import get_random_eeg_file_paths
from mne import Annotations
from mne.preprocessing import ICA
from mne_icalabel import label_components
from autoreject import AutoReject

# ---- EEG Preprocessing ----
class EEGPreprocessor:

    def __init__(self, excluded_dir, epoch_size, sample_size, epoch_rejection_threshold, ica_threshold, preprocessed_files_dir):
        self.excluded_dir = excluded_dir
        self.epoch_size = epoch_size
        self.sample_size = sample_size
        self.epoch_rejection_threshold = epoch_rejection_threshold
        self.ica_threshold = ica_threshold
        self.preprocessed_files_dir = preprocessed_files_dir
        logging.info("EEG Preprocessor Initialized")

    def get_raw_from_xdf(self, xdf_file_path: str, ref_electrode: str = "") -> mne.io.Raw:
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

        data = streams[stream_index]["time_series"].T

        data = data[:eeg_channel_count]

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

        # Event annotations (original markers)
        origin_time = streams[stream_index]["time_stamps"][0]
        markers_time_stamps = [
            e - origin_time for e in streams[stream_index_markers]["time_stamps"]
        ]
        markers_nb = len(markers_time_stamps)

        # ✅ Original event descriptions
        event_descriptions = ["Audio"] * 3 + ["Mental arithmetics moderate"] * 5 + ["Mental arithmetics hard"] * 5
        event_durations = [10] * 3 + [25] * 5 + [25] * 5  # Original durations

        # ✅ Define segment duration for splitting
        segment_duration = self.epoch_size  # Change this value if needed

        # ✅ Create new split events while keeping original descriptions
        new_onsets = []
        new_descriptions = []

        for onset, duration, desc in zip(markers_time_stamps, event_durations, event_descriptions):
            segment_times = np.arange(onset, onset + duration, segment_duration)  # Splitting into 5s segments
            new_onsets.extend(segment_times)
            new_descriptions.extend([desc] * len(segment_times))  # Assign the same description to each segment

        # ✅ Store all new split annotations
        split_markers = Annotations(onset=new_onsets, duration=[segment_duration] * len(new_onsets), description=new_descriptions)
        raw.set_annotations(split_markers)

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


    def apply_prep_pipeline(self, raw: mne.io.Raw, name_of_preprocessed):
        stats = {"bad_channels": 0, "rejected_epochs": 0, "avg_removed_amp": [], "kept_epoch_amplitudes": []}

        print("\n Starting Preprocessing...")

        # ✅ Step 1: Remove Line Noise using Multi-Taper Spectrum Fit
        print("\n🔹 Step 1: Removing Line Noise using Multi-Taper Spectrum Fit...")
        raw.notch_filter(freqs=[50], fir_design='firwin', method='spectrum_fit')

        # ✅ Step 2: High-pass filter at 0.5 Hz
        print("🔹 Step 2: High-Pass Filtering (0.5 Hz)...")
        raw.filter(l_freq=0.5, h_freq=40, fir_design='firwin')

        # ✅ Step 3: Detect bad channels using PyPREP
        print("🔹 Step 3: Detecting Bad Channels using PyPREP...")
        prep_handler = NoisyChannels(raw)

        try:
            prep_handler.find_bad_by_nan_flat()
            prep_handler.find_bad_by_deviation()
            prep_handler.find_bad_by_correlation(
                correlation_secs=1.5, correlation_threshold=0.45, frac_bad=0.03
            )
            np.random.seed(42)
            prep_handler.find_bad_by_ransac(
                n_samples=self.sample_size, sample_prop=0.25, corr_thresh=0.75,
                frac_bad=0.4, corr_window_secs=5, channel_wise=False
            )
        except Exception as e:
            print(f" Error in bad channel detection: {e}")

        bad_channels = prep_handler.get_bads()
        stats["bad_channels"] = len(bad_channels)

        print(f"\n Bad channels detected: {bad_channels}")
        if len(bad_channels) <= int(len(raw.ch_names) * 0.3):
            raw.info['bads'] = bad_channels
            raw.interpolate_bads(reset_bads=True)
            print(f" Interpolated {len(bad_channels)} channels")
        else:
            print(f" Too many bad channels ({len(bad_channels)}). Skipping interpolation.")

        # ✅ Step 4: Re-reference after detecting bad channels
        good_channels = [ch for ch in raw.ch_names if ch not in bad_channels]
        if len(good_channels) > 0:
            raw.set_eeg_reference(ref_channels=good_channels)
            print(f" Re-referenced using {len(good_channels)} good channels.")

        # # ✅ Step 5: ICA for artifact detection
        # print("🔹 Step 5: Running ICA to Detect Artifacts...")
        # try:
        #     ica = ICA(n_components=0.99, random_state=42, max_iter="auto")
        #     ica.fit(raw)

        #     try:
        #         # ✅ Detect muscle artifacts using z-score threshold
        #         muscle_indices, muscle_scores = ica.find_bads_muscle(raw, threshold=self.ica_threshold)
        #         if muscle_indices:
        #             ica.exclude.extend(muscle_indices)
        #             print(f"✅ Detected and removed {len(muscle_indices)} muscle artifacts")

        #     except Exception as e:
        #         print(f"❌ Error in muscle artifact detection: {e}")

        #     try:
        #         # ✅ Label ICA components using ICLabel
        #         labels = label_components(raw, ica, method="iclabel")
        #         bad_components = [
        #             idx for idx, label in enumerate(labels["labels"])
        #             if label in ["eye blink", "muscle artifact", "line noise"]
        #         ]
        #         if bad_components:
        #             ica.exclude.extend(bad_components)
        #             print(f"✅ Detected and removed {len(bad_components)} bad ICs via ICLabel")

        #     except Exception as e:
        #         print(f"❌ Error in ICLabel component labeling: {e}")

        #     try:
        #         # ✅ Apply ICA (remove bad components)
        #         ica.apply(raw)
        #         print(f"✅ Applied ICA — Total components removed: {len(ica.exclude)}")

        #     except Exception as e:
        #         print(f"❌ Error in ICA application: {e}")

        # except Exception as e:
        #     print(f"❌ Error in ICA fitting: {e}")

        # ✅ Step 6: Remove existing artifact annotations before epoching
        if len(raw.annotations) > 0:
            print("🔹 Step 6: Removing existing artifact annotations...")
            raw.annotations.delete(
                [i for i, desc in enumerate(raw.annotations.description) if "BAD_" in desc]
            )
            print(f" Removed {len(raw.annotations)} artifact annotations")

        print(" Preprocessing Complete!")
        # ✅ Step 7: Save preprocessed raw to FIF
        output_path = self.preprocessed_files_dir / f"{name_of_preprocessed}_preprocessed_raw.fif"
        raw.save(output_path, overwrite=True)
        print(f"💾 Saved preprocessed raw file to: {output_path}")


        return raw, stats


    def create_epochs(self, raw: mne.io.Raw, stats):
        """
        Slices raw EEG into 1s epochs using annotation priority,
        assigns 'Unlabeled' if no match, then cleans with AutoReject.
        """
        from mne import Annotations, events_from_annotations, Epochs
        from autoreject import AutoReject
        import numpy as np

        # === Annotation priority map ===
        annotation_priority = {
            "Mental arithmetics moderate": 0,
            "Mental arithmetics hard": 1,
            "Audio": 2
        }

        sfreq = raw.info["sfreq"]
        total_duration = raw.times[-1]
        slice_len = 1.0
        onsets, descriptions = [], []

        for start in np.arange(0, total_duration - slice_len, slice_len):
            end = start + slice_len
            best_label = "Unlabeled"
            best_score = float("inf")

            for annot in raw.annotations:
                annot_start = annot["onset"]
                annot_end = annot["onset"] + annot["duration"]
                if annot_start < end and annot_end > start:
                    label = annot["description"]
                    if label in annotation_priority:
                        score = annotation_priority[label]
                        if score < best_score:
                            best_label = label
                            best_score = score

            onsets.append(start)
            descriptions.append(best_label)

        # ✅ Replace annotations with new sliced labels
        raw.set_annotations(Annotations(
            onset=onsets,
            duration=[slice_len] * len(onsets),
            description=descriptions
        ))

        events, event_id = events_from_annotations(raw)

        if len(events) == 0:
            print("❌ No valid EEG events found! Skipping epoch creation.")
            return None, stats, 0

        print("\n\033[92m🔄 Creating 1s epochs from raw...\033[0m")
        epochs_all = Epochs(
            raw,
            events=events,
            event_id=event_id,
            preload=True,
            tmin=0.0,
            tmax=1.0,
            baseline=None,
            reject_by_annotation=False
        )

        all_epoch_amplitudes = np.max(np.abs(epochs_all.get_data()), axis=(1, 2))
        stats["max_rejected_amp"] = float(np.max(all_epoch_amplitudes))
        stats["avg_removed_amp"] = float(np.mean(all_epoch_amplitudes))
        print(f"   ✅ Max amplitude before rejection: {stats['max_rejected_amp']:.2f} µV")

        print("\n\033[92m🔄 Cleaning epochs with AutoReject...\033[0m")
        ar = AutoReject(thresh_method='bayesian_optimization', random_state=42, n_jobs=-1)
        epochs_clean = ar.fit_transform(epochs_all)
        reject_log = ar.get_reject_log(epochs_all)

        stats["rejected_epochs"] = int(np.sum(reject_log.bad_epochs))
        stats["interpolated_epochs"] = int(np.sum(np.any(reject_log.labels == 1, axis=1)))
        stats["interpolated_channels"] = int(np.sum(reject_log.labels))

        print(f"    Total epochs rejected by AutoReject: {stats['rejected_epochs']}")
        print(f"    Epochs with interpolated channels: {stats['interpolated_epochs']}")
        print(f"    Total channels interpolated across all epochs: {stats['interpolated_channels']}")

        if len(epochs_clean) == 0:
            print("⚠️ All epochs removed after AutoReject!")
            logging.warning("⚠️ All epochs removed after AutoReject.")
            return None, stats, len(events)

        return epochs_clean, stats, len(events)


    def process(self):
        logging.basicConfig(force=True, format="%(levelname)s - %(name)s - %(message)s")
        mne.set_log_level("WARNING")

        paths = get_random_eeg_file_paths("xdf", 5000)
        overall_stats = {"total_files": 0, "excluded_files": 0, "avg_bad_channels": [], "avg_rejected_epochs": [], "avg_removed_amp": [], "kept_data_ratio": []}
        skipped_files = []

        with open("files_with_2.txt", "r") as f:
            skip_ids = set(line.strip() for line in f if line.strip())

        for path in tqdm(paths):
            try:
                parts = Path(path).parts
                identifier = f"{parts[-4]}_{parts[-3]}_{parts[-2]}"
            except IndexError:
                logging.warning(f" Could not parse identifier from path: {path}")
                continue

            if identifier in skip_ids:
                logging.info(f"⏭ Skipping {identifier} — in skip list")
                # skipped_files.append(str(path))
                continue

            try:
                # Load raw EEG data
                raw = self.get_raw_from_xdf(path).load_data()
            except Exception as e:
                logging.error(f" Failed to load {path}: {e}")
                overall_stats["excluded_files"] += 1
                continue

            overall_stats["total_files"] += 1

            parts = path.parts[-4:-1]  # ['Patient ID 1 - U1 (UWS)', 'Session 1', 'Baseline']
            custom_name = "_".join(parts)

            # Apply PREP pipeline
            # raw.plot(block=True)
            raw, stats = self.apply_prep_pipeline(raw, custom_name)
            # raw.plot(block=True)

            # Create epochs
            epochs, stats, num_events = self.create_epochs(raw, stats)
            # 🚨 CASE 1: If epochs is None, log and skip file immediately
            if epochs is None:
                reason = "All epochs removed due to artifact rejection."
                logging.error(f" {reason} for {path}")
                skipped_files.append((path, reason))
                overall_stats["excluded_files"] += 1
                continue

            # 🚨 CASE 2: If epochs exist but all were rejected, log drop reasons
            if len(epochs.selection) == 0:
                drop_log_str = "; ".join([f"Epoch {i}: {reason}" for i, reason in enumerate(epochs.drop_log) if reason])
                reason = f"All epochs removed! Reasons: {drop_log_str if drop_log_str else 'Unknown'}"
                logging.error(f" {reason} for {path}")
                skipped_files.append((path, reason)) 

                logging.error(" Too many bad channels?")
                logging.error(f" Channels interpolated: {stats['bad_channels']}")
                logging.error(" Artifact rejection threshold too strict?")
                logging.error(f" Rejected Epochs: {stats['rejected_epochs']}")
                logging.error(" Consider adjusting rejection criteria or checking EEG quality.")

                overall_stats["excluded_files"] += 1
                continue

            rejected_epoch_logs = []
            for i, reason in enumerate(epochs.drop_log):
                if reason:  # Only log rejected epochs
                    rejected_epoch_logs.append(f"Epoch {i}: {reason}")

            # 🚨 If some epochs were rejected, save them to `rejected_epochs_log.txt`
            if rejected_epoch_logs:
                with open(self.excluded_dir / "rejected_epochs_log.txt", "a") as f:
                    f.write(f"\nFile: {path}\n")
                    for log in rejected_epoch_logs:
                        f.write(log + "\n")

                logging.warning(f" Some epochs were rejected in {path}. Reasons stored in rejected_epochs_log.txt")

            # Save Cleaned Data
            fif_file_path = str(path).replace(".xdf", "_clean-epo.fif")
            epochs.save(fif_file_path, overwrite=True)

            # Update statistics
            overall_stats["avg_bad_channels"].append(stats["bad_channels"])
            overall_stats["avg_rejected_epochs"].append(stats["rejected_epochs"])
            overall_stats["avg_removed_amp"].append(stats["avg_removed_amp"])
            overall_stats["kept_data_ratio"].append(len(epochs) / num_events if num_events > 0 else 0)

            logging.info(f" Successfully processed {path}.")
            logging.info(f" Bad Channels Interpolated: {stats['bad_channels']}")
            logging.info(f" Rejected Epochs: {stats['rejected_epochs']}")
            logging.info(f" Avg Removed Amplitude: {stats['avg_removed_amp'] * 1e6:.2f} μV" if stats["avg_removed_amp"] is not None else "🔹 Avg Removed Amplitude: N/A")
            logging.info(f" Kept Data Ratio: {100 * (len(epochs) / num_events):.2f}%")

            with open(self.excluded_dir / "skipped_files_log.txt", "w") as f:
                for file, reason in skipped_files:
                    f.write(f"{file}, {reason}\n")

            logging.info(f" Skipped files log saved: skipped_files_log.txt")

        # Final summary
        # Define the summary file path
        summary_file_path = self.excluded_dir / "preprocessing_summary.txt"

        # Write the summary to the file
        with open(summary_file_path, "w", encoding="utf-8") as f:
            f.write("**Final Preprocessing Summary:**\n")
            f.write(f"Total Files Processed: {overall_stats['total_files']}\n")
            f.write(f"Files Excluded: {overall_stats['excluded_files']}\n")
            f.write(f"Avg Bad Channels Interpolated: {np.mean(overall_stats['avg_bad_channels']):.2f}\n")
            f.write(f"Avg Epochs Rejected per File: {np.mean(overall_stats['avg_rejected_epochs']):.2f}\n")
            f.write(f"Avg Removed Amplitude: {np.mean(overall_stats['avg_removed_amp']) * 1e6:.2f} μV\n")
            f.write(f"Avg Kept Data Ratio: {np.mean(overall_stats['kept_data_ratio']) * 100:.2f}%\n")

        # Log that the summary has been saved
        logging.info(f"📄 Preprocessing summary saved to: {summary_file_path}")

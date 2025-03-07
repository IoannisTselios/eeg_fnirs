import logging
import time
import warnings
import numpy as np
import mne
import pyxdf
from tqdm import tqdm
from pyprep import NoisyChannels
from utils.file_mgt import get_random_eeg_file_paths
from mne import Annotations
from mne.preprocessing import ICA, compute_current_source_density, annotate_muscle_zscore
from scipy.signal import spectrogram
from mne.preprocessing import compute_current_source_density
from sklearn.cross_decomposition import CCA

# ---- EEG Preprocessing ----
class EEGPreprocessor:

    def __init__(self, excluded_dir):
        self.excluded_dir = excluded_dir
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


    def apply_prep_pipeline(self, raw: mne.io.Raw):
        """Applies PREP-like preprocessing to EEG data and tracks stats."""
        
        stats = {"bad_channels": 0, "rejected_epochs": 0, "avg_removed_amp": [], "kept_epoch_amplitudes": []}

        print("\U0001F504 Step 1: Removing Line Noise (50Hz Notch Filter)...")
        raw.notch_filter(freqs=[50], fir_design='firwin')

        print("\U0001F504 Step 2: High-Pass Filtering (0.5 Hz)...")  # More conservative filtering
        raw.filter(l_freq=0.5, h_freq=70, fir_design='firwin')

        print("\U0001F504 Step 3: Detecting Bad Channels using PyPREP...")
        prep_handler = NoisyChannels(raw)

        # Detect bad channels using multiple methods
        try:
            print("🔹 Step 1: Detecting NaN and Flat Channels...")
            prep_handler.find_bad_by_nan_flat()
        except Exception as e:
            print(f"⚠️ Error in `find_bad_by_nan_flat()`: {e}")

        try:
            print("🔹 Step 2: Detecting Extremely Noisy Channels (Deviation)...")
            prep_handler.find_bad_by_deviation()
        except Exception as e:
            print(f"⚠️ Error in `find_bad_by_deviation()`: {e}")

        try:
            print("🔹 Step 3: Detecting Weakly Correlated Channels...")
            prep_handler.find_bad_by_correlation(correlation_secs=1.5, correlation_threshold=0.45, frac_bad=0.03)
        except Exception as e:
            print(f"⚠️ Error in `find_bad_by_correlation()`: {e}")

        try:
            print("🔹 Step 4: Running RANSAC to Detect Bad Channels...")
            np.random.seed(42)  # Ensures RANSAC selects the same data subsets
            prep_handler.find_bad_by_ransac(n_samples=60, sample_prop=0.4, corr_thresh=0.80, frac_bad=0.35, corr_window_secs=5, channel_wise=False)
        except Exception as e:
            print(f"⚠️ Error in `find_bad_by_ransac()`: {e}")

        bad_channels = prep_handler.get_bads()
        stats["bad_channels"] = len(bad_channels)

        print(f"⚠️ Bad channels detected: {bad_channels}")
        if len(bad_channels) > 0:
            bad_channels = bad_channels[:max(1, int(len(raw.ch_names) * 0.05))]
            raw.info["bads"] = bad_channels
            print("\U0001F504 Interpolating Bad Channels...")
            raw.interpolate_bads()

        # print("\U0001F504 Step 4: Computing Current Source Density (CSD)...")
        # raw_csd = compute_current_source_density(raw.copy())
        # # print(raw_csd)

        # print("\U0001F504 Step 5: Running ICA to Remove Artifacts...")
        # try:
        #     # Try using an automatic number of components
        #     ica = ICA(n_components=0.99, random_state=97, method="fastica")
        #     ica.fit(raw)

        # except RuntimeError as e:
        #     print(f"⚠️ ICA failed: {e}")

        #     # If ICA fails due to a low number of components, try with `n_components=None`
        #     print("🔄 Retrying ICA with `n_components=None` (no PCA reduction)...")

        #     try:
        #         n_components = min(len(raw.ch_names) - 1, 25)  # Default: Keep at least 25 components
        #         ica = ICA(n_components=n_components, random_state=97, method="fastica")
        #         ica.fit(raw)

        #     except RuntimeError as e:
        #         print(f"❌ ICA failed again: {e}")
        #         print("⚠️ Skipping ICA and continuing the pipeline...")
        #         return raw, stats  # Skip ICA and return the partially processed data

        # # Detect and exclude ICA components for muscle artifacts
        # muscle_inds, _ = ica.find_bads_muscle(raw_csd, threshold=3.0)
        # ica.exclude.extend(muscle_inds)

        # print(f"✅ Removing {len(ica.exclude)} muscle-related ICA components...")
        # raw_clean = ica.apply(raw)
        # print("✅ Muscle artifacts removed successfully with ICA.")

        # print("\U0001F504 Step 6: Detecting and Annotating Muscle Artifacts...")

        # annot_muscle, scores_muscle = annotate_muscle_zscore(
        #     raw,
        #     ch_type=None,
        #     threshold=3.0,  
        #     min_length_good=0.2,
        #     filter_freq=[50, 70],
        # )

        # raw.set_annotations(raw.annotations + annot_muscle)  # Merge new and existing annotations
        # raw.plot(block=True)
        print("✅ Muscle artifacts detected and annotated.")

        return raw, stats


    # ---- Function to Create Epochs and Track Rejection Stats ----
    def create_epochs(self, raw: mne.io.Raw, stats):
        """Creates epochs from raw EEG data while ensuring valid events and tracking rejections."""

        events, event_id = mne.events_from_annotations(raw)

        if len(events) == 0:
            print("❌ No valid EEG events found! Skipping epoch creation.")
            return None, stats  

        # Define rejection criteria
        reject_criteria = dict(eeg=500e-6)  # Reject epochs with amplitude > 500 µV

        print("🔄 Step 1: Creating epochs WITHOUT rejection first (to analyze rejected epochs)...")
        epochs_all = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            preload=True,
            tmin=-10,
            tmax=25,
            baseline=(None, 0),
            reject_by_annotation=False  # Keep all epochs for analysis
        )

        # ✅ Compute max amplitude of each epoch BEFORE rejection
        all_epoch_amplitudes = np.max(np.abs(epochs_all.get_data()), axis=(1, 2))  # Max amplitude per epoch

        # ✅ Identify rejected epochs based on amplitude threshold (500 µV)
        rejected_epochs = all_epoch_amplitudes > 150e-6
        rejected_amplitudes = all_epoch_amplitudes[rejected_epochs]

        # ✅ Store rejection stats
        stats["max_rejected_amp"] = np.max(rejected_amplitudes) if len(rejected_amplitudes) > 0 else 0
        stats["avg_removed_amp"] = np.mean(rejected_amplitudes) if len(rejected_amplitudes) > 0 else 0
        stats["kept_epoch_amplitudes"] = all_epoch_amplitudes[~rejected_epochs].tolist()

        print("🔄 Step 2: Creating epochs with artifact rejection...")

        # ✅ Step 2: Create epochs with rejection, including "MUSCLE" exclusion
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            preload=True,
            tmin=-10,
            tmax=25,
            baseline=(None, 0),
            reject=reject_criteria,
            reject_by_annotation=True  # ✅ This ensures "BAD_muscle" segments are excluded
        )

        # ✅ Track rejected epochs count
        stats["rejected_epochs"] = len(events) - len(epochs.selection)

        # ✅ Count how many epochs were removed due to "BAD_muscle"
        muscle_rejected = sum(["BAD_muscle" in log for log in epochs.drop_log])
        stats["muscle_rejected_epochs"] = muscle_rejected

        # ✅ Store rejected epoch reasons inside `stats`
        stats["rejected_epoch_reasons"] = {f"Epoch {i}": reason for i, reason in enumerate(epochs.drop_log) if reason}

        # ✅ If all epochs are rejected, log and return None
        if len(epochs) == 0:
            print("⚠️ All epochs were removed due to artifact rejection!")
            logging.warning(f"⚠️ All epochs removed. Check EEG signal quality and rejection criteria.")
            return None, stats, len(events)  # Avoid further errors

        print(f"✅ Epoch creation completed. {muscle_rejected} epochs removed due to 'MUSCLE' artifacts.")
        
        return epochs, stats, len(events)


    def process(self):
        logging.basicConfig(force=True, format="%(levelname)s - %(name)s - %(message)s")
        mne.set_log_level("WARNING")

        paths = get_random_eeg_file_paths("xdf", 5000)
        overall_stats = {"total_files": 0, "excluded_files": 0, "avg_bad_channels": [], "avg_rejected_epochs": [], "avg_removed_amp": [], "kept_data_ratio": []}
        skipped_files = []

        for path in tqdm(paths):
            try:
                # Load raw EEG data
                raw = self.get_raw_from_xdf(path).load_data()
            except Exception as e:
                logging.error(f"❌ Failed to load {path}: {e}")
                overall_stats["excluded_files"] += 1
                continue

            overall_stats["total_files"] += 1

            # Apply PREP pipeline
            raw, stats = self.apply_prep_pipeline(raw)
            # raw.plot(block=True)

            # Create epochs
            epochs, stats, num_events = self.create_epochs(raw, stats)
            # 🚨 CASE 1: If epochs is None, log and skip file immediately
            if epochs is None:
                reason = "All epochs removed due to artifact rejection."
                logging.error(f"🛑 {reason} for {path}")
                skipped_files.append((path, reason))
                overall_stats["excluded_files"] += 1
                continue

            # 🚨 CASE 2: If epochs exist but all were rejected, log drop reasons
            if len(epochs.selection) == 0:
                drop_log_str = "; ".join([f"Epoch {i}: {reason}" for i, reason in enumerate(epochs.drop_log) if reason])
                reason = f"All epochs removed! Reasons: {drop_log_str if drop_log_str else 'Unknown'}"
                logging.error(f"🛑 {reason} for {path}")
                skipped_files.append((path, reason))

                logging.error("🔹 Too many bad channels?")
                logging.error(f"🔹 Channels interpolated: {stats['bad_channels']}")
                logging.error("🔹 Artifact rejection threshold too strict?")
                logging.error(f"🔹 Rejected Epochs: {stats['rejected_epochs']}")
                logging.error("🔹 Consider adjusting rejection criteria or checking EEG quality.")

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

                logging.warning(f"⚠️ Some epochs were rejected in {path}. Reasons stored in rejected_epochs_log.txt")

            # Save Cleaned Data
            fif_file_path = str(path).replace(".xdf", "_clean-epo.fif")
            epochs.save(fif_file_path, overwrite=True)

            # Update statistics
            overall_stats["avg_bad_channels"].append(stats["bad_channels"])
            overall_stats["avg_rejected_epochs"].append(stats["rejected_epochs"])
            overall_stats["avg_removed_amp"].append(stats["avg_removed_amp"])
            overall_stats["kept_data_ratio"].append(len(epochs) / num_events if num_events > 0 else 0)

            logging.info(f"✅ Successfully processed {path}.")
            logging.info(f"🔹 Bad Channels Interpolated: {stats['bad_channels']}")
            logging.info(f"🔹 Rejected Epochs: {stats['rejected_epochs']}")
            logging.info(f"🔹 Avg Removed Amplitude: {stats['avg_removed_amp'] * 1e6:.2f} μV" if stats["avg_removed_amp"] is not None else "🔹 Avg Removed Amplitude: N/A")
            logging.info(f"🔹 Kept Data Ratio: {100 * (len(epochs) / num_events):.2f}%")

            with open(self.excluded_dir / "skipped_files_log.txt", "w") as f:
                for file, reason in skipped_files:
                    f.write(f"{file}, {reason}\n")

            logging.info(f"📄 Skipped files log saved: skipped_files_log.txt")

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

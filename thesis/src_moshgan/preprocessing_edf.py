import csv
import logging
import os
import glob
from pathlib import Path
import mne
import numpy as np
from tqdm import tqdm
from pyprep import NoisyChannels
from mne.preprocessing import ICA
from mne_icalabel import label_components

class EEGPreprocessor:

    def __init__(self, excluded_dir, epoch_size, sample_size, epoch_rejection_threshold, ica_threshold, SOURCE_FOLDER, TARGET_FOLDER, SNR_TARGET):
        self.excluded_dir = excluded_dir
        self.epoch_size = epoch_size
        self.sample_size = sample_size
        self.epoch_rejection_threshold = epoch_rejection_threshold
        self.ica_threshold = ica_threshold
        self.SOURCE_FOLDER = SOURCE_FOLDER
        self.TARGET_FOLDER = TARGET_FOLDER
        self.SNR_TARGET = SNR_TARGET
        logging.info("EEG Preprocessor Initialized")

    def prepare_raw_data(self, raw: mne.io.Raw):
        print("\n🛠️ Preparing raw data before preprocessing...")

        # ✅ Step 1: Fix NaN or Inf values (replace with mean)
        print("🔹 Step 1: Fixing NaN/Inf values...")
        data = raw.get_data()
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            for ch in range(data.shape[0]):
                nan_mask = np.isnan(data[ch])
                inf_mask = np.isinf(data[ch])
                if np.any(nan_mask) or np.any(inf_mask):
                    mean_value = np.nanmean(data[ch])
                    data[ch, nan_mask] = mean_value
                    data[ch, inf_mask] = mean_value
                    print(f"✅ Fixed NaN/Inf values in channel {raw.ch_names[ch]}")
            raw._data = data

        # ✅ Step 2: Remove DC Offset (Baseline Drift)
        print("🔹 Step 2: Removing DC Offset...")
        raw._data -= np.mean(raw._data, axis=1, keepdims=True)
        print(f"✅ DC Offset removed")

        # ✅ Step 3: Resample to 256 Hz if needed
        target_sampling_rate = 256
        if raw.info['sfreq'] > target_sampling_rate:
            print(f"🔹 Step 3: Resampling to {target_sampling_rate} Hz...")
            raw.resample(target_sampling_rate)
            print(f"✅ Resampled to {target_sampling_rate} Hz")

        # ✅ Step 4: Set Montage Early to Avoid ICA Issues
        print("🔹 Step 4: Setting Montage...")
        try:
            montage = mne.channels.make_standard_montage("standard_1020")
            raw.set_montage(montage, on_missing="ignore")
            print(f"✅ Montage set to standard_1020")
        except Exception as e:
            print(f"⚠️ Failed to set montage: {e}")

        print("✅ Preparation Complete!")


    def apply_prep_pipeline(self, raw: mne.io.Raw):
        stats = {"bad_channels": 0, "rejected_epochs": 0, "avg_removed_amp": [], "kept_epoch_amplitudes": []}

        print("\n🧪 Starting Preprocessing...")

        # ✅ Step 1: Remove Non-EEG Channels (Immediately After Loading)
        print("🔹 Step 1: Removing Non-EEG Channels...")

        non_eeg_channels = ['Photic-REF', 'Photic', 'IBI', 'Bursts', 'Suppr', 'Pulse Rate']
        ica_aux_channels = ['EOG AOG', 'ECG EKG']

        # Remove non-EEG channels (keep EOG and ECG for ICA)
        raw.drop_channels([ch for ch in non_eeg_channels if ch in raw.ch_names])
        print(f"✅ Dropped irrelevant channels: {non_eeg_channels}")

        # ✅ Step 2: Remove Line Noise using Multi-Taper Spectrum Fit
        print("🔹 Step 2: Removing Line Noise using Multi-Taper Spectrum Fit...")
        raw.notch_filter(freqs=[50], fir_design='firwin', method='spectrum_fit')

        # ✅ Step 3: High-pass filter at 0.5 Hz
        print("🔹 Step 3: High-Pass Filtering (0.5 Hz)...")
        raw.filter(l_freq=0.5, h_freq=40, fir_design='firwin')

        # ✅ Step 4: Detect bad channels using PyPREP
        print("🔹 Step 4: Detecting Bad Channels using PyPREP...")

        eeg_only_raw = raw.copy()
        eeg_only_raw.drop_channels([ch for ch in ica_aux_channels if ch in eeg_only_raw.ch_names])  # Exclude EOG/ECG

        prep_handler = NoisyChannels(eeg_only_raw)
        try:
            prep_handler.find_bad_by_nan_flat()
            prep_handler.find_bad_by_deviation()
            prep_handler.find_bad_by_correlation(
                correlation_secs=1.5,
                correlation_threshold=0.45,  # Increased threshold
                frac_bad=0.03                # Increased from 0.03
            )
            np.random.seed(42)
            prep_handler.find_bad_by_ransac(
                            n_samples=self.sample_size, sample_prop=0.25, corr_thresh=0.75,
                            frac_bad=0.4, corr_window_secs=5, channel_wise=False
                        )
        except Exception as e:
            print(f"⚠️ Error in bad channel detection: {e}")

        bad_channels = prep_handler.get_bads()
        stats["bad_channels"] = len(bad_channels)

        print(f"\n⚠️ Bad channels detected: {bad_channels}")
        if len(bad_channels) <= int(len(raw.ch_names) * 0.2):
            raw.info['bads'] = bad_channels
            raw.interpolate_bads(reset_bads=True)
            print(f"✅ Interpolated {len(bad_channels)} channels")
        else:
            print(f"❌ Too many bad channels ({len(bad_channels)}). Skipping interpolation.")

        # ✅ Step 5: Re-reference after detecting bad channels
        good_channels = [ch for ch in raw.ch_names if ch not in bad_channels]
        if len(good_channels) > 0:
            raw.set_eeg_reference(ref_channels=good_channels)
            print(f"✅ Re-referenced using {len(good_channels)} good channels.")

        # ✅ Step 6: ICA for artifact detection
        print("🔹 Step 6: Running ICA to Detect Artifacts...")

        try:
            # ✅ Prepare raw data for ICA (filtering)
            ica_raw = raw.copy()
            ica_raw.filter(l_freq=1, h_freq=100, fir_design='firwin')

            # ✅ Keep EOG and ECG channels for ICA analysis
            ica_raw.drop_channels([ch for ch in non_eeg_channels if ch in ica_raw.ch_names])

            picks = mne.pick_types(ica_raw.info, eeg=True, eog=True, ecg=True)
            if len(picks) < 10:
                print(f"❌ Too few EEG channels ({len(picks)}) — skipping ICA.")
                return raw, stats

            ica_raw.set_montage('standard_1020', on_missing="ignore")
            ica_raw.set_eeg_reference(ref_channels="average", projection=False)

            # ✅ Compute rank for ICA
            rank = mne.compute_rank(ica_raw)
            n_components = rank.get("eeg", 0.9999)

            # ✅ Fit ICA model
            ica = ICA(
                n_components=0.9999,
                method='infomax',
                fit_params=dict(extended=True),
                random_state=42,
                max_iter="auto"
            )
            ica.fit(ica_raw, picks=picks)

            # ✅ Find ECG artifacts directly using correlation (if channel exists and is valid)
            if 'ECG EKG' in ica_raw.ch_names:
                try:
                    ecg_indices, _ = ica.find_bads_ecg(ica_raw, method="correlation", threshold="auto")
                    if ecg_indices:
                        ica.exclude.extend(ecg_indices)
                        print(f"✅ Detected and removed {len(ecg_indices)} ECG artifacts")
                except Exception as e:
                    print(f"❌ Error in ECG artifact detection: {e}")
            else:
                print("⚠️ No valid ECG channels found or channel is flat — skipping ECG detection.")

            # ✅ Detect EOG artifacts using correlation (if channel exists and is valid)
            if 'EOG AOG' in ica_raw.ch_names:
                try:
                    eog_indices, _ = ica.find_bads_eog(ica_raw, threshold=0.9)
                    if eog_indices:
                        ica.exclude.extend(eog_indices)
                        print(f"✅ Detected and removed {len(eog_indices)} EOG artifacts")
                except Exception as e:
                    print(f"❌ Error in EOG artifact detection: {e}")
            else:
                print("⚠️ No valid EOG channels found or channel is flat — skipping EOG detection.")

            # ✅ Detect muscle artifacts using z-score threshold
            muscle_indices, _ = ica.find_bads_muscle(ica_raw, threshold=self.ica_threshold)
            ica.exclude.extend(muscle_indices)

            # ✅ Apply ICA
            ica.apply(raw)
            print(f"✅ Applied ICA — Total components removed: {len(ica.exclude)}")

            # ✅ Step 7: SNR Calculation
            print("🔹 Step 7: Calculating SNR...")
            raw_data = raw.get_data()
            preprocessed_data = ica_raw.get_data()

            # ✅ Remove DC offset (drift)
            raw_data -= np.mean(raw_data, axis=1, keepdims=True)
            preprocessed_data -= np.mean(preprocessed_data, axis=1, keepdims=True)

            # ✅ Global SNR
            signal_power = np.mean(preprocessed_data ** 2)
            noise_power = np.mean((preprocessed_data - raw_data) ** 2)
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

            print(f"📊 SNR (Global): {round(snr, 2)} dB")
            with open(self.SNR_TARGET, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([raw.filenames[0] if raw.filenames else "unknown", round(snr, 2)])

        except Exception as e:
            print(f"❌ Error in ICA: {e}")

        # ✅ Step 8: Drop auxiliary channels after ICA
        raw.drop_channels([ch for ch in ica_aux_channels if ch in raw.ch_names])
        print(f"✅ Dropped auxiliary channels after ICA: {ica_aux_channels}")

        # ✅ Step 9: Remove existing artifact annotations
        if len(raw.annotations) > 0:
            raw.annotations.delete(
                [i for i, desc in enumerate(raw.annotations.description) if "BAD_" in desc]
            )
            print(f"✅ Removed {len(raw.annotations)} artifact annotations")

        print("✅ Preprocessing Complete!")

        return raw, stats


    def process(self):
        # ✅ Setup logging
        logging.basicConfig(force=True, format="%(levelname)s - %(name)s - %(message)s", level=logging.INFO)
        mne.set_log_level("WARNING")

        # ✅ Find all .fif files in the folder
        fif_files = glob.glob(os.path.join(self.SOURCE_FOLDER, "*.fif"))
        if not fif_files:
            logging.error("❌ No FIF files found.")
            return

        processed_count = 0
        failed_count = 0

        # ✅ Loop through each file
        for file in tqdm(fif_files, desc="Processing Files"):
            try:
                # ✅ Load raw EEG data
                logging.info(f"🔎 Loading file: {file}")
                raw = mne.io.read_raw_fif(file, preload=True)

                # ✅ Set montage BEFORE preprocessing to fix NaN issues
                try:
                    montage = mne.channels.make_standard_montage("standard_1020")
                    raw.set_montage(montage, on_missing="ignore")
                    # raw.info['dig'] = None
                    logging.info("✅ Montage set to standard_1020.")
                except Exception as e:
                    logging.warning(f"⚠️ Failed to set montage: {e}")

                # ✅ Apply preprocessing pipeline
                # raw.plot(block=True)
                raw, stats = self.apply_prep_pipeline(raw)

                # ✅ Re-reference to common average reference (without projection)
                try:
                    raw.set_eeg_reference('average', projection=False)
                    logging.info("✅ Re-referenced to common average.")
                except Exception as e:
                    logging.warning(f"⚠️ Failed to set reference: {e}")

                # ✅ Save processed file to target folder
                base_name = os.path.basename(file).replace(".fif", "_preprocessed_raw.fif")
                target_file = os.path.join(self.TARGET_FOLDER, base_name)
                raw.save(target_file, overwrite=True)
                logging.info(f"✅ Saved processed file to: {target_file}")

                processed_count += 1

            except FileNotFoundError:
                logging.error(f"❌ File not found: {file}")
                failed_count += 1
                continue

            except ValueError as e:
                logging.error(f"❌ Value error while processing {file}: {e}")
                failed_count += 1
                continue

            except Exception as e:
                logging.error(f"❌ Failed to process {file}: {e}")
                failed_count += 1
                continue

        # ✅ Summary Report
        print("\n📊 SUMMARY REPORT:")
        print(f"➡️ Total files found: {len(fif_files)}")
        print(f"✅ Successfully processed files: {processed_count}")
        print(f"❌ Failed to process files: {failed_count}")

        if failed_count > 0:
            logging.warning(f"⚠️ {failed_count} files failed during processing.")

        print("\n🚀 All files processed!")

# ✅ Create an instance of the preprocessor and run it
if __name__ == "__main__":
    preprocessor = EEGPreprocessor(
                                None,
                                1.0,
                                50,
                                150,
                                0.5,
                                Path("L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\thesis\\src_moshgan\\mapped_5mv_5fv_3resting"),
                                Path("L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\thesis\\src_moshgan\\Preprocessed_files")
                            )
    preprocessor.process()

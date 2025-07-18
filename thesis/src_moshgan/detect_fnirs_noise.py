import glob
import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import mne
from tqdm import tqdm
import traceback
import time

# === Logging Configuration ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("psd_fif_processing.log", mode='w', encoding='utf-8')
    ]
)
log = logging.getLogger(__name__)
start_time = time.time()

# === Configuration ===
root_dir = Path("L:/LovbeskyttetMapper/CONNECT-ME/CONMED3/Dataoptagelser/NIRS-EEG/")
extension = ".xdf"
window_width = 0.5
candidate_fundamentals = np.arange(5, 7, 0.1)
n_peaks = 6
harmonic_noise_ratio_threshold = 0.3  # can tune later based on histogram
required_harmonics_above_threshold = 2

# === Create folders ===
os.makedirs("psd_plots_fif/noise", exist_ok=True)
os.makedirs("psd_plots_fif/clean", exist_ok=True)
os.makedirs("psd_plots_fif/csv_files", exist_ok=True)
os.makedirs("psd_plots_fif/unknown", exist_ok=True)

# === Utility ===
def compute_psd_band_powers(raw, freqs_to_check, window_width=0.5):
    psd = raw.compute_psd(fmin=1, fmax=max(freqs_to_check) + 5)
    psd_data = psd.get_data()
    freqs = psd.freqs
    band_powers = {}
    for target in freqs_to_check:
        idx = np.where((freqs >= target - window_width) & (freqs <= target + window_width))[0]
        if len(idx) > 0:
            band_power = psd_data[:, idx].mean(axis=1) * 1e12
            band_powers[round(target, 1)] = band_power
        else:
            band_powers[round(target, 1)] = np.zeros(psd_data.shape[0])
    return band_powers, freqs

# === Main Script ===
peak_power_accumulator = defaultdict(list)
noise_files = []
clean_files = []
summary_stats = []
max_ratios = []
success_count, fail_count = 0, 0

fif_files = glob.glob(os.path.join(Path("L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\results_moshgan\\run_20250424_2058\\Preprocessd_files"), "*.fif"))

if not fif_files:
    log.warning("No files found.")
else:
    for idx, file in enumerate(tqdm(fif_files, desc="Processing Files"), 1):
        log.info(f"[{idx}/{len(fif_files)}] Processing file: {file}")
        try:
            raw = mne.io.read_raw_fif(file, preload=True)
            raw.pick("eeg")  # keep only EEG channels
            montage = mne.channels.make_standard_montage("standard_1020")
            raw.set_montage(montage, on_missing="ignore")

            fig = raw.compute_psd(fmin=1, fmax=45).plot(show=False)
            plot_name = Path(file).stem + ".png"
            fig_path = os.path.join("psd_plots_fif", "unknown", plot_name)
            fig.savefig(fig_path)
            plt.close(fig)

            best_freq = None
            best_band_powers = None
            best_harmonics = None
            max_channel_ratio = 0

            total_psd = raw.compute_psd(fmin=1, fmax=45)
            total_power_per_channel = total_psd.get_data().mean(axis=1) * 1e12

            for base_freq in candidate_fundamentals:
                harmonics = [base_freq * i for i in range(1, n_peaks + 1) if base_freq * i < 45]
                band_powers, _ = compute_psd_band_powers(raw, harmonics, window_width)

                harmonic_matrix = np.stack([band_powers[f] for f in band_powers])
                channel_harmonic_power = harmonic_matrix
                channel_ratios = channel_harmonic_power / total_power_per_channel[None, :]

                harmonic_above_thresh = (channel_ratios > harmonic_noise_ratio_threshold).sum(axis=0)
                noisy_channels = harmonic_above_thresh >= required_harmonics_above_threshold
                this_max_ratio = channel_ratios.max()

                if noisy_channels.any() and this_max_ratio > max_channel_ratio:
                    max_channel_ratio = this_max_ratio
                    best_freq = base_freq
                    best_band_powers = band_powers
                    best_harmonics = harmonics
                    best_channel_ratios = channel_ratios

            if best_harmonics is None:
                log.warning(f"No harmonics found in {file}")
                clean_files.append({"file": str(file)})
                new_fig_path = os.path.join("psd_plots_fif/clean", plot_name)
                os.replace(fig_path, new_fig_path)
                success_count += 1
                continue

            file_has_noise = (best_channel_ratios > harmonic_noise_ratio_threshold).sum(axis=0).max() >= required_harmonics_above_threshold
            row_data = {
                "file": str(file),
                "fundamental_freq": best_freq,
                "max_channel_ratio": max_channel_ratio,
                "total_power_mean": total_power_per_channel.mean()
            }

            max_ratios.append(max_channel_ratio)

            for f in best_harmonics:
                power_values = best_band_powers[round(f, 1)]
                peak_power_accumulator[f].append(power_values.mean())
                row_data[f"{f:.1f}_mean"] = power_values.mean()
                row_data[f"{f:.1f}_std"] = power_values.std()

                for ch_name, val in zip(raw.ch_names, power_values):
                    row_data[f"{f:.1f}_{ch_name}"] = val

            if file_has_noise:
                noise_files.append(row_data)
                new_fig_path = os.path.join("psd_plots_fif/noise", plot_name)
            else:
                clean_files.append({"file": str(file)})
                new_fig_path = os.path.join("psd_plots_fif/clean", plot_name)

            os.replace(fig_path, new_fig_path)
            success_count += 1
            result = "NOISY" if file_has_noise else "CLEAN"
            log.info(f"File processed: {Path(file).name} — Best freq: {best_freq:.1f} Hz — Max Channel Ratio: {max_channel_ratio:.3f} — Result: {result}")

        except Exception as e:
            log.error(f"Skipping {file} — {type(e).__name__}: {e}")
            log.exception("Detailed error:")
            fail_count += 1

    pd.DataFrame(noise_files).to_csv("psd_plots_fif/csv_files/eeg_noise_detected_files.csv", index=False)
    pd.DataFrame(clean_files).to_csv("psd_plots_fif/csv_files/eeg_clean_files.csv", index=False)

    for f, values in peak_power_accumulator.items():
        powers = np.array(values)
        if len(powers):
            summary_stats.append({
                "frequency": f,
                "mean": powers.mean(),
                "std": powers.std(),
                "min": powers.min(),
                "max": powers.max()
            })
            log.info(f"  - {f:.1f} Hz: {powers.mean():.4f} ± {powers.std():.4f} pW (range {powers.min():.4f} – {powers.max():.4f} pW)")
        else:
            log.info(f"  - {f:.1f} Hz: no data")

    pd.DataFrame(summary_stats).to_csv("psd_plots_fif/csv_files/eeg_noise_frequency_summary.csv", index=False)

    if max_ratios:
        log.info("Max Channel Ratio Stats:")
        log.info(f"Mean: {np.mean(max_ratios):.3f}")
        log.info(f"25th Percentile: {np.percentile(max_ratios, 25):.3f}")
        log.info(f"75th Percentile: {np.percentile(max_ratios, 75):.3f}")

    elapsed = time.time() - start_time
    log.info("Summary:")
    log.info(f"Successfully processed: {success_count}")
    log.info(f"Failed to load: {fail_count}")
    log.info(f"Total runtime: {elapsed:.2f} seconds")
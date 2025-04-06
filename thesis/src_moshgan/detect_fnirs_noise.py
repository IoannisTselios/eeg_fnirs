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

# === Configuration ===
root_dir = Path("L:/LovbeskyttetMapper/CONNECT-ME/CONMED3/Dataoptagelser/NIRS-EEG/")
extension = ".xdf"
window_width = 0.5
candidate_fundamentals = np.arange(9.5, 10.6, 0.1)  # Test 9.5 to 10.5 Hz in 0.1 steps
n_peaks = 6
noise_threshold_pW = 100  # Adjust threshold as needed

# === Utility ===
def compute_psd_band_powers(raw, freqs_to_check, window_width=0.5):
    psd = raw.compute_psd(fmin=1, fmax=max(freqs_to_check) + 5)
    psd_data = psd.get_data()  # shape: (n_channels, n_freqs)
    freqs = psd.freqs

    band_powers = {}
    for target in freqs_to_check:
        idx = np.where((freqs >= target - window_width) & (freqs <= target + window_width))[0]
        if len(idx) > 0:
            band_power = psd_data[:, idx].mean(axis=1) * 1e12  # Convert to pW
            band_powers[round(target, 1)] = band_power
        else:
            band_powers[round(target, 1)] = np.zeros(psd_data.shape[0])
    return band_powers, freqs

# === Main Script ===
peak_power_accumulator = defaultdict(list)
noise_files = []
clean_files = []
summary_stats = []
success_count, fail_count = 0, 0

# ✅ Find all .fif files in the folder
fif_files = glob.glob(os.path.join(Path("L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\thesis\\src_moshgan\\mapped_5mv_5fv_3resting"), "*.fif"))

if not fif_files:
    print("⚠️ No files found.")
else:
    for file in tqdm(fif_files, desc="Processing Files"):
        print(f"\n🔍 Processing file: {file}")
        try:
            raw = mne.io.read_raw_fif(file, preload=True)
            montage = mne.channels.make_standard_montage("standard_1020")
            raw.set_montage(montage, on_missing="ignore")

            best_freq = None
            max_above_thresh = 0
            best_band_powers = None
            best_harmonics = None

            # Try all candidate fundamentals and pick the one with strongest harmonic noise
            for base_freq in candidate_fundamentals:
                harmonics = []
                for i in range(1, n_peaks + 1):
                    h = base_freq * i
                    if h >= 45:
                        break
                    harmonics.append(h)

                band_powers, _ = compute_psd_band_powers(raw, harmonics, window_width)

                above_thresh = sum(
                    (vals > noise_threshold_pW).sum()
                    for vals in band_powers.values()
                )

                if above_thresh > max_above_thresh:
                    max_above_thresh = above_thresh
                    best_freq = base_freq
                    best_band_powers = band_powers
                    best_harmonics = harmonics

            # Handle case where no harmonics were selected
            if best_harmonics is None:
                print(f"⚠️ No harmonic noise detected in {file}")
                clean_files.append({"file": str(file)})
                success_count += 1
                continue

            # Save per-file values
            file_has_noise = max_above_thresh > 0
            row_data = {"file": str(file), "fundamental_freq": best_freq}

            for f in best_harmonics:
                power_values = best_band_powers[round(f, 1)]
                peak_power_accumulator[f].append(power_values.mean())
                row_data[f"{f:.1f}_mean"] = power_values.mean()
                row_data[f"{f:.1f}_std"] = power_values.std()

                for ch_name, val in zip(raw.ch_names, power_values):
                    row_data[f"{f:.1f}_{ch_name}"] = val

            if file_has_noise:
                noise_files.append(row_data)
            else:
                clean_files.append({"file": str(file)})

            success_count += 1
            print(f"✅ File processed (best freq: {best_freq:.1f} Hz)")

        except Exception as e:
            print(f"❌ Skipping {file} — {type(e).__name__}: {e}")
            traceback.print_exc()
            fail_count += 1

    # === Save Results ===
    pd.DataFrame(noise_files).to_csv("eeg_noise_detected_files.csv", index=False)
    pd.DataFrame(clean_files).to_csv("eeg_clean_files.csv", index=False)

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
            print(f"  - {f:.1f} Hz: {powers.mean():.4f} ± {powers.std():.4f} pW (range {powers.min():.4f} – {powers.max():.4f} pW)")
        else:
            print(f"  - {f:.1f} Hz: no data")

    pd.DataFrame(summary_stats).to_csv("eeg_noise_frequency_summary.csv", index=False)

    print(f"\n📊 Summary:")
    print(f"✅ Successfully processed: {success_count}")
    print(f"❌ Failed to load: {fail_count}")

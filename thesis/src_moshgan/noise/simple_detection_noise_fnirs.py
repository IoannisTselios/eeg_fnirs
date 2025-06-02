import glob
import os
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import mne
from tqdm import tqdm
import logging
import time
from scipy.signal import find_peaks

# === Logging Configuration ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("psd_processing.log", mode='w')
    ]
)
log = logging.getLogger(__name__)
start_time = time.time()

# === Folder Setup ===
fif_dir = Path("L:/LovbeskyttetMapper/CONNECT-ME/Ioannis/thesis_code/results_moshgan/run_20250424_2058/Preprocessd_files")
fif_files = glob.glob(str(fif_dir / "*.fif"))

os.makedirs("psd_plots/noisee", exist_ok=True)
os.makedirs("psd_plots/csv_files", exist_ok=True)

noise_files = []
success_count, fail_count = 0, 0

if not fif_files:
    log.warning("No files found.")
else:
    for idx, file in enumerate(tqdm(fif_files, desc="Processing Files"), 1):
        try:
            file_path = Path(file)
            filename = file_path.stem

            log.info(f"[{idx}/{len(fif_files)}] Processing file: {file_path}")

            raw = mne.io.read_raw_fif(file_path, preload=True)
            psd = raw.compute_psd(fmin=1, fmax=45)
            psd_data = psd.get_data() * 1e12  # Convert to pW
            freqs = psd.freqs
            mean_psd = psd_data.max(axis=0)

            # Detect peaks (all, initially)
            peaks, _ = find_peaks(mean_psd, prominence=(None, None))
            peak_freqs = freqs[peaks]
            peak_powers = mean_psd[peaks]

            # === Step 1: Show PSD ===
            fig = raw.compute_psd(fmin=1, fmax=45).plot(show=False)
            ax = fig.axes[0]

            print(f"\n🧠 File: {filename}")
            print("🔍 Click directly on each noisy peak.")
            print("👉 Press [Enter] when done.")

            plt.show(block=False)
            clicks = plt.ginput(n=-1, timeout=0)
            plt.close(fig)

            # === Step 2: Match each click to nearest detected peak ===
            peaks, _ = find_peaks(mean_psd, prominence=(None, None))
            peak_freqs = freqs[peaks]
            peak_powers = mean_psd[peaks]

            selected_peaks = []

            for click in clicks:
                click_freq = click[0]
                closest_idx = np.argmin(np.abs(peak_freqs - click_freq))
                selected_freq = round(peak_freqs[closest_idx], 2)
                selected_power = round(peak_powers[closest_idx], 2)
                selected_peaks.append((selected_freq, selected_power))

            # === Step 3: Replot with selected peaks highlighted ===
            fig = raw.compute_psd(fmin=1, fmax=45).plot(show=False)
            ax = fig.axes[0]

            for i, (f, p) in enumerate(selected_peaks):
                ax.scatter(f, p, color='red', zorder=5)

                if i % 2 == 0:
                    # Even index → put Hz above
                    ax.text(f, p + 1.05, f"{f:.1f} Hz", ha='center', va='bottom', fontsize=9, color='red')
                    ax.text(f, p + 1.05, f"{p:.1f} pW", ha='center', va='top', fontsize=9, color='red')
                else:
                    # Odd index → put pW below
                    ax.text(f, p - 10.5, f"{f:.1f} Hz", ha='center', va='bottom', fontsize=9, color='red')
                    ax.text(f, p - 10.5, f"{p:.1f} pW", ha='center', va='top', fontsize=9, color='red')


            fig_path = Path("psd_plots/noisee") / f"{filename}.png"
            fig.savefig(fig_path)
            plt.close(fig)

            # === Save row ===
            row_data = {"file": str(file_path)}
            row_data["manual_noise_peaks"] = "; ".join([f"{f}-{p}" for f, p in selected_peaks])
            noise_files.append(row_data)

            success_count += 1

        except Exception as e:
            log.error(f"Skipping {file_path} due to error: {e}")
            log.exception("Detailed error:")
            fail_count += 1

# === Save CSV Summary ===
csv_path = "psd_plots/csv_files/eeg_noise_detected_files.csv"
pd.DataFrame(noise_files).to_csv(csv_path, index=False)

elapsed = time.time() - start_time
log.info("\n Summary:")
log.info(f" Successfully processed: {success_count}")
log.info(f" Failed to load: {fail_count}")
log.info(f" Total runtime: {elapsed:.2f} seconds")

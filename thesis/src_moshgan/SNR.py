import os
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paths
RAW_FOLDER = r"L:\LovbeskyttetMapper\CONNECT-ME\Ioannis\thesis_code\EDF filer"
PREPROCESSED_FOLDER = r"L:\LovbeskyttetMapper\CONNECT-ME\Ioannis\thesis_code\thesis\src_moshgan\Preprocessed_files"
OUTPUT_FOLDER = r"L:\LovbeskyttetMapper\CONNECT-ME\Ioannis\thesis_code\results"

# Create output folder if missing
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Get list of raw and preprocessed files
raw_files = [f.replace('.fif', '') for f in os.listdir(RAW_FOLDER) if f.endswith('.fif')]
preprocessed_files = [
    f.replace('_preprocessed_raw', '').replace('.fif', '')
    for f in os.listdir(PREPROCESSED_FOLDER) if f.endswith('.fif')
]

# Suppress file name warning
mne.set_log_level("ERROR")

# Track missing files and SNR values
missing_files = []
snr_results = []
temp = 0
MAX_FILES = 800

for raw_file in raw_files:
    if raw_file not in preprocessed_files:
        missing_files.append(f"Raw file: {raw_file} → Expected preprocessed file: {raw_file}_preprocessed_raw")
    else:
        try:
            # Load raw and preprocessed files
            raw_path = os.path.join(RAW_FOLDER, raw_file + '.fif')
            preprocessed_path = os.path.join(PREPROCESSED_FOLDER, raw_file + '_preprocessed_raw.fif')

            if os.path.exists(preprocessed_path):
                print(f"\n🚀 Processing {raw_file}...")

                # Load data
                raw = mne.io.read_raw_fif(raw_path, preload=True)
                preprocessed = mne.io.read_raw_fif(preprocessed_path, preload=True)

                # ✅ Align channels between raw and preprocessed
                common_channels = np.intersect1d(raw.ch_names, preprocessed.ch_names)

                raw.pick_channels(common_channels)
                preprocessed.pick_channels(common_channels)

                # ✅ Make sure channel order is identical
                raw.reorder_channels(common_channels)
                preprocessed.reorder_channels(common_channels)

                # ✅ Extract raw and preprocessed data
                raw_data = raw.get_data()
                preprocessed_data = preprocessed.get_data()

                # ✅ Remove DC offset (mean) to fix drift issues
                raw_data -= np.mean(raw_data, axis=1, keepdims=True)
                preprocessed_data -= np.mean(preprocessed_data, axis=1, keepdims=True)

                # ---- 1️⃣ Compute Signal-to-Noise Ratio (Global) ----
                signal_power = np.mean(preprocessed_data ** 2)
                noise_power = np.mean((preprocessed_data - raw_data) ** 2)
                snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

                # ---- 2️⃣ Compute Channel-Wise SNR ----
                channel_snr = []
                for ch in range(preprocessed_data.shape[0]):
                    signal_power_ch = np.mean(preprocessed_data[ch, :] ** 2)
                    noise_power_ch = np.mean((preprocessed_data[ch, :] - raw_data[ch, :]) ** 2)
                    snr_ch = 10 * np.log10(signal_power_ch / noise_power_ch) if noise_power_ch > 0 else float('inf')
                    channel_snr.append(round(snr_ch, 2))

                # ---- 3️⃣ Compute Summary Statistics ----
                mean_raw = np.mean(raw_data)
                mean_preprocessed = np.mean(preprocessed_data)
                std_raw = np.std(raw_data)
                std_preprocessed = np.std(preprocessed_data)

                # ---- 4️⃣ Artifact Detection ----
                try:
                    bad_channels = mne.preprocessing.find_bad_channels_maxwell(raw)
                except Exception as e:
                    bad_channels = f"Failed to detect: {e}"

                # ---- 5️⃣ Save Results ----
                snr_results.append({
                    'File': raw_file,
                    'SNR (dB)': round(snr, 2),
                    'Mean Raw': round(mean_raw, 4),
                    'Mean Preprocessed': round(mean_preprocessed, 4),
                    'Std Raw': round(std_raw, 4),
                    'Std Preprocessed': round(std_preprocessed, 4),
                    'Channel-wise SNR': channel_snr,
                    'Bad Channels': bad_channels if isinstance(bad_channels, list) else 'None'
                })

                # ---- 6️⃣ Plot Raw vs Preprocessed ----
                plt.figure(figsize=(12, 6))
                plt.plot(raw.times[:500], raw_data[0, :500], label='Raw', alpha=0.6)
                plt.plot(preprocessed.times[:500], preprocessed_data[0, :500], label='Preprocessed', alpha=0.6)
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude (µV)')
                plt.legend()
                plt.title(f'Raw vs Preprocessed EEG Signal ({raw_file})')
                plt.savefig(os.path.join(OUTPUT_FOLDER, f"{raw_file}_comparison.png"))
                plt.close()

                # ---- 7️⃣ Power Spectral Density (PSD) ----
                plt.figure(figsize=(12, 6))
                raw.plot_psd(fmax=60, average=True, show=False)
                preprocessed.plot_psd(fmax=60, average=True, show=False)
                plt.title(f'PSD Comparison: Raw vs Preprocessed ({raw_file})')
                plt.savefig(os.path.join(OUTPUT_FOLDER, f"{raw_file}_psd_comparison.png"))
                plt.close()

        except Exception as e:
            print(f"❌ Error processing {raw_file}: {e}")
            snr_results.append({
                'File': raw_file,
                'SNR (dB)': 'ERROR'
            })

    if temp >= MAX_FILES:
        break
    temp += 1

# ✅ Report SNR Results
snr_df = pd.DataFrame(snr_results)
print("\n📊 SNR Results:")
print(snr_df)

# Save Results to CSV
snr_report_file = os.path.join(OUTPUT_FOLDER, "snr_report.csv")
snr_df.to_csv(snr_report_file, index=False)
print(f"\n✅ SNR report saved to: {snr_report_file}")

# ✅ Report Missing Files
if missing_files:
    print("\n❌ MISSING FILES:")
    for file in missing_files:
        print(f"➡️ {file}")
else:
    print("\n✅ No missing files!")

# ✅ All done
print("\n🚀 Done!")

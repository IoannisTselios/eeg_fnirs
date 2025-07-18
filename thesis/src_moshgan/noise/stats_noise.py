import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mne.viz import plot_topomap
from mne.channels import make_standard_montage
import mne

# === Load Data ===
noise_df = pd.read_csv("psd_plots_fif/csv_files/eeg_noise_detected_files.csv")

if os.path.getsize("psd_plots_fif/csv_files/eeg_clean_files.csv") > 0:
    clean_df = pd.read_csv("psd_plots_fif/csv_files/eeg_clean_files.csv")
else:
    clean_df = pd.DataFrame()
    print("⚠️ eeg_clean_files.csv is empty. No clean files found.")

# === Summary Stats ===
total_files = len(noise_df) + len(clean_df)
print(f"📁 Total EEG Files: {total_files}")
print(f"✅ Clean Files: {len(clean_df)} ({len(clean_df) / total_files:.2%})")
print(f"⚠️ Noisy Files: {len(noise_df)} ({len(noise_df) / total_files:.2%})")

# === Frequency Stats ===
freq_cols = [col for col in noise_df.columns if "_mean" in col]
summary_freq = []

for col in freq_cols:
    freq = col.split("_")[0]
    power_vals = noise_df[col]
    summary_freq.append({
        "Frequency (Hz)": freq,
        "Mean Power (pW)": round(power_vals.mean(), 2),
        "Std Dev (pW)": round(power_vals.std(), 2),
        "Files Above 100 pW": (power_vals > 100).sum(),
    })

freq_df = pd.DataFrame(summary_freq)
freq_df.to_csv("frequency_summary.csv", index=False)

# === Channel × Frequency Stats ===
channel_summary = []

for freq in [col.split("_")[0] for col in freq_cols]:
    channel_cols = [c for c in noise_df.columns if c.startswith(freq) and "_" in c[len(freq):]]
    for ch_col in channel_cols:
        ch_name = ch_col.split("_", 1)[1]
        values = noise_df[ch_col]
        channel_summary.append({
            "Channel": ch_name,
            "Frequency": freq,
            "Mean Power (pW)": round(values.mean(), 2),
            "Files > 100 pW": (values > 100).sum(),
            "Detectable in All Files": (values > 100).all()
        })

channel_df = pd.DataFrame(channel_summary)
channel_df.to_csv("channel_noise_summary.csv", index=False)

# === Plot 1: Frequency Count Bar Plot ===
plt.figure(figsize=(10, 5))
sns.barplot(data=freq_df, x="Frequency (Hz)", y="Files Above 100 pW", color='skyblue')
plt.title("Files Above 100 pW per Frequency")
plt.ylabel("Count")
plt.xlabel("Frequency (Hz)")
plt.tight_layout()
plt.show()

# === Plot 3: Topomap per Frequency ===
threshold = 100  # pW
montage = make_standard_montage("standard_1020")

for freq_focus in sorted(set(col.split("_")[0] for col in noise_df.columns if "_" in col)):
    cols = [c for c in noise_df.columns if c.startswith(freq_focus) and "_" in c]
    channel_values = noise_df[cols].mean(axis=0)

    # Map: {channel_name: mean_power}
    values_dict = {col.split("_", 1)[1]: val for col, val in channel_values.items()}
    valid_ch_names = [ch for ch in values_dict if ch in montage.ch_names]
    vals = np.array([values_dict[ch] for ch in valid_ch_names])

    # Get positions and mask
    info = mne.create_info(ch_names=valid_ch_names, sfreq=100, ch_types="eeg")
    info.set_montage(montage)
    pos = np.array([info['chs'][info.ch_names.index(ch)]['loc'][:2] for ch in valid_ch_names])
    mask = vals > threshold  # Highlight only high power channels

    title_text = f"{freq_focus} Hz — Highlighted: {np.sum(mask)} channels > 100 pW"
    print("🧠 Plot:", title_text)  # ✅ Console output

    fig, ax = plt.subplots(figsize=(6, 5))
    mask_params = dict(marker='o', markerfacecolor='none', markeredgecolor='black',
                    linewidth=0, markersize=12)
    plot_topomap(vals, pos, axes=ax, names=valid_ch_names,
                mask=mask, mask_params=mask_params,
                cmap="Reds", contours=0)
    ax.set_title(title_text)
    plt.tight_layout()
    plt.show()



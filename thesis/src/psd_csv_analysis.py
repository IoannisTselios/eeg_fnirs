import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# Path to the PSD CSV files
csv_output_dir = Path(r'L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\results\\run_20250402_1756\\csv_outputs')

# Collect all PSD files
psd_files = list(csv_output_dir.glob("EEG_PSD_Values_*.csv"))

# Combine all into a single DataFrame
all_psd = []
for file in psd_files:
    df = pd.read_csv(file)

    # 👉 Ensure numeric values only
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 👉 Drop any non-numeric columns
    df = df.dropna(axis=1, how='all')

    # 👉 Add metadata to track file
    df['source_file'] = file.stem
    all_psd.append(df)

# Combine into a single DataFrame
psd_df = pd.concat(all_psd, ignore_index=True)

# ✅ Drop any remaining non-numeric columns
psd_df = psd_df.select_dtypes(include=['number'])

# 🔎 Total Power Across All Channels
total_power = psd_df.sum(axis=1)  # Sum across frequencies for each channel

# 🔎 Power in 20–50 Hz Range
freqs = psd_df.index.astype(float)

# 👉 Filter within the 20–50 Hz band
band_mask = (freqs >= 20) & (freqs <= 50)
band_power = psd_df.loc[band_mask].sum(axis=0)

# 🏆 Identify Outliers
high_muscle_artifact = band_power[band_power > 1e-10]  # High power in muscle range
low_power = total_power[total_power < 1e-12]           # Extremely low total power

# ✅ Flag bad channels
bad_channels = {}
for ch in high_muscle_artifact.index:
    bad_channels[ch] = "High muscle artifact"
for ch in low_power.index:
    if ch in bad_channels:
        bad_channels[ch] += " + Low total power"
    else:
        bad_channels[ch] = "Low total power"

# 👉 Print flagged channels
print("\n🚨 Flagged Bad Channels:")
for ch, reason in bad_channels.items():
    print(f"{ch}: {reason}")

# 👉 Provide Preprocessing Suggestions
if high_muscle_artifact.any():
    print("\n💡 SUGGESTED FIXES for High Muscle Artifact:")
    print("- Lower the ICA threshold for muscle detection.")
    print("- Try applying a low-pass filter at 30 Hz to reduce muscle activity.")
    print("- Increase the number of ICA components to remove more muscle artifacts.")

if low_power.any():
    print("\n💡 SUGGESTED FIXES for Low Total Power:")
    print("- Check electrode connectivity.")
    print("- Increase signal-to-noise ratio by adjusting reference electrodes.")
    print("- Verify that the EEG cap is fitted properly.")

# 👉 Plot Total Power for Each Channel
plt.figure(figsize=(12, 6))
colors = ["red" if ch in bad_channels else "skyblue" for ch in psd_df.columns]
psd_df.mean().plot(kind="bar", color=colors)
plt.title('Mean Total PSD Power for Each Channel')
plt.xlabel('Channel')
plt.ylabel('Power (µV²/Hz)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# 👉 Save Summary to CSV
summary_path = csv_output_dir / "PSD_Summary.csv"
summary_df = pd.DataFrame({
    'Mean Power': psd_df.mean(),
    'Std Power': psd_df.std(),
    'Min Power': psd_df.min(),
    'Max Power': psd_df.max(),
    'Flag': [bad_channels.get(ch, 'OK') for ch in psd_df.columns]
})
summary_df.to_csv(summary_path)
print(f"\n✅ Saved summary to: {summary_path}")

# 👉 Heatmap for Frequency Bands (with better scaling)
plt.figure(figsize=(14, 8))
sns.heatmap(psd_df, cmap="viridis", robust=True)
plt.title('Heatmap of PSD Across Channels and Frequencies')
plt.xlabel('Channel')
plt.ylabel('Frequency (Hz)')
plt.show()

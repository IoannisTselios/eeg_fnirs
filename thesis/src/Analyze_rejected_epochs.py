import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import Pipeline_preprocess
import mne
import numpy as np


excluded_dir = "skipped_files"
# 🔹 Path to your rejected epochs log file
log_file_path = "L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\results\\run_20250305_1241\\skipped_files\\rejected_epochs_log.txt"
preprocess = Pipeline_preprocess.EEGPreprocessor(excluded_dir)
# 🔹 Output CSV file
output_csv = "rejected_epochs_summary1.csv"

# 🔹 Store extracted data
rejected_data = []

# 🔹 Read and parse log file
with open(log_file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

# 🔹 Regex to extract file path and epoch rejections
file_pattern = re.compile(r"File: (.+)")
epoch_pattern = re.compile(r"Epoch (\d+): \((.+)\)")

current_file = None

for line in lines:
    line = line.strip()

    # ✅ If the line starts with "File:", extract the filename
    match_file = file_pattern.match(line)
    if match_file:
        current_file = match_file.group(1)
        continue

    # ✅ If it's an epoch rejection line, extract epoch number and channels
    match_epoch = epoch_pattern.match(line)
    if match_epoch and current_file:
        epoch_num = int(match_epoch.group(1))
        channels = match_epoch.group(2).replace("'", "").split(", ")

        # Extract patient ID and session from file path
        path_parts = current_file.split("\\")
        patient_id = path_parts[-4]
        session = path_parts[-3]

        # Store in structured format
        rejected_data.append({
            "Patient ID": patient_id,
            "Session": session,
            "Epoch": epoch_num,
            "Rejected Channels": channels,
            "File Path": current_file
        })

# 🔹 Convert to DataFrame
df = pd.DataFrame(rejected_data)

# 🔹 Flatten rejected channels for frequency analysis
all_rejected_channels = [channel for sublist in df["Rejected Channels"] for channel in sublist if channel]

# 🔹 Count occurrences of each channel
channel_counts = Counter(all_rejected_channels)

# 🔹 Convert to DataFrame
df_channels = pd.DataFrame(channel_counts.items(), columns=["Channel", "Rejections"])
df_channels = df_channels.sort_values(by="Rejections", ascending=False)

# 🔹 Save to CSV
df.to_csv(output_csv, index=False)

# 📊 Display Data Overview
summary_stats = {
    "Total Unique Patients": df["Patient ID"].nunique(),
    "Total Unique Sessions": df["Session"].nunique(),
    "Total Rejected Epochs": len(df),
    "Top Rejected Channel": df_channels.iloc[0]["Channel"] if not df_channels.empty else "N/A",
    "Top Rejected Channel Count": df_channels.iloc[0]["Rejections"] if not df_channels.empty else "N/A",
    "Average Channels Rejected per Epoch": df["Rejected Channels"].apply(len).mean(),
    "Median Channels Rejected per Epoch": df["Rejected Channels"].apply(len).median(),
    "Max Channels Rejected in One Epoch": df["Rejected Channels"].apply(len).max(),
}

# 📊 Top 10 Rejected Channels
print("\n📊 Top 10 Rejected Channels:")
print(df_channels.head(10))

# 🔹 Generate Summary Statistics
print("\n📈 Summary Statistics:")
for key, value in summary_stats.items():
    print(f"{key}: {value}")

# 📊 Plot histogram of rejected channels
plt.figure(figsize=(12, 6))
plt.bar(df_channels["Channel"][:10], df_channels["Rejections"][:10], color='blue', alpha=0.7)
plt.xlabel("Channel")
plt.ylabel("Number of Rejections")
plt.title("Top 10 Most Frequently Rejected EEG Channels")
plt.xticks(rotation=45)
plt.show()

# 📊 Plot Distribution of Rejected Channels Per Epoch
plt.figure(figsize=(10, 5))
sns.histplot(df["Rejected Channels"].apply(len), bins=20, kde=True, color="red", alpha=0.7)
plt.xlabel("Number of Channels Rejected in an Epoch")
plt.ylabel("Frequency")
plt.title("Distribution of Rejected Channels Per Epoch")
plt.show()

# 📊 Plot Rejections Per Patient
plt.figure(figsize=(12, 6))
df["Patient ID"].value_counts().plot(kind="bar", color="green", alpha=0.7)
plt.xlabel("Patient ID")
plt.ylabel("Number of Rejected Epochs")
plt.title("Rejected Epochs Per Patient")
plt.xticks(rotation=45)
plt.show()

# 📊 Heatmap of Top 15 Channels by Session
top_channels = df_channels.head(15)["Channel"].tolist()
df_filtered = df.explode("Rejected Channels")
df_filtered = df_filtered[df_filtered["Rejected Channels"].isin(top_channels)]

plt.figure(figsize=(12, 6))
heatmap_data = df_filtered.pivot_table(index="Rejected Channels", columns="Session", aggfunc="size", fill_value=0)
sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", fmt="d")
plt.title("Top 15 Rejected Channels Across Sessions")
plt.xlabel("Session")
plt.ylabel("Channel")
plt.show()

# 🔹 Load the rejected epochs summary CSV
df = pd.read_csv("rejected_epochs_summary1.csv")

# 🔹 Flatten and count rejected channels
all_rejected_channels = [channel for sublist in df["Rejected Channels"] for channel in eval(sublist) if channel]
channel_counts = Counter(all_rejected_channels)

# 🔹 Convert to DataFrame
df_channels = pd.DataFrame(channel_counts.items(), columns=["Channel", "Rejections"])
df_channels = df_channels.sort_values(by="Rejections", ascending=False)

# 🔹 Select the top 10 rejected channels
top_channels = df_channels.head(10)

# 🔹 Load standard EEG positions (10-20 system)
montage = mne.channels.make_standard_montage("standard_1020")

# 🔹 Correct channel mapping (Custom EEG → Standard 10-20)
channel_mapping = {
    "AFp1": "Fp1", "AFF1h": "Fz", "AF7": "F3", "AFF5h": "F7", "FT7": "F9",
    "FC5": "FC5", "FC3": "FC1", "FCC3h": "C3", "FFC1h": "T7", "FCC1h": "CP5",
    "CCP3h": "CP1", "CCP1h": "Pz", "CP1": "P3", "CP3": "P7", "CPP3h": "P9",
    "P1": "O1", "AFp2": "Oz", "AFF2h": "O2", "AF8": "P10", "AFF6h": "P8",
    "FT8": "P4", "FC6": "CP2", "FC4": "CP6", "FCC4h": "T8", "FFC2h": "C4",
    "FCC2h": "Cz", "CCP4h": "FC2", "CCP2h": "FC6", "CP2": "F10", "CP4": "F8",
    "CPP4h": "F4", "P2": "Fp2"
}

# 🔹 Map rejected channel names back to standard 10-20 names
mapped_channels = [channel_mapping.get(ch, ch) for ch in top_channels["Channel"]]

# 🔹 Get valid positions for the mapped channels (Avoid KeyError)
ch_pos = {ch: montage.get_positions()["ch_pos"].get(ch) for ch in mapped_channels if ch in montage.ch_names}
valid_ch_names = list(ch_pos.keys())

# 🔹 Extract rejection values for valid channels
valid_ch_values = np.array([channel_counts.get(ch, 0) for ch in valid_ch_names])

# 🔹 Create MNE info object
info = mne.create_info(valid_ch_names, sfreq=1000, ch_types="eeg")

# 🔹 Create subset montage with only valid channels
montage_subset = mne.channels.make_dig_montage({ch: ch_pos[ch] for ch in valid_ch_names})

# 🔹 Ensure correct shape for EvokedArray
evoked_data = valid_ch_values.reshape(-1, 1)  

# 🔹 Create Evoked object
evoked = mne.EvokedArray(evoked_data, info)
evoked.set_montage(montage_subset)

# 🔹 Plot rejected channels on a head map
fig, ax = plt.subplots(figsize=(6, 5))
im, _ = mne.viz.plot_topomap(evoked.data[:, 0], evoked.info, cmap="Reds", names=valid_ch_names, axes=ax)
plt.colorbar(im, ax=ax, label="Number of Rejections")
plt.title("Top 10 Most Rejected EEG Channels on Head Map")
plt.show()


# 🔹 Path to your rejected files log
rejected_files_log = "L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\results\\run_20250225_1549\\skipped_files\\skipped_files_log.txt"

# 🔹 Load the rejected files
with open(rejected_files_log, "r") as file:
    rejected_files = [line.strip().split(",")[0] for line in file.readlines()]

# 🔹 Threshold for rejection (500 µV = 500e-6 V)
threshold = 500e-6  

# 🔹 Iterate through each rejected file
for file_path in rejected_files:
    print(f"\n🔍 Checking file: {file_path}")

    try:
        # Load raw EEG data
        raw = preprocess.get_raw_from_xdf(file_path).load_data()

        data, times = raw.get_data(return_times=True)  # Get EEG data
        
        # Compute the max absolute amplitude per channel
        mean_amplitudes = np.mean(np.abs(data), axis=1)

        # Check if any channel exceeds the threshold
        exceeds_threshold = mean_amplitudes > threshold

        # Print results
        for ch_idx, max_amp in enumerate(mean_amplitudes):
            ch_name = raw.ch_names[ch_idx]
            print(f"  🧠 {ch_name}: {max_amp*1e6:.2f} µV {'⚠️ ABOVE THRESHOLD' if exceeds_threshold[ch_idx] else ''}")
        
        raw.plot(block=True)

        # Summary: If any channel exceeds threshold
        if np.any(exceeds_threshold):
            print("🚨 High amplitude values detected! This file likely got rejected due to artifact rejection.\n")
        else:
            print("✅ No extreme values detected. Investigate why this file was rejected.\n")

    except Exception as e:
        print(f"❌ Failed to process {file_path}: {e}")

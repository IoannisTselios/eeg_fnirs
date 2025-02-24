import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# 🔹 Path to your rejected epochs log file
log_file_path = "rejected_epochs_log.txt"

# 🔹 Output CSV file
output_csv = "rejected_epochs_summary.csv"

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
    "Total Unique Rejected Channels": len(df_channels),
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
plt.xticks(rotation=90)
plt.show()

# 📊 Heatmap of Top 15 Channels by Session
top_channels = df_channels.head(15)["Channel"].tolist()
df_filtered = df.explode("Rejected Channels")
df_filtered = df_filtered[df_filtered["Rejected Channels"].isin(top_channels)]

plt.figure(figsize=(12, 6))
heatmap_data = df_filtered.pivot_table(index="Rejected Channels", columns="Session", aggfunc="size", fill_value=0)
sns.heatmap(heatmap_data, annot=True, cmap="Blues", fmt="d")
plt.title("Top 15 Rejected Channels Across Sessions")
plt.xlabel("Session")
plt.ylabel("Channel")
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import mne
from collections import Counter

# 🔹 Load the rejected epochs summary CSV
df = pd.read_csv("rejected_epochs_summary.csv")
matplotlib.use("Qt5Agg")  # Switch to a different interactive backend

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

# 🔹 Define mapping from custom names to standard 10-20 names
channel_mapping = {
    "AFp1": "Fp1", "AFF1h": "Fz", "AF7": "F3", "AFF5h": "F7", "FT7": "F9", "FC5": "FC5", "FC3": "FC1", "FCC3h": "C3",
    "FFC1h": "T7", "FCC1h": "CP5", "CCP3h": "CP1", "CCP1h": "Pz", "CP1": "P3", "CP3": "P7", "CPP3h": "P9", "P1": "O1",
    "AFp2": "Oz", "AFF2h": "O2", "AF8": "P10", "AFF6h": "P8", "FT8": "P4", "FC6": "CP2", "FC4": "CP6", "FCC4h": "T8",
    "FFC2h": "C4", "FCC2h": "Cz", "CCP4h": "FC2", "CCP2h": "FC6", "CP2": "F10", "CP4": "F8", "CPP4h": "F4", "P2": "Fp2"
}

# 🔹 Map rejected channel names back to standard 10-20 names
mapped_channels = [channel_mapping.get(ch, ch) for ch in top_channels["Channel"]]

# 🔹 Get valid positions for the mapped channels
ch_pos = {ch: montage.get_positions()["ch_pos"][ch] for ch in mapped_channels if ch in montage.ch_names}

# 🔹 Extract names and rejection values (only valid channels)
valid_ch_names = list(ch_pos.keys())
valid_ch_values = np.array([top_channels[top_channels["Channel"] == ch]["Rejections"].values[0] if ch in top_channels["Channel"].values else 0 for ch in valid_ch_names])

# 🔹 Create an info object with correct number of channels
info = mne.create_info(valid_ch_names, sfreq=1000, ch_types="eeg")

# 🔹 Create a subset montage with only the valid channels
montage_subset = mne.channels.make_dig_montage({ch: ch_pos[ch] for ch in valid_ch_names})

# 🔹 Reshape data to match the number of channels
evoked_data = valid_ch_values.reshape(-1, 1)  # Convert to (channels, 1)

# 🔹 Create an Evoked object
evoked = mne.EvokedArray(evoked_data, info)
evoked.set_montage(montage_subset)

# 🔹 Plot the rejected channels on a head map
fig, ax = plt.subplots(figsize=(6, 5))
im, _ = mne.viz.plot_topomap(evoked.data[:, 0], evoked.info, cmap="Reds", names=valid_ch_names, axes=ax)
plt.colorbar(im, ax=ax, label="Number of Rejections")
plt.title("Top 10 Most Rejected EEG Channels on Head Map")
plt.show()




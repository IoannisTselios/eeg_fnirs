from utils.file_mgt import get_random_eeg_file_paths

import os
import numpy as np
import pyxdf
import pandas as pd

# Function to process each XDF file
def process_xdf_file(file_path):
    try:
        # Load the XDF file
        streams, header = pyxdf.load_xdf(file_path)

        # Identify EEG stream index (assumed to be Stream 1)
        eeg_stream_index = 1  

        # Extract channel names
        if "desc" in streams[eeg_stream_index]["info"]:
            try:
                channel_info = streams[eeg_stream_index]["info"]["desc"][0]["channels"][0]["channel"]
                channel_list = [ch["label"][0] for ch in channel_info]
            except (KeyError, IndexError):
                channel_list = ["Channel info not available"]
        else:
            channel_list = ["No channel description found"]

        # Extract EEG Data
        eeg_data = np.array(streams[eeg_stream_index]["time_series"]).T  # Convert to NumPy
        eeg_timestamps = np.array(streams[eeg_stream_index]["time_stamps"])  # EEG time axis

        # Find Accelerometer Channels
        acc_channels = ["ACC_X", "ACC_Y", "ACC_Z"]
        acc_indices = [i for i, ch in enumerate(channel_list) if ch in acc_channels]

        if len(acc_indices) == 3:
            acc_data = eeg_data[acc_indices, :]
            
            # Compute Movement Energy (Magnitude of Acceleration)
            acc_magnitude = np.sqrt(acc_data[0]**2 + acc_data[1]**2 + acc_data[2]**2)
            
            # Find Time Periods with High Movement
            movement_threshold = np.percentile(acc_magnitude, 95)  # 95th percentile as threshold
            high_movement_indices = np.where(acc_magnitude > movement_threshold)[0]
            
            # Find EEG Channels with High Amplitude
            eeg_indices = [i for i in range(len(channel_list)) if channel_list[i] not in acc_channels]
            eeg_amplitudes = np.max(np.abs(eeg_data[eeg_indices, :]), axis=0)  # Max EEG amplitude across channels
            
            eeg_threshold = np.percentile(eeg_amplitudes, 95)  # 95th percentile for EEG artifacts
            high_eeg_indices = np.where(eeg_amplitudes > eeg_threshold)[0]
            
            # Find Overlapping Time Points (Movement & EEG Artifacts)
            overlapping_indices = np.intersect1d(high_movement_indices, high_eeg_indices)

            # Return the count of detected movement-related artifacts
            return len(overlapping_indices)

        else:
            return "No accelerometer data"

    except Exception as e:
        return f"Error: {str(e)}"

paths = get_random_eeg_file_paths("xdf", 5000)
# Process all files and store results
results = []
for file_path in paths:
    artifact_count = process_xdf_file(file_path)
    results.append({"File Path": file_path, "Movement Artifacts": artifact_count})

# Save results to CSV
df_results = pd.DataFrame(results)
csv_path = "movement_artifacts_summary.csv"
df_results.to_csv(csv_path, index=False)


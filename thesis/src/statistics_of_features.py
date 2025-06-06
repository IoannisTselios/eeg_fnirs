import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# === Load CSV ===
csv_path = "L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\results\\run_20250527_1826\\feature_extraction_files\\eeg_features.csv"
df = pd.read_csv(csv_path)

# === Output directory ===
output_dir = Path("features_plot")
output_dir.mkdir(parents=True, exist_ok=True)

# === Channels to consider ===
channels = ["AFF1h", "AF7", "AFF5h", "AFF6h", "AF8", "FC5", "FC3", "FCC3h", "FCC4h", "FFC2h", "FCC2h", "CCP3h", "CCP4h"]

# === Melt delta and theta separately ===
delta_cols = [f"delta_{ch}" for ch in channels]
theta_cols = [f"theta_{ch}" for ch in channels]

df_delta_melted = df.melt(id_vars=["id", "drug", "time", "epoch_idx"],
                          value_vars=delta_cols,
                          var_name="channel",
                          value_name="delta_power")
df_delta_melted["channel"] = df_delta_melted["channel"].str.replace("delta_", "")

df_theta_melted = df.melt(id_vars=["id", "drug", "time", "epoch_idx"],
                          value_vars=theta_cols,
                          var_name="channel",
                          value_name="theta_power")
df_theta_melted["channel"] = df_theta_melted["channel"].str.replace("theta_", "")

# === Merge for joint plotting ===
df_melted = pd.merge(df_delta_melted, df_theta_melted,
                      on=["id", "drug", "time", "epoch_idx", "channel"])

# === Compute log10 of power (add small constant to avoid log(0)) ===
df_melted["log_delta_power"] = np.log10(df_melted["delta_power"] + 1e-9)
df_melted["log_theta_power"] = np.log10(df_melted["theta_power"] + 1e-9)

# === Plot and save histograms ===
for ch in channels:
    plt.figure(figsize=(12, 5))
    sns.histplot(data=df_melted[df_melted["channel"] == ch],
                 x="log_delta_power", bins=30, kde=True, color="skyblue")
    plt.title(f"Log Delta Power Histogram (Channel: {ch}) - All Values")
    plt.xlabel("log10(Delta Power + 1e-9) (µV²)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / f"log_delta_power_{ch}.png")
    plt.close()

    plt.figure(figsize=(12, 5))
    sns.histplot(data=df_melted[df_melted["channel"] == ch],
                 x="log_theta_power", bins=30, kde=True, color="salmon")
    plt.title(f"Log Theta Power Histogram (Channel: {ch}) - All Values")
    plt.xlabel("log10(Theta Power + 1e-9) (µV²)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / f"log_theta_power_{ch}.png")
    plt.close()

print("\n✅ Histograms saved in the 'features_plot' folder!")

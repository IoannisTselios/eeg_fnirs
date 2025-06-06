import mne
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# === Path to your epochs FIF file ===
fif_path = "L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\results_moshgan\\run_20250604_2104\\Epochs_files\\02IT-EDF+_mapped_clean-epo.fif"  # <- change this
# L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\results_moshgan\\run_20250604_2104\\Epochs_files\\02IT-EDF+_mapped_clean-epo.fif
# === Load the epochs ===
print(f"\n📂 Loading epochs from: {fif_path}")
epochs = mne.read_epochs(fif_path, preload=True)

# === Basic Info ===
print("\n✅ Epochs loaded!")
print(f"📏 Number of epochs: {len(epochs)}")
print(f"🧠 Channels: {len(epochs.ch_names)} | Duration: {epochs.tmax - epochs.tmin + 1/epochs.info['sfreq']:.3f} sec per epoch")
print(f"🔢 Sampling rate: {epochs.info['sfreq']} Hz")

# === Check Event IDs ===
print("\n🔍 Event ID mapping (labels):")
print(epochs.event_id)

# === Count label frequency ===
label_id_map = {v: k for k, v in epochs.event_id.items()}
labels = [label_id_map[e[-1]] for e in epochs.events]
label_counts = Counter(labels)

print("\n📊 Label distribution:")
for label, count in label_counts.items():
    print(f"  {label}: {count} epochs")

# === Optional: plot power spectral density ===
print("📈 Plotting PSD...")
psd = epochs.compute_psd()
psd.plot(show=False)  # Don't auto-show yet
plt.show(block=True)  # This keeps the window open until you close it

# === Optional: save label list to CSV ===
import pandas as pd
df = pd.DataFrame({"label": labels})
df.to_csv("epoch_labels.csv", index=False)
print("\n💾 Saved label list to epoch_labels.csv")

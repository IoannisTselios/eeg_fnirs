import mne
from pathlib import Path
import matplotlib.pyplot as plt

# === Config ===
# Replace this with the path to your .fif file
fif_file = Path("L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\thesis\\src_moshgan\\mapped_5mv_5fv_3resting\\23IA-EDF+_mapped.fif")

# === Load Raw Data ===
print(f"🔄 Loading file: {fif_file.name}")
raw = mne.io.read_raw_fif(fif_file, preload=True)
print("✅ File loaded.")

# === Print Basic Info ===
print(f"📦 Data info:\n{raw.info}")
print(f"🧠 Available channels: {raw.ch_names[:10]} ...")

# === Plot Raw Signal ===
raw.plot(block=True)

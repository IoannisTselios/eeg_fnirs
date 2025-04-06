import os
import glob
import mne
import matplotlib.pyplot as plt

# 📁 Path to the folder containing the FIF files
folder_path = "L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\EDF filer"  # Update this path if needed

# ✅ Find all .fif files in the folder
fif_files = glob.glob(os.path.join(folder_path, "*.fif"))

if not fif_files:
    print("❌ No FIF files found in the folder.")
else:
    print(f"✅ Found {len(fif_files)} FIF files in {folder_path}\n")

# 🔄 Loop through each file
for file in fif_files:
    try:
        print(f"\n🔎 Reading file: {file}")

        # ✅ Load the FIF file
        raw = mne.io.read_raw_fif(file, preload=True)

        # ✅ Check if annotations exist
        if len(raw.annotations) == 0:
            print("⚠️ No annotations found in this file.")
        else:
            print(f"✅ {len(raw.annotations)} annotations found in {file}")
            print(f"\n📝 Annotations:\n{raw.annotations}")

        # ✅ Plot the raw data with annotations
        print("📈 Plotting raw data with annotations...")
        raw.plot(
            duration=10,               # Display 10-second windows
            n_channels=20,             # Plot up to 20 channels at once
            scalings="auto",           # Auto-scale the plot
            show=True,                 # Show the plot
            block=True,                # Block execution until plot is closed
            title=f"Raw Data - {os.path.basename(file)}"
        )

    except Exception as e:
        print(f"\n❌ Failed to process {file}: {e}\n")

print("\n🚀 All files processed!")

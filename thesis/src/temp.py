from pathlib import Path

# Define the base directory where .fif files are located
base_directory = Path(r"L:\\LovbeskyttetMapper\\CONNECT-ME\\CONMED3\Dataoptagelser\\NIRS-EEG")

# Find all .fif files in the directory and subdirectories
fif_files = list(base_directory.rglob("*.fif"))

# Delete each .fif file
for fif_file in fif_files:
    try:
        fif_file.unlink()  # Remove the file
        print(f"🗑️ Deleted: {fif_file}")
    except Exception as e:
        print(f"❌ Failed to delete {fif_file}: {e}")

print(f"✅ Removed {len(fif_files)} .fif files from {base_directory}")

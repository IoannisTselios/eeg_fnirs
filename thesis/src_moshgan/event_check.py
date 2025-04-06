import os
import glob
import mne
import shutil
from collections import defaultdict

# 📁 Paths
folder_path = "L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\EDF filer"
target_path = "L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\thesis\\src_moshgan\\mapped_5mv_5fv_3resting"

# ✅ Create target folder if it doesn't exist
os.makedirs(target_path, exist_ok=True)

# ✅ Expected counts
EXPECTED_RESTING = 3
EXPECTED_MEDICAL = 5
EXPECTED_FAMILIAR = 5

# ✅ Find all processed files ending with "_mapped.fif"
edf_files = glob.glob(os.path.join(folder_path, "*_mapped.fif"))

if not edf_files:
    print("❌ No mapped files found in the folder.")
else:
    print(f"✅ Found {len(edf_files)} mapped files in {folder_path}\n")

# ✅ Dictionary to store final subject counts
subject_counts = defaultdict(lambda: {"resting": 0, "medical": 0, "familiar": 0, "other": 0})

# ✅ Group files by subject and count annotations
for file in edf_files:
    try:
        print(f"\n🔎 Reading file: {file}")

        # Extract subject ID from filename (first two characters)
        subject_id = os.path.basename(file)[:2]

        # ✅ Load the mapped file
        raw = mne.io.read_raw_fif(file, preload=True)
        print(f"➡️ Channels: {raw.ch_names}")
        
        if len(raw.annotations) == 0:
            print(f"⚠️ No annotations found in {file}")
            continue

        # ✅ Reset counts for each file (so they don't carry over)
        counts = {"resting": 0, "medical": 0, "familiar": 0, "other": 0}

        # ✅ Debug: Print raw annotations
        for ann in raw.annotations.description:
            print(f"➡️ Raw annotation: '{ann}'")

        # ✅ Count the annotations for the current file only
        for ann in raw.annotations.description:
            ann = ann.strip().lower()  # ✅ Strip and normalize case
            if ann == "resting":
                counts["resting"] += 1
            elif ann == "medical voice":
                counts["medical"] += 1
            elif ann == "familiar voice":
                counts["familiar"] += 1
            else:
                counts["other"] += 1

        # ✅ Save if the file meets the criteria
        if (counts["resting"] >= EXPECTED_RESTING and 
            counts["medical"] >= EXPECTED_MEDICAL and 
            counts["familiar"] >= EXPECTED_FAMILIAR):
            
            # ✅ Save to target folder
            target_file = os.path.join(target_path, os.path.basename(file))
            shutil.copy(file, target_file)
            print(f"✅ File passed conditions and saved to: {target_file}")

            # ✅ Update global subject counts after acceptance
            subject_counts[subject_id]["resting"] += counts["resting"]
            subject_counts[subject_id]["medical"] += counts["medical"]
            subject_counts[subject_id]["familiar"] += counts["familiar"]
            subject_counts[subject_id]["other"] += counts["other"]

            # ✅ Print annotation summary
            print(f"📊 Annotation Summary for Accepted File (Subject {subject_id}):")
            print(f"   - Resting: {counts['resting']}")
            print(f"   - Medical voice: {counts['medical']}")
            print(f"   - Familiar voice: {counts['familiar']}")
            print(f"   - Other annotations: {counts['other']}")

            # ✅ Plot the raw EEG file for accepted files
            print("🔎 Plotting accepted raw data... (Close plot to continue)")
            raw.plot(block=True)

        else:
            # ✅ Print reason for rejection
            print(f"❌ File REJECTED (Subject {subject_id}):")
            print(f"   - Resting: {counts['resting']} (expected {EXPECTED_RESTING})")
            print(f"   - Medical voice: {counts['medical']} (expected {EXPECTED_MEDICAL})")
            print(f"   - Familiar voice: {counts['familiar']} (expected {EXPECTED_FAMILIAR})")
            print(f"   - Other annotations: {counts['other']}")

            # ✅ Plot the raw EEG file for rejected files for manual review
            print("🔎 Plotting rejected raw data... (Close plot to continue)")
            raw.plot(block=True)

    except Exception as e:
        print(f"❌ Failed to process {file}: {e}")

# ✅ Analyze the results
matching_subjects = 0
non_matching_subjects = 0

for subject, counts in subject_counts.items():
    if (counts["resting"] >= EXPECTED_RESTING and 
        counts["medical"] >= EXPECTED_MEDICAL and 
        counts["familiar"] >= EXPECTED_FAMILIAR):
        matching_subjects += 1
    else:
        non_matching_subjects += 1

# ✅ Summary Report
print("\n📊 SUMMARY REPORT:")
print(f"➡️ Total subjects processed: {len(subject_counts)}")
print(f"✅ Subjects with correct annotation counts: {matching_subjects}")
print(f"❌ Subjects with incorrect annotation counts: {non_matching_subjects}\n")

# ✅ Optional: Print detailed information for incorrect subjects
if non_matching_subjects > 0:
    print("❌ SUBJECTS WITH INCORRECT COUNTS:")
    for subject, counts in subject_counts.items():
        if not (counts["resting"] >= EXPECTED_RESTING and 
                counts["medical"] >= EXPECTED_MEDICAL and 
                counts["familiar"] >= EXPECTED_FAMILIAR):
            print(f"➡️ Subject: {subject} | Resting: {counts['resting']}, Medical: {counts['medical']}, Familiar: {counts['familiar']}, Other: {counts['other']}")

print("\n🚀 All files processed!")

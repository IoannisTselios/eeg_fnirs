import pandas as pd
from pathlib import Path

# === Load your DataFrame ===
df = pd.read_csv("L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\thesis\\src\\final_description.csv")  # Replace with your actual file path if needed

# === Function to extract the short identifier from the full path ===
def extract_id(file_path):
    name = Path(file_path).stem  # Removes the extension
    return name.replace("_preprocessed_raw", "")

# === Apply the identifier extraction ===
df['identifier'] = df['file'].apply(extract_id)

# === Create the two lists ===
list_only_2 = df[df['condition'] == 2]['identifier'].tolist()
list_1_and_2 = df[df['condition'].isin([1, 2])]['identifier'].tolist()

# === Save lists to text files ===
with open("files_with_2.txt", "w") as f:
    for item in list_only_2:
        f.write(f"{item}\n")

with open("files_with_1_and_2.txt", "w") as f:
    for item in list_1_and_2:
        f.write(f"{item}\n")

print("✅ Lists saved to 'files_with_2.txt' and 'files_with_1_and_2.txt'")

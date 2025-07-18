import pandas as pd
from pathlib import Path

# 📁 Set your base directory containing all run folders
base_dir = Path("L:/LovbeskyttetMapper/CONNECT-ME/Ioannis/thesis_code/results_moshgan")

# 🔍 Recursively find all snr_log.csv files
snr_logs = list(base_dir.rglob("snr_log.csv"))

best_run = None
best_avg_snr = float('-inf')

# 📊 Track all runs for reference
all_runs = []

for snr_log in snr_logs:
    try:
        df = pd.read_csv(snr_log, header=None, names=["filename", "snr"])
        avg_snr = df["snr"].mean()
        all_runs.append((snr_log.parent, avg_snr))
        if avg_snr > best_avg_snr:
            best_avg_snr = avg_snr
            best_run = snr_log.parent
    except Exception as e:
        print(f"❌ Failed to read {snr_log}: {e}")

# 🏆 Output the best run
if best_run:
    print(f"\n🏅 Best run folder based on SNR: {best_run}")
    print(f"📈 Average SNR: {best_avg_snr:.2f} dB")
else:
    print("⚠️ No valid snr_log.csv files found.")

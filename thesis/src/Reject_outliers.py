from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import euclidean
from scipy.stats import zscore
from sklearn.decomposition import PCA
import mne

class EEGEpochQualityChecker:
    def __init__(self, root_path, file_path, summary_csv, threshold_config=None, min_failures=2, verbose=True):
        self.root_path = Path(root_path)
        self.file_path = Path(file_path)
        self.summary_csv = Path(summary_csv)
        self.min_failures = min_failures
        self.verbose = verbose
        self.threshold_config = threshold_config or {
            "distance_z": 3.0,
            "high_beta_power": 5e-10,
            "snr": 0.5,
            "max_gfp": 5e-5
        }

    def compute_distance_from_mean(self, epochs, indices):
        mean_epoch = epochs.get_data().mean(axis=0)
        data = epochs[indices].get_data()
        distances = [euclidean(epoch.flatten(), mean_epoch.flatten()) for epoch in data]
        return np.array(distances), zscore(distances)

    def compute_band_power(self, epochs, indices, fmin, fmax):
        psd = epochs[indices].compute_psd(fmin=1, fmax=50)
        psds = psd.get_data()
        freqs = psd.freqs
        band_mask = (freqs >= fmin) & (freqs <= fmax)
        return psds[:, :, band_mask].mean(axis=-1).mean(axis=1)

    def compute_theta_power(self, epochs, indices):
        return self.compute_band_power(epochs, indices, 4, 8)

    def compute_low_high_freq_ratio(self, epochs, indices):
        psd = epochs[indices].compute_psd(fmin=1, fmax=40)
        psds = psd.get_data()
        freqs = psd.freqs
        low = psds[:, :, (freqs >= 1) & (freqs <= 12)].mean(axis=-1).mean(axis=1)
        high = psds[:, :, freqs >= 25].mean(axis=-1).mean(axis=1)
        return low / (high + 1e-8)

    def compute_gfp_values(self, epochs, indices):
        data = epochs[indices].get_data()
        gfp = [epoch.std(axis=0) for epoch in data]
        return np.array([np.max(g) for g in gfp]), np.array([np.mean(g) for g in gfp])

    def compute_pca_embedding(self, epochs, indices):
        data = epochs.get_data().reshape(len(epochs), -1)
        pca_all = PCA(n_components=2).fit_transform(data)
        return pca_all, pca_all[indices]

    def check_quality(self, epochs, indices):
        report = {
            'distance': None,
            'distance_z': None,
            'snr': None,
            'max_gfp': None,
            'mean_gfp': None,
            'pca_all': None,
            'pca_outliers': None,
            'theta_power': None,
            'delta_power': None,
        }

        report['distance'], report['distance_z'] = self.compute_distance_from_mean(epochs, indices)
        report['theta_power'] = self.compute_theta_power(epochs, indices)
        report['delta_power'] = self.compute_band_power(epochs, indices, 1, 4)
        report['snr'] = self.compute_low_high_freq_ratio(epochs, indices)
        report['max_gfp'], report['mean_gfp'] = self.compute_gfp_values(epochs, indices)
        report['pca_all'], report['pca_outliers'] = self.compute_pca_embedding(epochs, indices)

        if self.verbose:
            print("\n📊 Thresholds Applied:")
            for k, v in self.threshold_config.items():
                op = '<' if k == 'snr' else '>' if isinstance(v, float) else 'range'
                print(f"  - {k} {op} {v}")

        bad_epochs = []
        for i, idx in enumerate(indices):
            fails = []
            if abs(report['distance_z'][i]) > self.threshold_config['distance_z']:
                fails.append(f"distance_z={report['distance_z'][i]:.2f}")
            if report['snr'][i] < self.threshold_config['snr']:
                fails.append(f"snr={report['snr'][i]:.2f}")
            if report['max_gfp'][i] > self.threshold_config['max_gfp']:
                fails.append(f"max_gfp={report['max_gfp'][i]:.2e}")

            # Optional thresholds for delta/theta (if defined as (min, max))
            theta_thresh = self.threshold_config.get("theta_power", None)
            if theta_thresh:
                min_theta, max_theta = theta_thresh
                if report['theta_power'][i] < min_theta:
                    fails.append(f"theta_low={report['theta_power'][i]:.2e}")
                elif report['theta_power'][i] > max_theta:
                    fails.append(f"theta_high={report['theta_power'][i]:.2e}")

            delta_thresh = self.threshold_config.get("delta_power", None)
            if delta_thresh:
                min_delta, max_delta = delta_thresh
                if report['delta_power'][i] < min_delta:
                    fails.append(f"delta_low={report['delta_power'][i]:.2e}")
                elif report['delta_power'][i] > max_delta:
                    fails.append(f"delta_high={report['delta_power'][i]:.2e}")

            if len(fails) >= self.min_failures:
                bad_epochs.append(idx)
                if self.verbose:
                    print(f"❌ Epoch {idx} flagged (failed {len(fails)}): {', '.join(fails)}")
            elif self.verbose:
                print(f"✅ Epoch {idx} passed (only {len(fails)} fail{'s' if len(fails)==1 else 's'})")

        if self.verbose:
            print(f"\n🚨 Total bad epochs (failed ≥{self.min_failures} checks): {len(bad_epochs)} → {bad_epochs}")

        # Plot PCA scatter
        pca_all = report["pca_all"]
        pca_bad = report["pca_outliers"]
        all_indices = np.arange(len(pca_all))
        bad_indices = [i for i in all_indices if i in bad_epochs]
        good_indices = [i for i in all_indices if i not in bad_epochs]

        plt.figure(figsize=(8, 6))
        if good_indices:
            plt.scatter(pca_all[good_indices, 0], pca_all[good_indices, 1], label="Clean Epochs", alpha=0.5, color="blue")
        if bad_indices:
            plt.scatter(pca_all[bad_indices, 0], pca_all[bad_indices, 1], color='red', label="Bad Epochs", alpha=0.8)
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.title("PCA of EEG Epochs: Clean vs. Flagged")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Optional: Plot a few example bad epochs (if any)
        if bad_epochs:
            print(f"\n🧠 Plotting first 5 bad epochs...")
            for i, epoch_idx in enumerate(bad_epochs[:5]):
                fig = epochs[epoch_idx].plot(scalings='auto', n_channels=30, title=f"Bad Epoch {epoch_idx}", show=True, block=True)


        return report, bad_epochs


    def resolve_outlier_file_paths(self, df, extension=".fif"):
        time_mapping = {
            "0": ["Baseline"],
            "1": ["Post administration 1", "Post-administration 1"],
            "2": ["Post administration 2", "Post-administration 2"]
        }

        paths = []
        for idx, row in df.iterrows():
            patient_id = str(row["id"]).strip()
            drug = str(row["drug"]).strip()
            time_key = str(row["time"]).strip()
            matched_file = None

            for time_variant in time_mapping.get(time_key, []):
                folder = self.root_path / patient_id / drug / time_variant
                if folder.exists():
                    fif_files = list(folder.glob(f"*.{extension.lstrip('.')}"))
                    if fif_files:
                        matched_file = fif_files[0]
                        break

            paths.append(str(matched_file) if matched_file else None)
        df["resolved_file_path"] = paths
        return df

    def process(self):
        df = pd.read_csv(self.file_path)
        df = self.resolve_outlier_file_paths(df)

        summary_records = []

        for i, row in df.iterrows():
            file_path = row['resolved_file_path']
            if not file_path or not Path(file_path).exists():
                print(f"❌ File not found for row {i}: {file_path}")
                continue

            print(f"\n📂 Processing file: {file_path}")
            try:
                epochs = mne.read_epochs(file_path, preload=True)
                all_indices = list(range(len(epochs)))
                n_before = len(epochs)

                report, bad_epochs = self.check_quality(epochs, all_indices)

                print("\n📊 Quality Check Report")
                print("Distance z-scores:", report['distance_z'][:5])
                print("High Beta Power:", report['high_beta_power'][:5])
                print("SNR:", report['snr'][:5])
                print("Max GFP:", report['max_gfp'][:5])
                print("Alpha Power:", report['alpha_power'][:5])
                print("Theta Power:", report['theta_power'][:5])
                print("Gamma Power:", report['gamma_power'][:5])

                n_removed = len(bad_epochs)
                n_after = n_before - n_removed

                summary_records.append({
                    "file_path": file_path,
                    "n_epochs_before": n_before,
                    "n_removed": n_removed,
                    "n_epochs_after": n_after,
                    "bad_epoch_indices": bad_epochs if bad_epochs else None
                })

                if bad_epochs:
                    print(f"🧹 Removing {n_removed} bad epochs from file.")
                    epochs.drop(bad_epochs)
                    epochs.save(file_path, overwrite=True)
                    print(f"✅ Saved cleaned epochs to: {file_path}")
                else:
                    print("✅ No bad epochs found. File remains unchanged.")

            except Exception as e:
                print(f"❌ Failed to process file: {e}")
                summary_records.append({
                    "file_path": file_path,
                    "n_epochs_before": "ERROR",
                    "n_removed": "ERROR",
                    "n_epochs_after": "ERROR",
                    "bad_epoch_indices": str(e)
                })

        summary_df = pd.DataFrame(summary_records)
        summary_df.to_csv(self.summary_csv, index=False)
        print(f"\n📝 Saved epoch removal summary to: {self.summary_csv}")

# qc = EEGEpochQualityChecker(
#     root_path=Path("L:\\LovbeskyttetMapper\\CONNECT-ME\\CONMED3\\Dataoptagelser\\NIRS-EEG\\"), 
#     file_path=Path(f"L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\results\\run_20250428_1902\\feature_extraction_files\\eeg_outliers_detail_df.csv"), 
#     summary_csv="", 
#     threshold_config={
#         "distance_z": 4.0,
#         "snr": 0.4,
#         "max_gfp": 1e-4,
#         "theta_power": (1e-10, 1e-8),
#         "delta_power": (2e-10, 2e-8)
#     },
#     min_failures=2,
#     verbose=True
# )

# qc.process()
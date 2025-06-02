import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.signal import savgol_filter

# === Control plot display ===
show_plots = False

# === Load data ===
csv_path = "L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\results\\run_20250527_1826\\feature_extraction_files\\eeg_features.csv"
df = pd.read_csv(csv_path)

# === List of delta/theta feature columns ===
feature_cols = [col for col in df.columns if col.startswith("delta_") or col.startswith("theta_")]

# === Log-transform the power features (avoid log(0)) ===
df_log = df.copy()
df_log[feature_cols] = np.log10(df_log[feature_cols] + 1e-9)

# === Create an empty list to collect all results ===
summary_results = []

# === Loop over each patient/session ===
for patient_id in df_log["id"].unique():
    df_patient = df_log[df_log["id"] == patient_id].copy()

    for drug in df_patient["drug"].unique():
        df_session = df_patient[df_patient["drug"] == drug].copy()

        if len(df_session) < 5:
            print(f"⚠️ Skipping {patient_id} / {drug} (too few samples: {len(df_session)})")
            continue

        # === Prepare features (log-transformed) ===
        X_log = df_session[feature_cols].fillna(0)

        # === Scale ===
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_log)

        # === PCA reduction: keep enough components to explain 95% of variance ===
        pca_full = PCA(n_components=0.95, svd_solver="full")
        X_pca_full = pca_full.fit_transform(X_scaled)
        num_components_95 = pca_full.n_components_
        print(f"ℹ️ PCA reduced to {num_components_95} components explaining 95% of variance for {patient_id} / {drug}")

        # === Cumulative variance plot ===
        cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
        X_pca_2d = X_pca_full[:, :2]

        # === Apply KMeans and Spectral Clustering ===
        clustering_results = {}
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)  # Updated!
        clustering_results["KMeans"] = kmeans.fit_predict(X_pca_full)

        spectral = SpectralClustering(n_clusters=2, affinity='rbf', gamma=0.01, n_neighbors=10, random_state=42)
        clustering_results["Spectral"] = spectral.fit_predict(X_pca_full)

        # === Save cluster counts and silhouette scores ===
        silhouette_scores = {}
        for model_name, labels in clustering_results.items():
            if len(set(labels)) > 1 and -1 not in set(labels):
                score = silhouette_score(X_pca_full, labels)
                silhouette_scores[model_name] = score
                print(f"{model_name}: {score:.3f}")
            else:
                silhouette_scores[model_name] = np.nan
                print(f"{model_name}: N/A")

        for model_name, labels in clustering_results.items():
            counts = pd.Series(labels).value_counts().to_dict()
            row = {
                "patient_id": patient_id,
                "session": drug,
                "model": model_name,
                "pca_components_95": num_components_95,
                "silhouette_score": silhouette_scores[model_name]
            }
            for cluster_label, count in counts.items():
                row[f"cluster_{cluster_label}"] = count
            summary_results.append(row)

        # === Visuals only if show_plots is True ===
        if show_plots:
            fig, axes = plt.subplots(3, 2, figsize=(12, 12))
            axes = axes.flatten()

            # Plot 1: histogram
            sns.histplot(X_log.values.flatten(), bins=50, kde=True, color="purple", ax=axes[0])
            axes[0].set_title("Log Power Distribution", fontsize=10)
            axes[0].set_xlabel("log10(Power + 1e-9)", fontsize=8)
            axes[0].set_ylabel("Count", fontsize=8)
            axes[0].tick_params(axis='both', labelsize=8)

            # Plot 2: cumulative variance
            axes[1].plot(np.arange(1, len(cumsum_variance) + 1), cumsum_variance, marker='o', linewidth=1)
            axes[1].axhline(0.95, color='red', linestyle='--', label="95% variance")
            axes[1].axvline(num_components_95, color='green', linestyle='--', label=f"{num_components_95} components")
            axes[1].set_title("PCA Cumulative Variance", fontsize=10)
            axes[1].set_xlabel("Number of Components", fontsize=8)
            axes[1].set_ylabel("Cumulative Variance Explained", fontsize=8)
            axes[1].legend(fontsize=8)
            axes[1].tick_params(axis='both', labelsize=8)

            # Plot 3: PC1 Time Series with trend
            df_session["sample_idx"] = np.arange(len(X_pca_full))
            df_session["pc1"] = X_pca_full[:, 0]
            df_session["cluster_kmeans"] = clustering_results["KMeans"]
            colors_kmeans = df_session["cluster_kmeans"].map(dict(zip(np.unique(df_session["cluster_kmeans"]), sns.color_palette("Set2", len(np.unique(df_session["cluster_kmeans"]))))))

            trend_pc1 = savgol_filter(df_session["pc1"], window_length=21, polyorder=2)

            axes[2].plot(df_session["sample_idx"], df_session["pc1"], color='gray', linewidth=0.8, alpha=0.5)
            axes[2].scatter(df_session["sample_idx"], df_session["pc1"], c=colors_kmeans, s=15, edgecolor='black')
            axes[2].plot(df_session["sample_idx"], trend_pc1, color='red', linestyle='-', linewidth=1.5, alpha=0.5, label='Trend')
            axes[2].set_title("PC1 (KMeans) + Trend", fontsize=10)
            axes[2].set_xlabel("Sample Index", fontsize=8)
            axes[2].set_ylabel("PC1 Value", fontsize=8)
            axes[2].grid(True)
            axes[2].tick_params(axis='both', labelsize=8)
            axes[2].legend(fontsize=6)

             # Plot 4: PC1 Time Series (Spectral Clusters)
            df_session["cluster_spectral"] = clustering_results["Spectral"]
            colors_spectral = df_session["cluster_spectral"].map(dict(zip(np.unique(df_session["cluster_spectral"]), sns.color_palette("Set2", len(np.unique(df_session["cluster_spectral"]))))))

            axes[3].plot(df_session["sample_idx"], df_session["pc1"], color='gray', linewidth=0.8, alpha=0.5)
            axes[3].scatter(df_session["sample_idx"], df_session["pc1"], c=colors_spectral, s=15, edgecolor='black')
            axes[3].plot(df_session["sample_idx"], trend_pc1, color='red', linestyle='-', linewidth=1.5, alpha=0.5, label='Trend')

            axes[3].set_title("PC1 (Spectral) + Trend", fontsize=10)
            axes[3].set_xlabel("Sample Index", fontsize=8)
            axes[3].set_ylabel("PC1 Value", fontsize=8)
            axes[3].grid(True)
            axes[3].tick_params(axis='both', labelsize=8)
            axes[3].legend(fontsize=6)

            # Plot 5: KMeans 2D Clusters
            df_session["pca1"] = X_pca_2d[:, 0]
            df_session["pca2"] = X_pca_2d[:, 1]
            sns.scatterplot(data=df_session, x="pca1", y="pca2", hue="cluster_kmeans", palette="Set2", ax=axes[4], legend=False, s=30)
            axes[4].set_title("KMeans Clusters", fontsize=10)
            axes[4].set_xlabel("PCA 1", fontsize=8)
            axes[4].set_ylabel("PCA 2", fontsize=8)
            axes[4].grid(True)
            axes[4].tick_params(axis='both', labelsize=8)

            # Plot 6: Spectral 2D Clusters
            sns.scatterplot(data=df_session, x="pca1", y="pca2", hue="cluster_spectral", palette="Set2", ax=axes[5], legend=False, s=30)
            axes[5].set_title("Spectral Clusters", fontsize=10)
            axes[5].set_xlabel("PCA 1", fontsize=8)
            axes[5].set_ylabel("PCA 2", fontsize=8)
            axes[5].grid(True)
            axes[5].tick_params(axis='both', labelsize=8)

            fig.suptitle(f"Patient: {patient_id} — Session: {drug}", fontsize=12)
            plt.tight_layout(pad=1.5)
            plt.show()

# Save summary to CSV
summary_df = pd.DataFrame(summary_results)
summary_df.to_csv("clustering_summary_kmeans_spectral.csv", index=False)
print("\n✅ Cluster summaries saved to clustering_summary_kmeans_spectral.csv!")
print("\n✅ Visual comparison complete with plots only if show_plots=True!")

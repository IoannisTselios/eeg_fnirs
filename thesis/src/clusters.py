import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

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
        print(f"\nℹ️ PCA reduced to {num_components_95} components explaining 95% of variance for {patient_id} / {drug}")

        # === Cumulative variance plot ===
        cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)

        # === Use first two components for 2D plots ===
        X_pca_2d = X_pca_full[:, :2]

        # === Apply KMeans and Spectral Clustering ===
        clustering_results = {}
        kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
        clustering_results["KMeans"] = kmeans.fit_predict(X_pca_full)

        spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=42)
        clustering_results["Spectral"] = spectral.fit_predict(X_pca_full)

        # === Calculate silhouette scores ===
        print("\n=== Silhouette Scores for this session ===")
        silhouette_scores = {}
        for model_name, labels in clustering_results.items():
            if len(set(labels)) > 1 and -1 not in set(labels):
                score = silhouette_score(X_pca_full, labels)
                silhouette_scores[model_name] = score
                print(f"{model_name}: {score:.3f}")
            else:
                silhouette_scores[model_name] = np.nan
                print(f"{model_name}: N/A (only one cluster or noise detected)")

        # === Save cluster counts and silhouette scores ===
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

        # === Create 4x3 grid for plots ===
        fig, axes = plt.subplots(4, 3, figsize=(18, 16))
        axes = axes.flatten()

        # Plot 1: histogram of log-transformed data
        sns.histplot(X_log.values.flatten(), bins=50, kde=True, color="purple", ax=axes[0])
        axes[0].set_title("Log Power Distribution")
        axes[0].set_xlabel("log10(Power + 1e-9)")
        axes[0].set_ylabel("Count")

        # Plot 2: cumulative variance plot
        axes[1].plot(np.arange(1, len(cumsum_variance) + 1), cumsum_variance, marker='o')
        axes[1].axhline(0.95, color='red', linestyle='--', label="95% variance")
        axes[1].axvline(num_components_95, color='green', linestyle='--', label=f"{num_components_95} components")
        axes[1].set_title("PCA Cumulative Variance")
        axes[1].set_xlabel("Number of Components")
        axes[1].set_ylabel("Cumulative Variance Explained")
        axes[1].legend()

        # Plot 3: PC1 Time Series with Clusters (using KMeans for example)
        df_session["sample_idx"] = np.arange(len(X_pca_full))
        df_session["pc1"] = X_pca_full[:, 0]
        df_session["cluster"] = clustering_results["KMeans"]
        sns.scatterplot(
            data=df_session,
            x="sample_idx", y="pc1",
            hue="cluster",
            palette=sns.color_palette("Set2", len(np.unique(df_session["cluster"]))),
            ax=axes[2], s=20, legend=False
        )
        axes[2].set_title("PC1 Time Series with Clusters")
        axes[2].set_xlabel("Sample Index")
        axes[2].set_ylabel("PC1 Value")
        axes[2].grid(True)

        # Plots 4-5: clustering comparisons (KMeans & Spectral)
        cluster_models = ["KMeans", "Spectral"]
        for idx, model_name in enumerate(cluster_models, start=3):
            ax = axes[idx]
            labels = clustering_results[model_name]
            df_session["pca1"] = X_pca_2d[:, 0]
            df_session["pca2"] = X_pca_2d[:, 1]
            df_session["cluster"] = labels

            num_clusters = len(np.unique(labels))
            palette = sns.color_palette("Set2", num_clusters)

            sns.scatterplot(
                data=df_session,
                x="pca1", y="pca2",
                hue="cluster", style="time",
                palette=palette,
                ax=ax, legend=False, s=60
            )
            ax.set_title(model_name)
            ax.set_xlabel("PCA 1")
            ax.set_ylabel("PCA 2")
            ax.grid(True)

        # Hide any unused subplots
        for i in range(5, 12):
            axes[i].axis('off')

        fig.suptitle(f"Patient: {patient_id} — Session: {drug} — Overview (KMeans & Spectral & PC1 TS)", fontsize=18)
        plt.tight_layout()
        plt.show()

# === Save summary to CSV ===
summary_df = pd.DataFrame(summary_results)
summary_df.to_csv("clustering_summary_kmeans_spectral.csv", index=False)
print("\n✅ Cluster summaries saved to clustering_summary_kmeans_spectral.csv!")
print("\n✅ Full comparison with PC1 time series included complete!")

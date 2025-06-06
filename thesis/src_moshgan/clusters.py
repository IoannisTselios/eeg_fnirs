import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.signal import savgol_filter
from pathlib import Path

# Optional nonlinear projection
from sklearn.manifold import TSNE
import umap.umap_ as umap


# === Control plot display ===
show_plots = True

# === Load new features CSV ===
csv_path = "L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\results_moshgan\\run_20250606_2222\\feature_extraction_files\\eeg_features.csv"
df = pd.read_csv(csv_path)

# === Use only delta and theta features ===
feature_cols = [col for col in df.columns if col.startswith("delta_") or col.startswith("theta_")]

# === Log-transform the power features (avoid log(0)) ===
df_log = df.copy()
df_log[feature_cols] = np.log10(df_log[feature_cols] + 1e-9)

# === Collect results ===
summary_results = []

# === Loop over each patient ===
for patient_id in df_log["id"].unique():
    df_patient = df_log[df_log["id"] == patient_id].copy()

    if len(df_patient) < 5:
        print(f"⚠️ Skipping {patient_id} (too few samples: {len(df_patient)})")
        continue

    # === Scale log-transformed features ===
    X_log = df_patient[feature_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    # === PCA (95% variance explained) ===
    pca = PCA(n_components=0.95, svd_solver="full")
    X_pca = pca.fit_transform(X_scaled)
    num_components = pca.n_components_
    print(f"ℹ️ PCA reduced to {num_components} components for {patient_id}")

    # Save PCA 2D version for scatter plot
    X_pca_2d = X_pca[:, :2]
    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)

    # === Clustering ===
    clustering_results = {
        "KMeans": KMeans(n_clusters=2, random_state=42, n_init=20).fit_predict(X_pca),
        "Spectral": SpectralClustering(n_clusters=2, affinity='nearest_neighbors', gamma=0.01, n_neighbors=10, random_state=42).fit_predict(X_pca),
    }

    # === Evaluate clustering ===
    silhouette_scores = {}
    for model_name, labels in clustering_results.items():
        if len(set(labels)) > 1 and -1 not in set(labels):
            score = silhouette_score(X_pca, labels)
        else:
            score = np.nan
        silhouette_scores[model_name] = score
        print(f"{model_name} for {patient_id}: {score:.3f}" if not np.isnan(score) else f"{model_name} for {patient_id}: N/A")

    for model_name, labels in clustering_results.items():
        counts = pd.Series(labels).value_counts().to_dict()
        row = {
            "patient_id": patient_id,
            "model": model_name,
            "pca_components_95": num_components,
            "silhouette_score": silhouette_scores[model_name]
        }
        for cluster_label, count in counts.items():
            row[f"cluster_{cluster_label}"] = count
        summary_results.append(row)

    # === Optional plots ===
    if show_plots:
        fig, axes = plt.subplots(4, 2, figsize=(14, 14))
        axes = axes.flatten()

        # Plot 1: Log Power Histogram
        sns.histplot(X_log.values.flatten(), bins=50, kde=True, color="purple", ax=axes[0])
        axes[0].set_title("Log Power Distribution")

        # Plot 2: PCA Cumulative Variance
        axes[1].plot(np.arange(1, len(cumsum_variance) + 1), cumsum_variance, marker='o')
        axes[1].axhline(0.95, color='red', linestyle='--')
        axes[1].axvline(num_components, color='green', linestyle='--')
        axes[1].set_title("PCA Cumulative Variance")

        # Plot 3–4: PC1 trends with clustering
        df_patient["sample_idx"] = np.arange(len(X_pca))
        df_patient["pc1"] = X_pca[:, 0]
        trend = savgol_filter(df_patient["pc1"], 21, 2)

        for i, model_name in enumerate(["KMeans", "Spectral"]):
            df_patient[f"cluster_{model_name.lower()}"] = clustering_results[model_name]
            colors = df_patient[f"cluster_{model_name.lower()}"].map(dict(zip(np.unique(clustering_results[model_name]), sns.color_palette("Set2"))))

            axes[2 + i].plot(df_patient["sample_idx"], df_patient["pc1"], color='gray', alpha=0.6)
            axes[2 + i].scatter(df_patient["sample_idx"], df_patient["pc1"], c=colors, s=20, edgecolor='black')
            axes[2 + i].plot(df_patient["sample_idx"], trend, color='red', linewidth=1.5, label='Trend')
            axes[2 + i].set_title(f"PC1 ({model_name})")

        # Plot 5–6: 2D PCA Scatterplots
        df_patient["pca1"] = X_pca_2d[:, 0]
        df_patient["pca2"] = X_pca_2d[:, 1]
        for i, model_name in enumerate(["KMeans", "Spectral"]):
            sns.scatterplot(data=df_patient, x="pca1", y="pca2", hue=f"cluster_{model_name.lower()}", palette="Set2", ax=axes[4 + i], legend=False)
            axes[4 + i].set_title(f"{model_name} Clusters (2D PCA)")

        # Plot 7: 2D PCA colored by annotation (if exists)
        if "annotation" in df_patient.columns:
            sns.scatterplot(data=df_patient, x="pca1", y="pca2", hue="annotation", palette="tab10", ax=axes[6])
            axes[6].set_title("2D PCA Colored by Annotation")

        # Plot 8: UMAP and t-SNE comparison
        try:
            reducer_umap = umap.UMAP(n_components=2, random_state=42)
            X_umap = reducer_umap.fit_transform(X_scaled)
            sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], ax=axes[7], hue=df_patient.get("annotation", None), palette="tab10", legend=False)
            axes[7].set_title("UMAP Projection")
        except Exception as e:
            axes[7].text(0.5, 0.5, f"UMAP error: {str(e)}", ha='center', va='center')

        fig.suptitle(f"Patient: {patient_id}", fontsize=12)
        plt.tight_layout(pad=1.5)
        plt.show()

# === Save summary ===
summary_df = pd.DataFrame(summary_results)
summary_df.to_csv("clustering_summary_per_patient.csv", index=False)
print("\n✅ Saved to clustering_summary_per_patient.csv!")

# === Optional: Later step — filter unlabeled/resting epochs ===
# unlabeled_labels = ["Resting", "Unlabeled", "None", np.nan]
# df_filtered = df_log[~df_log["annotation"].isin(unlabeled_labels)].copy()
# → You can re-run the clustering section on df_filtered instead of df_log.

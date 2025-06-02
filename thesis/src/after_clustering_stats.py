import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# === Load the clustering summary CSV ===
csv_path = "L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\thesis\\src\\clustering_summary_kmeans_spectral.csv"
df_summary = pd.read_csv(csv_path)

# === Clean data: remove rows with missing silhouette scores ===
df_summary_clean = df_summary.dropna(subset=["silhouette_score"])

# === Extract clinical state (UWS vs MCS) ===
df_summary_clean['clinical_state'] = df_summary_clean['patient_id'].apply(
    lambda x: 'MCS' if 'MCS' in x or ' M' in x else 'UWS'
)

# === Extract session number ===
df_summary_clean['session_number'] = df_summary_clean['session'].str.extract('(\d+)').astype(float)

# === 1️⃣ Cluster Size Difference (balance) ===
def cluster_size_diff(row):
    cluster_cols = [col for col in row.index if col.startswith("cluster_")]
    if len(cluster_cols) == 2:
        return abs(row[cluster_cols[0]] - row[cluster_cols[1]])
    else:
        return None

df_summary_clean["cluster_size_diff"] = df_summary_clean.apply(cluster_size_diff, axis=1)
df_summary_clean['cluster_ratio'] = df_summary_clean.apply(
    lambda row: min(row['cluster_0'], row['cluster_1']) / max(row['cluster_0'], row['cluster_1']),
    axis=1
)

# Averages & Best Cases
size_diff_avg = df_summary_clean.groupby("model")["cluster_size_diff"].mean()
best_row = df_summary_clean.loc[df_summary_clean["silhouette_score"].idxmax()]
avg_silhouette_patient = df_summary_clean.groupby("patient_id")["silhouette_score"].mean().sort_values(ascending=False)
avg_silhouette_session = df_summary_clean.groupby("session")["silhouette_score"].mean().sort_values(ascending=False)

# Clinical comparison (t-test)
mcs_scores = df_summary_clean[df_summary_clean['clinical_state'] == 'MCS']['silhouette_score']
uws_scores = df_summary_clean[df_summary_clean['clinical_state'] == 'UWS']['silhouette_score']
stat, pval = stats.ttest_ind(mcs_scores, uws_scores)

# === Create directory structure ===
base_dir = "clustering_analysis_outputs"
sub_dirs = [
    "1_cluster_size_balance",
    "2_silhouette_patient",
    "3_silhouette_session",
    "4_session_progression",
    "5_cluster_balance_vs_performance",
    "6_pca_components",
    "7_silhouette_distribution"
]
for sub in sub_dirs:
    os.makedirs(os.path.join(base_dir, sub), exist_ok=True)

# === 1️⃣ Cluster Size Difference by Model ===
plt.figure(figsize=(8, 4))
sns.boxplot(x="model", y="cluster_size_diff", hue="model", data=df_summary_clean, palette="Set2", dodge=False)
plt.title("Cluster Size Difference by Model")
plt.ylabel("Absolute Cluster Size Difference")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "1_cluster_size_balance", "cluster_size_difference.png"))
plt.close()

# === 2️⃣ Silhouette Score by Patient ===
plt.figure(figsize=(10, 5))
sns.boxplot(x="patient_id", y="silhouette_score", hue="model", data=df_summary_clean, palette="Set2", dodge=True)
plt.title("Silhouette Score by Patient")
plt.ylabel("Silhouette Score")
plt.xlabel("Patient ID")
plt.xticks(rotation=90)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "2_silhouette_patient", "silhouette_by_patient.png"))
plt.close()

# Average silhouette score by patient
plt.figure(figsize=(10, 5))
sns.barplot(x=avg_silhouette_patient.index, y=avg_silhouette_patient.values, palette='coolwarm')
plt.title("Average Silhouette Score by Patient")
plt.ylabel("Silhouette Score")
plt.xlabel("Patient ID")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "2_silhouette_patient", "average_silhouette_by_patient.png"))
plt.close()

# === 3️⃣ Silhouette Score by Session ===
plt.figure(figsize=(6, 4))
sns.boxplot(x="session", y="silhouette_score", hue="model", data=df_summary_clean, palette="Set2", dodge=True)
plt.title("Silhouette Score by Session")
plt.ylabel("Silhouette Score")
plt.xlabel("Session")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "3_silhouette_session", "silhouette_by_session.png"))
plt.close()

# Average silhouette score by session
plt.figure(figsize=(6, 4))
sns.barplot(x=avg_silhouette_session.index, y=avg_silhouette_session.values, palette='coolwarm')
plt.title("Average Silhouette Score by Session")
plt.ylabel("Silhouette Score")
plt.xlabel("Session")
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "3_silhouette_session", "average_silhouette_by_session.png"))
plt.close()

# === 4️⃣ Session Progression (by patient) ===
plt.figure(figsize=(12, 6))
for patient in df_summary_clean['patient_id'].unique():
    patient_data = df_summary_clean[df_summary_clean['patient_id'] == patient]
    if len(patient_data) > 1:
        plt.plot(patient_data['session_number'], 
                 patient_data['silhouette_score'], 
                 'o-', 
                 label=patient)
plt.title('Silhouette Score Progression Across Sessions')
plt.xlabel('Session Number')
plt.ylabel('Silhouette Score')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "4_session_progression", "silhouette_progression.png"), bbox_inches='tight')
plt.close()

# === 5️⃣ Cluster Balance vs Performance ===
plt.figure()
sns.scatterplot(data=df_summary_clean, 
                x='cluster_ratio', 
                y='silhouette_score',
                hue='clinical_state',
                style='model', palette="Set2")
plt.title('Cluster Balance vs Clustering Performance')
plt.xlabel('Cluster Balance Ratio (smaller/larger)')
plt.ylabel('Silhouette Score')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "5_cluster_balance_vs_performance", "cluster_balance_vs_performance.png"))
plt.close()

# === 6️⃣ PCA Components by Clinical State ===
plt.figure()
sns.boxplot(x='clinical_state', y='pca_components_95', data=df_summary_clean, palette="Set2")
plt.title('PCA Components Needed by Clinical State')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "6_pca_components", "pca_components_by_state.png"))
plt.close()

# === 7️⃣ Silhouette Score Distribution ===
plt.figure()
sns.histplot(data=df_summary_clean, x='silhouette_score', hue='clinical_state', multiple="stack", palette="Set2")
plt.title('Distribution of Silhouette Scores by Clinical State')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "7_silhouette_distribution", "silhouette_distribution.png"))
plt.close()

# === Summary ===
print("\n=== Summary of Findings ===")
print(f"- Best-performing: {best_row['patient_id']} / {best_row['session']} (Model: {best_row['model']} with Silhouette Score: {best_row['silhouette_score']:.3f})")
print(f"- Average cluster size difference:\n{size_diff_avg}")
print(f"- Top 3 patients by silhouette score:\n{avg_silhouette_patient.head(3)}")
print(f"- Best session by average silhouette score: Session {avg_silhouette_session.idxmax()} ({avg_silhouette_session.max():.3f})")
print(f"- UWS vs MCS comparison (t-test): t={stat:.3f}, p={pval:.3f}")

print(f"\n✅ All plots and stats saved in '{base_dir}' subfolders.")

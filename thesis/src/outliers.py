import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Load processed EEG dataset
df = pd.read_csv("L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\feature_extraction_files\\lmm_Processed_data.csv")

# List of numerical feature columns (excluding categorical ones)
feature_cols = [col for col in df.columns if col not in ["id", "drug", "time"]]

outliers_iqr = {}

for feature in feature_cols:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_iqr[feature] = ((df[feature] < lower_bound) | (df[feature] > upper_bound)).sum()

outliers_iqr = pd.Series(outliers_iqr).sort_values(ascending=False)

print("\nNumber of outliers detected per feature (IQR Method):")
print(outliers_iqr)


plt.figure(figsize=(15, 10))
df[feature_cols].boxplot(rot=45, patch_artist=True)
plt.title("Boxplot of EEG Features (Detecting Outliers)")
plt.xticks(rotation=90)
plt.show()

for feature in feature_cols[:5]:  # Visualize first 5 features
    plt.figure(figsize=(8, 5))
    sns.histplot(df[feature], kde=True, bins=30)
    plt.axvline(df[feature].mean(), color='r', linestyle='dashed', label="Mean")
    plt.axvline(df[feature].quantile(0.25), color='g', linestyle='dashed', label="Q1")
    plt.axvline(df[feature].quantile(0.75), color='g', linestyle='dashed', label="Q3")
    plt.legend()
    plt.title(f"Distribution of {feature}")
    plt.show()


# df["outlier_count"] = ((df[feature_cols] < (df[feature_cols].quantile(0.25) - 1.5 * (df[feature_cols].quantile(0.75) - df[feature_cols].quantile(0.25)))) | 
#                        (df[feature_cols] > (df[feature_cols].quantile(0.75) + 1.5 * (df[feature_cols].quantile(0.75) - df[feature_cols].quantile(0.25)))))
# df["outlier_count"] = df["outlier_count"].sum(axis=1)  # Sum up outliers per patient

# # Filter patients with most outliers
# df_outliers = df[df["outlier_count"] > 0]

# print("\n🔍 Patients with the most outliers:")
# print(df_outliers[["id", "drug", "outlier_count"]].sort_values(by="outlier_count", ascending=False).head(10))


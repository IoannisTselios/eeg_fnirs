import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# ---- [Data preparation] Normalize features ----
def normalization(df):
    ss = StandardScaler()
    for feature in df.columns:
        if feature in ['id', 'drug', 'time']:
            continue
        df[feature] = ss.fit_transform(df[feature].values.reshape(-1, 1))

    return df


# ---- Visualizations ----
def general_violin_plots(df, ylabels, name):

    for feature in df.columns:
        if feature in ['id', 'drug', 'time']:
            continue

        plt.figure(figsize=(8, 6))
        sns.violinplot(data=df, x="drug", y=feature, hue="time", palette='pastel')
        plt.title("Distribution of the values for the feature: {}".format(feature))
        plt.ylabel(ylabels[feature])
        plt.xlabel("Drug group")
        plt.savefig(r"L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\plots\\" + feature + "_" + name +".png")


def individual_violin_plots(df, ylabels):
    output_dir = r"L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\plots\\By Patient"
    os.makedirs(output_dir, exist_ok=True)

    # Generate violin plots for each patient and feature
    for patient_id in df['id'].unique():  # Iterate over unique patients
        df_patient = df[df['id'] == patient_id]  # Filter data for the specific patient

        for feature in df.columns:
            if feature in ['id', 'drug', 'time']:  # Skip categorical columns
                continue

            plt.figure(figsize=(8, 6))
            sns.violinplot(data=df_patient, x="drug", y=feature, hue="time", palette='pastel')
            plt.title(f"Patient {patient_id} - {feature} Distribution")
            plt.ylabel(ylabels.get(feature, feature))  # Use label if available
            plt.xlabel("Drug group")
            
            # Save plot for each patient-feature combination
            save_path = os.path.join(output_dir, f"Patient_{patient_id}_{feature}.png")
            plt.savefig(save_path)
            plt.close()  # Close plot to avoid memory issues

    print("Violin plots saved in:", output_dir)

def outliers_plots(df, ylabels, top_patients):
    # Define an output folder for individual patients
    output_dir_outliers = os.path.join("L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\", "outliers")
    os.makedirs(output_dir_outliers, exist_ok=True)

    for feature in df.columns:
        if feature in ['id', 'drug', 'time']:  # Skip categorical columns
            continue

        # Get the patient with the largest deviation for this feature
        outlier_patient = top_patients[feature]

        # Filter the dataframe for the outlier patient
        df_outlier = df[df["id"] == outlier_patient]

        # Create the violin plot
        plt.figure(figsize=(8, 6))
        sns.violinplot(data=df_outlier, x="drug", y=feature, hue="time", palette='pastel')
        plt.title(f"Outlier Patient {outlier_patient} - {feature} Distribution")
        plt.ylabel(ylabels.get(feature, feature))  # Use label if available
        plt.xlabel("Drug group")

        # Save plot for each outlier patient-feature combination
        save_path = os.path.join(output_dir_outliers, f"Outlier_Patient_{outlier_patient}_{feature}.png")
        plt.savefig(save_path)
        plt.close()

    print("Outlier patient violin plots saved in:", output_dir_outliers)


def remove_outliers_iqr(df, features):
    """
    Identify and remove outliers using the Interquartile Range (IQR) method for each feature.
    Returns:
    - The filtered dataframe.
    - A dictionary with the number of outliers removed per feature.
    - A detailed DataFrame listing each outlier, its patient ID, and feature values.
    """
    df_filtered = df.copy()
    outlier_counts = {}
    outlier_details = []

    for feature in features:
        if feature in ['id', 'drug', 'time']:  # Skip categorical columns
            continue

        Q1 = df_filtered[feature].quantile(0.25)  # 25th percentile (lower quartile)
        Q3 = df_filtered[feature].quantile(0.75)  # 75th percentile (upper quartile)
        IQR = Q3 - Q1  # Interquartile range

        # Define outlier thresholds (1.5 * IQR rule)
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers before removal
        outlier_mask = (df_filtered[feature] < lower_bound) | (df_filtered[feature] > upper_bound)
        outlier_count = outlier_mask.sum()
        outlier_counts[feature] = outlier_count

        # Collect outlier details
        if outlier_count > 0:
            outlier_rows = df_filtered[outlier_mask][['id', 'drug', 'time', feature]].copy()
            outlier_rows['Feature'] = feature
            outlier_rows['Lower Bound'] = lower_bound
            outlier_rows['Upper Bound'] = upper_bound
            outlier_details.append(outlier_rows)

        # Remove outliers for this feature
        df_filtered = df_filtered[~outlier_mask]

    # Combine all outlier details into a single DataFrame
    outlier_details_df = pd.concat(outlier_details, ignore_index=True) if outlier_details else pd.DataFrame()

    return df_filtered, outlier_counts, outlier_details_df


# ---- Load ----

df = pd.read_csv(os.path.join("L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\feature_extraction_files\\eeg_features.csv"))

# ---- [Data preparation] Substract baseline from corresponding post administration 1 and 2 ----

use_median_baseline = False

only_baseline_df = df[df['time'] == 0].copy()
only_post_one_df = df[df['time'] == 1].copy()
only_post_two_df = df[df['time'] == 2].copy()
del df

median_baseline = only_baseline_df.iloc[:, 3:].median().values if use_median_baseline else None

for df_part  in [only_post_one_df, only_post_two_df]:

    for index, row in df_part.iterrows():
        patient_id = row['id']
        drug = row['drug']

        baseline_session = only_baseline_df[(only_baseline_df['drug'] == drug) & (only_baseline_df['id'] == patient_id)]

        if not baseline_session.empty:
            baseline_values = baseline_session.iloc[0, 3:].values  # Extract baseline
            df_part.loc[index, df_part.columns.values[3:]] -= baseline_values
        elif median_baseline is not None:
            df_part.loc[index, df_part.columns.values[3:]] -= median_baseline
        else:
            df_part.loc[index, df_part.columns.values[3:]] = np.nan  # If no baseline, set NaN

df = pd.concat([only_post_one_df, only_post_two_df])
del only_baseline_df, only_post_one_df, only_post_two_df
        
df = df.fillna(0)
df.to_csv(os.path.join("L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\feature_extraction_files\\lmm_Processed_data.csv"), index=False)

ylabels = {
    'delta': 'Percentage of the total signal\'s power spectral density',
    'theta': 'Percentage of the total signal\'s power spectral density',
    'alpha': 'Percentage of the total signal\'s power spectral density',
    'ratio': 'Alpha/Delta Ratio',
}

# df_baseline = only_baseline_df.copy().drop(columns=["time"])  # Keep only id, drug, and features
# df_baseline = df_baseline.rename(columns=lambda x: x if x in ["id", "drug"] else f"{x}_baseline")

# df_post = pd.concat([only_post_one_df, only_post_two_df])  # Combine post-administration data
# df_post = df_post.merge(df_baseline, on=["id", "drug"], how="left")  # Merge on 'id' and 'drug'

# # Subtract baseline where available, otherwise use median baseline or NaN
# feature_cols = [col for col in df.columns if col not in ["id", "drug", "time"]]

# for feature in feature_cols:
#     df_post[feature] = df_post[feature] - df_post[f"{feature}_baseline"]

# # Drop baseline columns
# df_post = df_post.drop(columns=[f"{feature}_baseline" for feature in feature_cols])
# df_post = df_post.dropna()  # Remove NaN values
print("Number of data points: {}".format(len(df)))


# ---- Compute Patient Variability ----
numeric_df = df.select_dtypes(include=[np.number]).copy()
feature_variability = numeric_df.groupby(df["id"]).std()
top_patients = feature_variability.idxmax()

print("Patients with the highest variability per feature:")
print(top_patients)

# ---- Run Visualizations ----
# normalization(df)  # Normalize the data
general_violin_plots(df, ylabels, "before_outliers")  # Generate general violin plots
# individual_violin_plots(df)  # Generate violin plots per patient
outliers_plots(df, ylabels, top_patients)  # Generate outlier patient plots
# Step 1: Identify outlier patients
outlier_patients = top_patients.unique()  # Get unique patient IDs with highest variability

# Step 2: Remove outlier patients from the dataset
df_no_outliers = df[~df["id"].isin(outlier_patients)]

# Step 3: Re-run general violin plots without outliers
general_violin_plots(df_no_outliers, ylabels, "After_the_most_extreme_value")

df_no_outliers, outlier_stats, outlier_details_df = remove_outliers_iqr(df, df.columns)

# Display how many outliers were removed per feature
print("Outliers removed per feature:")
for feature, count in outlier_stats.items():
    print(f"Feature: {feature}, Outliers Removed: {count}")

print(outlier_details_df.head())

# Show all removed outliers
print("Outliers Details:")
print(outlier_details_df.to_string())

# Count how many times each patient appears in the outlier list
patient_outlier_counts = outlier_details_df['id'].value_counts()
print(patient_outlier_counts)

general_violin_plots(df_no_outliers, ylabels, "After_removing_multiple_values")


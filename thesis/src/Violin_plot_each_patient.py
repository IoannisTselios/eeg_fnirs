import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# ---- Load ----

df = pd.read_csv(os.path.join("L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\feature_extraction_files\\eeg_features.csv"))

# ---- [Data preparation] Substract baseline from corresponding post administration 1 and 2 ----

substract_baseline = True
use_median_baseline = False

if substract_baseline:
    
    # Save the average (with 'mean') or median (with 'median') baseline for the recordings that don't have a baseline # TODO : is that a good idea? Probably not
    median_baseline = df.drop(['drug', 'id'], axis=1).groupby("time").agg({'median'}).values[0]

    only_baseline_df = df[df['time'] == 0]
    only_post_one_df = df[df['time'] == 1]
    only_post_two_df = df[df['time'] == 2]
    del df

    for df in [only_post_one_df, only_post_two_df]:

        for index, row in df.iterrows():
            id = row['id']
            drug = row['drug']
            try:
                baseline_for_this_session = only_baseline_df.loc[(only_baseline_df['drug'] == drug) & (only_baseline_df['id'] == id)].values[0]
                df.loc[index, df.columns.values[3:]] -= baseline_for_this_session[3:]
            except Exception as e: # Most likely the baseline was not found
                print(e)
                if use_median_baseline:
                    df.loc[index,df.columns.values[3:]] -= median_baseline
                else:
                    df.loc[index,df.columns.values[3:]] = [np.nan] * len(median_baseline)

    df = pd.concat([only_post_one_df, only_post_two_df])
    del only_baseline_df, only_post_one_df, only_post_two_df
        
df = df.dropna()

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

# ---- [Data preparation] Normalize features ----

normalize = True

if normalize:
    ss = StandardScaler()
    for feature in df.columns:
        if feature in ['id', 'drug', 'time']:
            continue
        df[feature] = ss.fit_transform(df[feature].values.reshape(-1, 1))

df.to_csv(os.path.join("L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\feature_extraction_files\\lmm_Processed_data.csv"), index=False)

# ---- Visualizations ----

violin_plots = True

if violin_plots:
    ylabels = {'delta': 'Percentage of the total signal\'s power spectral density',
               'theta': 'Percentage of the total signal\'s power spectral density',
               'alpha': 'Percentage of the total signal\'s power spectral density',
               'ratio': 'Ratio'}
    for feature in df.columns:
        if feature in ['id', 'drug', 'time']:
            continue

        plt.figure(figsize=(8, 6))
        sns.violinplot(data=df, x="drug", y=feature, hue="time", palette='pastel')
        plt.title("Distribution of the values for the feature: {}".format(feature))
        plt.ylabel(ylabels[feature])
        plt.xlabel("Drug group")
        plt.savefig(r"L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\plots\\" + feature + ".png")


violin_plots_individual = False
if violin_plots_individual:
    output_dir = r"L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\plots\\By Patient"
    os.makedirs(output_dir, exist_ok=True)

    # Define custom labels for better readability
    ylabels = {
        'delta': 'Percentage of the total signal\'s power spectral density',
        'theta': 'Percentage of the total signal\'s power spectral density',
        'alpha': 'Percentage of the total signal\'s power spectral density',
        'ratio': 'Alpha/Delta Ratio',
        'pe': 'Permutation entropy',
        'se': 'Spectral entropy',
        'fnirs_1': 'Average slope of the hemoglobin signal',
        'pupillometry_score': 'Pupil dilation count during mental tasks'
    }

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
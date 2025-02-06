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
                # print(e)
                if use_median_baseline:
                    df.loc[index,df.columns.values[3:]] -= median_baseline
                else:
                    df.loc[index,df.columns.values[3:]] = [np.nan] * len(median_baseline)

    df = pd.concat([only_post_one_df, only_post_two_df])
    del only_baseline_df, only_post_one_df, only_post_two_df
        
df = df.dropna()
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
# df.to_excel(os.path.join("data", "processed", "lmm_data.xlsx"), index=False)

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
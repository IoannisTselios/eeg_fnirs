import logging
import os
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

class EEGVisualizer:

    def __init__(self, feature_output_dir, plot_output_dir):
        self.feature_output_dir = Path(feature_output_dir)
        self.plot_output_dir = Path(plot_output_dir)
        logging.info("EEG Outlier Analyzer Initialized")

    def general_violin_plots(self, df, ylabels, outlier_details_df, name):
        output_dir = self.plot_output_dir / "Violin_Plots"
        output_dir.mkdir(parents=True, exist_ok=True)

        for feature in df.columns:
            if feature == 'id':
                continue

            plt.figure(figsize=(8, 6))
            sns.violinplot(y=df[feature], palette='pastel')

            if not outlier_details_df.empty:
                feature_outliers = outlier_details_df[outlier_details_df['Feature'] == feature]
                for _, row in feature_outliers.iterrows():
                    plt.scatter(0, row[feature], color='red', s=50)
                    plt.text(0.05, row[feature], row['id'], color='red', fontsize=8)

            plt.title(f"Distribution of {feature}")
            plt.ylabel(ylabels.get(feature, feature))
            plt.xlabel("")
            plt.savefig(output_dir / f"{feature}_{name}.png")
            plt.close()

    def plot_histograms(self, df, ylabels, outlier_details_df, name="histogram"):
        output_dir = self.plot_output_dir / "Histograms"
        output_dir.mkdir(parents=True, exist_ok=True)

        for feature in df.columns:
            if feature == 'id':
                continue

            plt.figure(figsize=(8, 6))
            sns.histplot(df[feature], kde=True, color='skyblue')

            if not outlier_details_df.empty:
                feature_outliers = outlier_details_df[outlier_details_df['Feature'] == feature]
                for _, row in feature_outliers.iterrows():
                    plt.axvline(x=row[feature], color='red', linestyle='--')
                    plt.text(row[feature], plt.ylim()[1]*0.8, row['id'], rotation=90, color='red', fontsize=8)

            plt.title(f"Histogram of {feature}")
            plt.xlabel(ylabels.get(feature, feature))
            plt.ylabel("Count")
            plt.savefig(output_dir / f"{feature}_{name}.png")
            plt.close()

    def plot_correlation(self, df, outlier_details_df):
        # Familiar voice correlation
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df["delta_familiar"], y=df["theta_familiar"], color='skyblue', s=40)

        if not outlier_details_df.empty:
            for _, row in outlier_details_df.iterrows():
                if row['Feature'] in ['delta_familiar', 'theta_familiar']:
                    x = row['delta_familiar'] if not pd.isna(row['delta_familiar']) else df.loc[df['id'] == row['id'], 'delta_familiar'].values[0]
                    y = row['theta_familiar'] if not pd.isna(row['theta_familiar']) else df.loc[df['id'] == row['id'], 'theta_familiar'].values[0]
                    plt.scatter(x, y, color='red', s=80)
                    plt.text(x, y, row['id'], color='red', fontsize=8)

        plt.title("Familiar Voice: Delta vs Theta Power")
        plt.xlabel("Delta Band Power (Familiar)")
        plt.ylabel("Theta Band Power (Familiar)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.plot_output_dir / "delta_theta_correlation_familiar.png")
        plt.close()

        # Medical voice correlation
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df["delta_medical"], y=df["theta_medical"], color='lightgreen', s=40)

        if not outlier_details_df.empty:
            for _, row in outlier_details_df.iterrows():
                if row['Feature'] in ['delta_medical', 'theta_medical']:
                    x = row['delta_medical'] if not pd.isna(row['delta_medical']) else df.loc[df['id'] == row['id'], 'delta_medical'].values[0]
                    y = row['theta_medical'] if not pd.isna(row['theta_medical']) else df.loc[df['id'] == row['id'], 'theta_medical'].values[0]
                    plt.scatter(x, y, color='red', s=80)
                    plt.text(x, y, row['id'], color='red', fontsize=8)

        plt.title("Medical Voice: Delta vs Theta Power")
        plt.xlabel("Delta Band Power (Medical)")
        plt.ylabel("Theta Band Power (Medical)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.plot_output_dir / "delta_theta_correlation_medical.png")
        plt.close()


    def basic_statistics(self, df):
        stats = df.describe().T
        stats['skewness'] = df.select_dtypes(include=[np.number]).skew()
        stats['kurtosis'] = df.select_dtypes(include=[np.number]).kurtosis()
        stats.to_csv(self.feature_output_dir / "basic_statistics.csv")

    def remove_outliers_iqr(self, df, features):
        df_filtered = df.copy()
        outlier_counts = {}
        outlier_details = []

        for feature in features:
            if feature == 'id':
                continue

            Q1 = df_filtered[feature].quantile(0.25)
            Q3 = df_filtered[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_mask = (df_filtered[feature] < lower_bound) | (df_filtered[feature] > upper_bound)
            outlier_count = outlier_mask.sum()
            outlier_counts[feature] = outlier_count

            if outlier_count > 0:
                outlier_rows = df_filtered[outlier_mask][['id', feature]].copy()
                outlier_rows['Feature'] = feature
                outlier_rows['Lower Bound'] = lower_bound
                outlier_rows['Upper Bound'] = upper_bound
                outlier_details.append(outlier_rows)

            df_filtered = df_filtered[~outlier_mask]

        outlier_details_df = pd.concat(outlier_details, ignore_index=True) if outlier_details else pd.DataFrame()

        return df_filtered, outlier_counts, outlier_details_df

    def process(self):
        df = pd.read_csv(self.feature_output_dir / "eeg_features.csv")
        df = df.fillna(0)
        df.to_csv(self.feature_output_dir / "lmm_Processed_data.csv", index=False)

        ylabels = {
            'delta': 'Delta Band Power (0.5-4 Hz)',
            'theta': 'Theta Band Power (4-8 Hz)',
        }

        # df_no_outliers, outlier_stats, outlier_details_df = self.remove_outliers_iqr(df, df.columns)
        # outlier_details_df.to_csv(self.feature_output_dir / "eeg_outliers_detail_df.csv", index=False)

        print(f"Number of data points: {len(df)}")
        print("Outliers loaded:")
        # print(outlier_details_df[['id', 'Feature']])

        self.basic_statistics(df)
        self.general_violin_plots(df, ylabels, outlier_details_df = [], name="before_outliers")
        self.plot_histograms(df, ylabels, outlier_details_df = [])
        self.plot_correlation(df, outlier_details_df = [])

        print(f"✅ All plots saved in {self.plot_output_dir}")
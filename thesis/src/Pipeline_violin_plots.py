import logging
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import os

warnings.filterwarnings("ignore")

# ---- EEG Visualization ----
class EEGVisualizer:

    def __init__(self, feature_output_dir, plot_output_dir):
        self.feature_output_dir = feature_output_dir
        self.plot_output_dir = plot_output_dir
        logging.info("EEG Outlier Analyzer Initialized")

    # ---- Visualizations ----
    def general_violin_plots(self, df, ylabels, name):

        for feature in df.columns:
            if feature in ['id', 'drug', 'time']:
                continue

            plt.figure(figsize=(8, 6))
            sns.violinplot(data=df, x="drug", y=feature, hue="time", palette='pastel')
            plt.title("Distribution of the values for the feature: {}".format(feature))
            plt.ylabel(ylabels[feature])
            plt.xlabel("Drug group")
            plt.savefig(self.plot_output_dir / Path(f"{feature}_{name}.png"))

    def individual_violin_plots(self, df, ylabels):
        output_dir = self.plot_output_dir / Path("By Patient")
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

    def outliers_plots(self, df, ylabels, top_patients):
        # Define an output folder for individual patients
        output_dir_outliers = os.path.join(self.plot_output_dir, "outliers")
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
            save_path = os.path.join(output_dir_outliers, Path(f"Outlier_Patient_{outlier_patient}_{feature}.png"))
            plt.savefig(save_path)
            plt.close()

        print("Outlier patient violin plots saved in:", output_dir_outliers)

    def plot_all_outlier_patients(self, df, outlier_details_df, ylabels, name_prefix="outliers_analysis", use_log_scale=False):
        output_dir = self.plot_output_dir / Path("All_Outlier_Patients")
        os.makedirs(output_dir, exist_ok=True)

        outlier_patients = outlier_details_df['id'].unique()
        patient_colors = sns.color_palette("tab10", len(outlier_patients))
        patient_color_map = {pid: color for pid, color in zip(outlier_patients, patient_colors)}

        for patient_id in outlier_patients:
            df_patient = df[df['id'] == patient_id].copy()
            df_outlier_patient = outlier_details_df[outlier_details_df['id'] == patient_id].copy()

            if df_patient.empty:
                print(f"⚠️ Warning: No data found for Patient {patient_id}")
                continue

            df_patient["time"] = df_patient["time"].astype(int)
            df_outlier_patient["time"] = df_outlier_patient["time"].astype(int)

            num_features = len(ylabels.keys())
            fig, axes = plt.subplots(num_features, 2, figsize=(18, num_features * 4), sharex=True)

            if num_features == 1:
                axes = np.array([axes])

            for i, feature in enumerate(ylabels.keys()):
                ax_ts, ax_box = axes[i]

                # ---- Time Series Plot ----
                try:
                    ax_ts.plot(df_patient['time'], df_patient[feature], marker='o', linestyle='dashed',
                            markersize=6, alpha=0.7, color=patient_color_map[patient_id], label=f"Patient {patient_id}")

                    outlier_points = df_outlier_patient[df_outlier_patient["Feature"] == feature]
                    if not outlier_points.empty:
                        ax_ts.scatter(outlier_points["time"], outlier_points[feature], color="red", s=80, label="Outlier")

                        for _, row in outlier_points.iterrows():
                            ax_ts.annotate(f"{row[feature]:.2f}", (row["time"], row[feature]), textcoords="offset points",
                                        xytext=(5, 5), ha='center', fontsize=10, color="red")

                    ax_ts.set_ylabel(ylabels[feature])
                    ax_ts.set_title(f"Patient {patient_id} - {feature} Over Time")
                    ax_ts.grid(True)

                    if use_log_scale:
                        ax_ts.set_yscale("log")

                except Exception as e:
                    print(f"❌ Error plotting time series for patient {patient_id}, feature {feature}: {e}")

                # ---- Box Plot ----
                try:
                    sns.boxplot(data=df, x="time", y=feature, ax=ax_box, palette="pastel")

                    # ✅ Catch if the data is empty or malformed
                    if not df_patient.empty:
                        sns.stripplot(data=df_patient, x="time", y=feature, ax=ax_box,
                                    color=patient_color_map[patient_id], size=8, jitter=True)

                    if not df_outlier_patient.empty:
                        sns.stripplot(data=df_outlier_patient, x="time", y=feature, ax=ax_box,
                                    color="red", size=8, jitter=True, label="Outliers")

                    ax_box.set_ylabel(ylabels[feature])
                    ax_box.set_title(f"Feature Distribution Over Time - {feature}")
                    ax_box.grid(True)

                except Exception as e:
                    print(f"❌ Error plotting boxplot for patient {patient_id}, feature {feature}: {e}")

            axes[-1, 0].set_xlabel("Time")
            axes[-1, 1].set_xlabel("Time")
            plt.xticks(sorted(df_patient["time"].unique()), rotation=45)

            # ✅ Clean up plots and avoid memory leaks
            try:
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"Outlier_Patient_{patient_id}_{name_prefix}.png"))
                plt.close()
            except Exception as e:
                print(f"❌ Error saving plot for patient {patient_id}: {e}")

        print(f"📊 All outlier patient plots saved in: {output_dir}")


    def remove_outliers_iqr(self, df, features):
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

    def process(self):
        # Load dataset
        df = pd.read_csv(os.path.join(self.feature_output_dir / "eeg_features.csv"))

        # ---- [Data preparation] Compute log(post) - log(baseline) ----

        use_median_baseline = False  # Toggle to use median baseline

        # Separate the dataset by time points
        only_baseline_df = df[df['time'] == 0].copy()
        only_post_one_df = df[df['time'] == 1].copy()
        only_post_two_df = df[df['time'] == 2].copy()
        del df  # Free memory

        # Compute log(median baseline) if applicable (adding 1e-10 to avoid log(0))
        median_baseline = np.log(np.array(only_baseline_df.iloc[:, 3:].median().values, dtype=np.float64) + 1e-10) if use_median_baseline else None

        # Apply log-based transformation
        for df_part in [only_post_one_df, only_post_two_df]:

            for index, row in df_part.iterrows():
                patient_id = row['id']
                drug = row['drug']

                # Find corresponding baseline session
                baseline_session = only_baseline_df[(only_baseline_df['drug'] == drug) & (only_baseline_df['id'] == patient_id)]

                if not baseline_session.empty:
                    baseline_values = np.array(baseline_session.iloc[0, 3:], dtype=np.float64)  # Convert explicitly to array
                    post_values = np.array(row.iloc[3:], dtype=np.float64)  # Convert explicitly to array
                    
                    log_baseline = np.log(baseline_values + 1e-10)  # Log transform baseline
                    log_post = np.log(post_values + 1e-10)  # Log transform post values

                    df_part.loc[index, df_part.columns.values[3:]] = log_post - log_baseline  # Log difference
                elif median_baseline is not None:
                    post_values = np.array(row.iloc[3:], dtype=np.float64)  # Convert explicitly to array
                    log_post = np.log(post_values + 1e-10)  # Log transform post values
                    df_part.loc[index, df_part.columns.values[3:]] = log_post - median_baseline
                else:
                    df_part.loc[index, df_part.columns.values[3:]] = np.nan  # If no baseline, set NaN

        # Combine processed data
        df = pd.concat([only_post_one_df, only_post_two_df])

        # Clean up memory
        del only_baseline_df, only_post_one_df, only_post_two_df
                
        df = df.fillna(0)
        #df["time"] = df["time"].astype("category")  # Ensure time is treated as categorical
        df.to_csv(os.path.join(self.feature_output_dir / "lmm_Processed_data.csv"), index=False)

        ylabels = {
            'delta': 'Percentage of the total signal\'s power spectral density',
            'theta': 'Percentage of the total signal\'s power spectral density',
            'alpha': 'Percentage of the total signal\'s power spectral density',
            'ratio': 'Alpha/Delta Ratio',
        }

        print("Number of data points: {}".format(len(df)))


        # ---- Compute Patient Variability ----
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        feature_variability = numeric_df.groupby(df["id"]).std()
        top_patients = feature_variability.idxmax()

        print("Patients with the highest variability per feature:")
        print(top_patients)

        # ---- Run Visualizations ----
        # normalization(df)  # Normalize the data
        self.general_violin_plots(df, ylabels, "before_outliers")  # Generate general violin plots
        # individual_violin_plots(df)  # Generate violin plots per patient
        # self.outliers_plots(df, ylabels, top_patients)  # Generate outlier patient plots
        # # Step 1: Identify outlier patients
        # outlier_patients = top_patients.unique()  # Get unique patient IDs with highest variability

        # # Step 2: Remove outlier patients from the dataset
        # df_no_outliers = df[~df["id"].isin(outlier_patients)]

        # # Step 3: Re-run general violin plots without outliers
        # self.general_violin_plots(df_no_outliers, ylabels, "After_the_most_extreme_value")

        # df_no_outliers, outlier_stats, outlier_details_df = self.remove_outliers_iqr(df, df.columns)
        # outlier_details_df.to_csv( self.feature_output_dir / "eeg_outliers_detail_df.csv")

        # # Display how many outliers were removed per feature
        # # print("Outliers removed per feature:")
        # # for feature, count in outlier_stats.items():
        #     # print(f"Feature: {feature}, Outliers Removed: {count}")

        # # print(outlier_details_df.head())

        # # Show all removed outliers
        # print("Outliers Details:")
        # print(outlier_details_df.to_string())

        # # Count how many times each patient appears in the outlier list
        # patient_outlier_counts = outlier_details_df['id'].value_counts()
        # print(patient_outlier_counts)

        # self.general_violin_plots(df_no_outliers, ylabels, "After_removing_multiple_values")

        # self.plot_all_outlier_patients(df, outlier_details_df, ylabels, "Deep_Dive_All", use_log_scale=False)
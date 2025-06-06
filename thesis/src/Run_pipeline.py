import Pipeline_preprocess
import Pipeline_feature_extraction
import Pipeline_violin_plots
import Pipeline_backwards
import Reject_outliers
from datetime import datetime
from pathlib import Path
import os
epoch_sizes = [1]
sample_sizes = [50]
epoch_rejection_thresholds = [150e-6]
ica_thresholds = [0.5]
for epoch_size in epoch_sizes:
    for sample_size in sample_sizes:
        for epoch_rejection_threshold in epoch_rejection_thresholds:
            for ica_threshold in ica_thresholds:
                root_path = Path("L:\\LovbeskyttetMapper\\CONNECT-ME\\CONMED3\\Dataoptagelser\\NIRS-EEG\\")
                base_output_dir = Path(f"L:/LovbeskyttetMapper/CONNECT-ME/Ioannis/thesis_code/results/run_{datetime.now().strftime('%Y%m%d_%H%M')}")
                feature_output_dir = base_output_dir / "feature_extraction_files"
                plot_output_dir = base_output_dir / "plots"
                excluded_dir = base_output_dir / "skipped_files"
                preprocessed_files_dir = base_output_dir / "preprocessed_files"
                os.makedirs(feature_output_dir, exist_ok=True)
                os.makedirs(plot_output_dir, exist_ok=True)
                os.makedirs(excluded_dir, exist_ok=True)
                os.makedirs(preprocessed_files_dir, exist_ok=True)

                outlier_csv_path = feature_output_dir / "eeg_outliers_detail_df.csv"
                rejected_summary = feature_output_dir / "eeg_outliers_rejected.csv"
                output_dir = plot_output_dir / "EEG_Outlier_Segments"
                os.makedirs(output_dir, exist_ok=True)


                preprocess = Pipeline_preprocess.EEGPreprocessor(excluded_dir, epoch_size, sample_size, epoch_rejection_threshold, ica_threshold, preprocessed_files_dir)
                preprocess.process()

                features = Pipeline_feature_extraction.EEGFeatureExtractor()
                features.extract_features(feature_output_dir) # feature_output_dir

                plots = Pipeline_violin_plots.EEGVisualizer(feature_output_dir, plot_output_dir) # feature_output_dir, plot_output_dir
                plots.process()

                backwards_pipeline = Pipeline_backwards.EEGOutlierAnalyzer(root_path, outlier_csv_path, output_dir)
                backwards_pipeline.process_outliers()

                # reject_outlier = Reject_outliers.EEGEpochQualityChecker(root_path, outlier_csv_path, rejected_summary)
                # reject_outlier.process()

                # features = Pipeline_feature_extraction.EEGFeatureExtractor()
                # features.extract_features(feature_output_dir) # feature_output_dir
                
                # plots = Pipeline_violin_plots.EEGVisualizer(feature_output_dir, plot_output_dir) # feature_output_dir, plot_output_dir
                # plots.process()

                # backwards_pipeline = Pipeline_backwards.EEGOutlierAnalyzer(root_path, outlier_csv_path, output_dir)
                # backwards_pipeline.process_outliers()

                # # Define the base directory where .fif files are located
                # base_directory = Path(r"L:\\LovbeskyttetMapper\\CONNECT-ME\\CONMED3\Dataoptagelser\\NIRS-EEG")

                # # Find all .fif files in the directory and subdirectories
                # fif_files = list(base_directory.rglob("*.fif"))

                # # Delete each .fif file
                # for fif_file in fif_files:
                #     try:
                #         fif_file.unlink()  # Remove the file
                #         print(f"🗑️ Deleted: {fif_file}")
                #     except Exception as e:
                #         print(f"❌ Failed to delete {fif_file}: {e}")

                # print(f"✅ Removed {len(fif_files)} .fif files from {base_directory}")
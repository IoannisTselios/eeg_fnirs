from datetime import datetime
from pathlib import Path
import os
import preprocessing_edf
import feature_extraction
import Violin_plots
import Backwards_check

sample_sizes = [50]
ica_thresholds = [0.75]
for sample_size in sample_sizes:
    for ica_threshold in ica_thresholds:
        root_path = Path("L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\thesis\\src_moshgan\\mapped_5mv_5fv_3resting")
        base_output_dir = Path(f"L:/LovbeskyttetMapper/CONNECT-ME/Ioannis/thesis_code/results_moshgan/run_{datetime.now().strftime('%Y%m%d_%H%M')}")
        preprocessed_output_dir = base_output_dir / "Preprocessd_files"
        SNR_dir = base_output_dir / "SNR"
        epochs_ready_output_dir = base_output_dir / "Epochs_files"
        plot_output_dir = base_output_dir / "plots"
        excluded_dir = base_output_dir / "skipped_files"
        feature_output_dir = base_output_dir / "feature_extraction_files"
        os.makedirs(preprocessed_output_dir, exist_ok=True)
        os.makedirs(SNR_dir, exist_ok=True)
        os.makedirs(excluded_dir, exist_ok=True)
        os.makedirs(epochs_ready_output_dir, exist_ok=True)
        os.makedirs(feature_output_dir, exist_ok=True)
        os.makedirs(plot_output_dir, exist_ok=True)
        outlier_csv_path = feature_output_dir / "eeg_outliers_detail_df.csv"
        # rejected_summary = feature_output_dir / "eeg_outliers_rejected.csv"
        SNR_LOG_PATH = SNR_dir / "snr_log.csv"
        output_dir = plot_output_dir / "EEG_Outlier_Segments"
        os.makedirs(output_dir, exist_ok=True)

        # preprocess = preprocessing_edf.EEGPreprocessor(
        #                                 excluded_dir,
        #                                 1.0,
        #                                 sample_size,
        #                                 150,
        #                                 ica_threshold,
        #                                 root_path,
        #                                 preprocessed_output_dir,
        #                                 SNR_LOG_PATH,
        #                                 epochs_ready_output_dir,
        #                             )
        # preprocess.process()

        features = feature_extraction.EEGFeatureExtractor(Path("L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\results_moshgan\\run_20250604_2104\\Epochs_files")) # epochs_ready_output_dir Path("L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\results_moshgan\\run_20250425_0538\\Epochs_files")
        features.extract_features(feature_output_dir) # feature_output_dir

        plots = Violin_plots.EEGVisualizer(feature_output_dir, plot_output_dir) # feature_output_dir, plot_output_dir
        plots.process()
        # Path("L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\results_moshgan\\run_20250425_0538\\Epochs_files")

        backwards_pipeline = Backwards_check.EEGOutlierAnalyzer(epochs_ready_output_dir, outlier_csv_path, output_dir) # epochs_ready_output_dir, outlier_csv_path, output_dir Path("L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\results_moshgan\\run_20250425_0538\\Epochs_files")
        backwards_pipeline.process_outliers()
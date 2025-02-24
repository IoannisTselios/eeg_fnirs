import Pipeline_preprocess
import Pipeline_feature_extraction
import Pipeline_violin_plots
import Pipeline_backwards
from datetime import datetime
from pathlib import Path
import os

root_path = Path("L:\\LovbeskyttetMapper\\CONNECT-ME\\CONMED3\\Dataoptagelser\\NIRS-EEG\\")
base_output_dir = Path(f"L:/LovbeskyttetMapper/CONNECT-ME/Ioannis/thesis_code/results/run_{datetime.now().strftime('%Y%m%d_%H%M')}")
feature_output_dir = base_output_dir / "feature_extraction_files"
plot_output_dir = base_output_dir / "plots"
excluded_dir = base_output_dir / "skipped_files"
os.makedirs(feature_output_dir, exist_ok=True)
os.makedirs(plot_output_dir, exist_ok=True)
os.makedirs(excluded_dir, exist_ok=True)

outlier_csv_path = feature_output_dir / "eeg_outliers_detail_df.csv"
output_dir = plot_output_dir / "EEG_Outlier_Segments"
os.makedirs(output_dir, exist_ok=True)


preprocess = Pipeline_preprocess.EEGPreprocessor(excluded_dir)
preprocess.process()

features = Pipeline_feature_extraction.EEGFeatureExtractor()
features.extract_features(feature_output_dir)

plots = Pipeline_violin_plots.EEGVisualizer(feature_output_dir, plot_output_dir)
plots.process()

backwards_pipeline = Pipeline_backwards.EEGOutlierAnalyzer(root_path, outlier_csv_path, output_dir)
backwards_pipeline.process_outliers()


from datetime import datetime
from pathlib import Path
import os
import preprocessing_edf
sample_sizes = [50, 100, 150]
ica_thresholds = [0.25, 0.5, 0.75]
for sample_size in sample_sizes:
    for ica_threshold in ica_thresholds:
        root_path = Path("L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\thesis\\src_moshgan\\mapped_5mv_5fv_3resting")
        base_output_dir = Path(f"L:/LovbeskyttetMapper/CONNECT-ME/Ioannis/thesis_code/results_moshgan/run_{datetime.now().strftime('%Y%m%d_%H%M')}")
        preprocessed_output_dir = base_output_dir / "Preprocessd_files"
        SNR_dir = base_output_dir / "SNR"
        # plot_output_dir = base_output_dir / "plots"
        # excluded_dir = base_output_dir / "skipped_files"
        os.makedirs(preprocessed_output_dir, exist_ok=True)
        os.makedirs(SNR_dir, exist_ok=True)
        # os.makedirs(excluded_dir, exist_ok=True)
        SNR_LOG_PATH = SNR_dir / "snr_log.csv"

        preprocess = preprocessing_edf.EEGPreprocessor(
                                        None,
                                        1.0,
                                        sample_size,
                                        150,
                                        ica_threshold,
                                        root_path,
                                        preprocessed_output_dir,
                                        SNR_LOG_PATH
                                    )
        preprocess.process()


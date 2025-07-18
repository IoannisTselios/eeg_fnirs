import os
import glob
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import mne
from fooof import FOOOFGroup
from nice import Markers
from nice.markers import (
    PowerSpectralDensity,
    KolmogorovComplexity,
    PermutationEntropy,
    SymbolicMutualInformation,
    PowerSpectralDensitySummary,
    PowerSpectralDensityEstimator
)
from joblib import Parallel, delayed

class EEGFeatureExtractor:
    def __init__(self, SOURCE_FOLDER):
        self.SOURCE_FOLDER = SOURCE_FOLDER
        logging.info("EEG Feature Extractor Initialized")

    def compute_foof_features(self, epochs_batch):
        psd = epochs_batch.compute_psd()
        psds, freqs = psd.get_data(return_freqs=True)

        valid_idx = freqs > 0
        freqs = freqs[valid_idx]
        psds = psds[:, :, valid_idx]

        slopes_mean = []
        slopes_std = []

        for i in range(psds.shape[0]):
            epoch_psds = psds[i]
            fm = FOOOFGroup(peak_width_limits=[2.0, 12.0], verbose=False)
            fm.set_check_data_mode(False)
            fm.fit(freqs, epoch_psds)
            slopes = -fm.get_params('aperiodic_params', 'exponent')
            slopes_mean.append(np.nanmean(slopes))
            slopes_std.append(np.nanstd(slopes))

        return slopes_mean, slopes_std

    def process_file(self, file_path):
        def entropy(a, axis=0):
            return -np.nansum(a * np.log(a + 1e-12), axis=axis) / np.log(a.shape[axis])

        psds_params = dict(n_fft=4096, n_overlap=100, n_jobs='auto', nperseg=250)
        base_psd = PowerSpectralDensityEstimator(
            psd_method='welch', tmin=None, tmax=0.6, fmin=1., fmax=45.,
            psd_params=psds_params, comment='default')

        marker_list = [
            PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=4., normalize=False, comment='delta'),
            PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=4., normalize=True, comment='deltan'),
            PowerSpectralDensity(estimator=base_psd, fmin=4., fmax=8., normalize=False, comment='theta'),
            PowerSpectralDensity(estimator=base_psd, fmin=4., fmax=8., normalize=True, comment='thetan'),
            PowerSpectralDensity(estimator=base_psd, fmin=8., fmax=12., normalize=False, comment='alpha'),
            PowerSpectralDensity(estimator=base_psd, fmin=8., fmax=12., normalize=True, comment='alphan'),
            PowerSpectralDensity(estimator=base_psd, fmin=12., fmax=30., normalize=False, comment='beta'),
            PowerSpectralDensity(estimator=base_psd, fmin=12., fmax=30., normalize=True, comment='betan'),
            PowerSpectralDensity(estimator=base_psd, fmin=30., fmax=45., normalize=False, comment='gamma'),
            PowerSpectralDensity(estimator=base_psd, fmin=30., fmax=45., normalize=True, comment='gamman'),
            PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=45., normalize=True, comment='summary_se'),
            PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=45., percentile=0.5, comment='summary_msf'),
            PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=45., percentile=0.9, comment='summary_sef90'),
            PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=45., percentile=0.95, comment='summary_sef95'),
            PermutationEntropy(tmin=None, tmax=0.6, backend='python'),
            SymbolicMutualInformation(tmin=None, tmax=0.6, method='weighted', backend='python',
                                      method_params={'nthreads': 'auto'}, comment='weighted'),
            KolmogorovComplexity(tmin=None, tmax=0.6, backend='python', method_params={'nthreads': 'auto'}),
        ]

        try:
            epochs = mne.read_epochs(file_path, preload=True)
            ch_indices = mne.pick_types(epochs.info, eeg=True, exclude='bads')
            events, event_ids = epochs.events, epochs.event_id
            event_id_to_name = {v: k for k, v in event_ids.items()}
            file_name = Path(file_path).stem
        except Exception as e:
            logging.warning(f"❌ Failed to process {file_path}: {e}")
            return None

        try:
            marker_calc = Markers(marker_list)
            marker_calc.fit(epochs)
            reduced = marker_calc.reduce_to_epochs(marker_params={
                'PowerSpectralDensity': [
                    {'axis': 'epochs', 'function': np.mean},
                    {'axis': 'channels', 'function': np.mean},
                    {'axis': 'frequency', 'function': np.mean}],
                'PowerSpectralDensity/summary_se': [
                    {'axis': 'frequency', 'function': entropy},
                    {'axis': 'epochs', 'function': np.mean},
                    {'axis': 'channels', 'function': np.mean}],
                'PowerSpectralDensitySummary': [
                    {'axis': 'epochs', 'function': np.mean},
                    {'axis': 'channels', 'function': np.mean}],
                'PermutationEntropy': [
                    {'axis': 'epochs', 'function': np.mean},
                    {'axis': 'channels', 'function': np.mean}],
                'SymbolicMutualInformation': [
                    {'axis': 'epochs', 'function': np.mean},
                    {'axis': 'channels_y', 'function': np.median},
                    {'axis': 'channels', 'function': np.mean}],
                'KolmogorovComplexity': [
                    {'axis': 'epochs', 'function': np.mean},
                    {'axis': 'channels', 'function': np.mean}],
            }, as_dict=True)
        except Exception as e:
            logging.warning(f"❌ Marker calc failed: {e}")
            return None

        try:
            slope_mean, slope_std = self.compute_foof_features(epochs)
        except Exception as e:
            logging.warning(f"FOOOF failed: {e}")
            slope_mean = [np.nan] * len(epochs)
            slope_std = [np.nan] * len(epochs)

        rows = []
        for i in range(len(epochs)):
            event_code = events[i, 2]
            event_name = event_id_to_name.get(event_code, '').lower()
            label = 'resting' if 'rest' in event_name else 'stimulus'

            row = {
                "id": file_name,
                "type": label,
                "epoch": i,
                "foof_slope_mean": slope_mean[i],
                "foof_slope_std": slope_std[i]
            }

            for key, val in reduced.items():
                try:
                    row[key] = float(val[i]) if isinstance(val, (np.ndarray, list)) else float(val)
                except Exception:
                    row[key] = np.nan

            rows.append(row)

        return pd.DataFrame(rows)

    def extract_features_parallel(self, feature_output_dir, n_jobs=-1):
        mne.set_log_level("WARNING")
        logging.basicConfig(force=True, format='%(levelname)s - %(name)s - %(message)s', level=logging.INFO)

        fif_files = glob.glob(os.path.join(self.SOURCE_FOLDER, "*.fif"))
        if not fif_files:
            logging.error("❌ No FIF files found.")
            return

        Path(feature_output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(feature_output_dir) / "eeg_features.csv"

        logging.info(f"🚀 Running in parallel with {n_jobs if n_jobs > 0 else 'all available'} jobs")
        dfs = Parallel(n_jobs=n_jobs)(
            delayed(self.process_file)(f) for f in fif_files
        )

        dfs = [df for df in dfs if df is not None]
        df_all = pd.concat(dfs, ignore_index=True)
        df_all.to_csv(output_path, index=False)
        logging.info(f"✅ Feature extraction complete. Output saved to {output_path}")

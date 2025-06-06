import mne
import numpy as np
from mne.preprocessing import ICA
from scipy.signal import welch
import logging

# === LOGGING CONFIGURATION ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

# === CONFIGURATION ===
fif_path = "L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\results\\run_20250604_2124\\preprocessed_files\\Patient ID 1 - U1 (UWS)_Session 1_Baseline_preprocessed_raw.fif"
fmin, fmax = 1, 50
band_step = 2
# L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\results\\run_20250515_1827_exclude_files_with_2\\preprocessed_files\\Patient ID 1 - U1 (UWS)_Session 1_Baseline_preprocessed_raw.fif
# L:\\LovbeskyttetMapper\\CONNECT-ME\\Ioannis\\thesis_code\\results_moshgan\\run_20250604_2017\\Preprocessd_files\\02IT-EDF+_mapped_preprocessed_raw.fif
# === LOAD DATA ===
log.info(f"🔍 Loading data from {fif_path}")
raw = mne.io.read_raw_fif(fif_path, preload=True)
print(raw.annotations)
raw.plot(block=True)
raw.pick_types(eeg=True)
sfreq = raw.info['sfreq']
log.info(f"✅ Data loaded. EEG channel count: {len(raw.info['ch_names'])}")

# === GLOBAL PSD ===
log.info("📊 Plotting global PSD (Power Spectral Density)")
raw.plot_psd(fmin=fmin, fmax=fmax)

# === BANDWISE ICA LOOP ===
for low in range(fmin, fmax - band_step + 1, band_step):
    high = low + band_step
    log.info(f"\n⚙️ Running ICA on band {low}-{high} Hz")

    # === Bandpass filter for ICA ===
    raw_band = raw.copy().filter(low, high, fir_design='firwin')

    # === Z-score scaling per channel ===
    data = raw_band.get_data()
    data -= data.mean(axis=1, keepdims=True)
    data /= data.std(axis=1, keepdims=True)
    raw_band._data = data

    # === ICA fitting ===
    ica = ICA(n_components=0.9999, random_state=97, max_iter='auto',
              method='infomax', fit_params=dict(extended=True))
    ica.fit(raw_band)
    log.info(f"✅ ICA fitting complete for {low}-{high} Hz with {ica.n_components_} components")

    # === ICLabel classification ===
    try:
        from mne_icalabel import label_components
        labels = label_components(raw_band, ica, method="iclabel")
        log.info(f"🔍 ICLabel results for band {low}-{high} Hz:")

        # === Calculate peak frequencies of ICA components ===
        ica_sources = ica.get_sources(raw_band).get_data()
        sfreq = raw_band.info['sfreq']

        for idx, label in enumerate(labels['labels']):
            freqs_ic, psd_ic = welch(ica_sources[idx], fs=sfreq, nperseg=2048)
            peak_freq = freqs_ic[np.argmax(psd_ic)]
            log.info(f"    Component {idx}: {label} (peak freq: {peak_freq:.2f} Hz)")

    except ImportError:
        log.warning("⚠️ ICLabel not installed.")
    except Exception as e:
        log.error(f"❌ ICLabel failed for band {low}-{high} Hz: {e}")


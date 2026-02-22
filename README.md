# Speech Recognition

This project implements speech recognition components based on acoustic signal processing, MFCC features, and HMM-GMM modeling.

---

## Notebooks / Scripts

### 1. Soundwave-Visualization
- **Purpose**: Visualize the waveform (amplitude vs. time) of a given audio file.
- **Packages**: `wave`, `numpy`, `matplotlib.pyplot`
- **Variables**: `wav_file` (input audio path), `out_plot` (output plot path)
- **Steps**: Read audio attributes (sampling frequency, amplitude, channels, samples), extract waveform, visualize, and export the plot.

---

### 2. Fourier_transition
- **Purpose**: Apply Fourier transformation to a frame at a target time in the waveform.
- **Packages**: `wave`, `numpy`, `matplotlib.pyplot`
- **Variables**: `wav_file`, `out_plot`, `target_time`, `fft_size`
- **Steps**: Read waveform, extract frame at `target_time`, compute spectrum (FFT), plot waveform and log-absolute spectrum.

---

### 3. Understanding+Cepstrum
- **Purpose**: Extract formant elements from audio via cepstrum analysis. Articulated phones are distinguished using low-frequency information.
- **Packages**: `wave`, `numpy`, `matplotlib`
- **Variables**: `wav_path`, `target_time`, `fft_size`, `cep_threshold`
- **Steps**:
  - Open wave file
  - FFT → spectrum → log-power
  - IFFT → cepstrum → extract `cepstrum_low` and `cepstrum_high`
  - FFT of each → log-power spectra, visualize

---

### 4. Short_Time_Fourier_transformation
- **Purpose**: Short-time Fourier transformation (placeholder; content to be added).

---

### 5. Compute+MFCC
- **Purpose**: Extract Mel-frequency cepstral coefficients (MFCC) from audio using a `FeatureExtractor` class.
- **Packages**: `wave`, `numpy`, `pandas`, `os`
- **Class `FeatureExtractor`**:
  - `Herz2Mel` – convert Hz to Mel
  - `MakeMelFilterBank` – mel filter bank from low/high frequency range
  - `ExtractWindow` – window extraction, dithering, pre-emphasis, DC removal, Hamming
  - `ComputeFBANK` – Mel filter bank features
  - `MakeDCTMatrix` – DCT matrix
  - `MakeLifter` – liftering for cepstrum
  - `ComputeMFCC` – MFCC features

---

### 6. Compute+Mean_Std
- **Purpose**: Compute mean and standard deviation of feature datasets (e.g. fbank, mfcc) for normalization.
- **Packages**: `wave`, `numpy`
- **Variables**: `feat_scp` (feature file list), `out_dir` (output directory)
- **Steps**: Read feature files, compute mean and variance, save to `mean_std.txt`.

---

### 7. Create+Label
- **Purpose**: Convert string labels to integer indices and optionally insert silence.
- **Packages**: `numpy`
- **Function** `phone_to_index`: maps string labels to indices; outputs `label_int` and `phone_out_file`.
- **Variables**: `phone_file`, `label_str`, `out_dir`, `phone_silence`, `insert_sil`

---

### 8. Create+Proto
- **Purpose**: Create the initial HMM prototype.
- **Packages**: `numpy`, `os`
- **Variables**: `phones_file`, `num_states`, `num_dims`, `prob_loop`, `out_dir`
- **Steps**: Build phone list, create `MonoPhone` instance, call `make_proto`, save prototype.

---

### 9. Class+MonoPhoneHMM
- **Purpose**: Single-Gaussian mono-phone HMM class for training and recognition.
- **Packages**: `numpy`
- **Class `MonoPhoneHMM`**:
  - `make_proto` – create HMM prototype
  - `calc_gconst`, `calc_pdf`, `calc_out_prob` – emission probabilities
  - `calc_alpha`, `calc_beta` – forward/backward
  - `flat_init` – initialize GMM parameters
  - `reset_accumulators`, `update_accumulators`, `update_parameters` – training
  - `viterbi_decoding`, `back_track` – decoding
  - `save_hmm`, `load_hmm` – I/O
  - `recognize` – word recognition over a lexicon

---

### 10. Estimate+Alignment
- **Purpose**: Estimate state alignment for each utterance given labels.
- **Packages**: `numpy`
- **Variables**: `hmm_file`, `feat_scp`, `label_file`, `align_file`
- **Steps**: Load HMM, read features and labels, run `state_alignment`, save alignment to file.

---

### 11. Estimate+Alignment+Function
- **Purpose**: Implementation of `state_alignment`: Viterbi decoding and back-tracking to phone×state index per frame.

---

### 12. Count+States
- **Purpose**: Count occurrences of each phone×state from alignment data.
- **Packages**: `numpy`
- **Variables**: `hmm_file`, `align_file`, `count_file`
- **Steps**: Load HMM, read alignment file, count each state, save counts.

---

### 13. Train+SGM+HMM
- **Purpose**: Train HMM using labels and features.
- **Packages**: `numpy`, `os`
- **Variables**: `base_name`, `feat_scp`, `label_file`, `out_dir`, `num_iters`, `num_utterance`
- **Steps**: Load HMM prototype, read labels and features, run training loop, save HMM per iteration.

---

### 14. Dp-matching
- **Purpose**: Dynamic time warping between two feature sequences (e.g. MFCC) for alignment.
- **Packages**: `numpy`
- **Function** `dp_matching(feature_1, feature_2)` → `(total_cost, min_path)`
- **Variables**: `mfcc_file_1`, `mfcc_file_2`, `result` (alignment output path)

---

### 15. Recognize+SGM+HMM
- **Purpose**: Recognize words from features using HMM and a lexicon.
- **Variables**: `hmm_file`, `feat_scp`, `lexicon_file`, `phones_file`, `insert_sil`
- **Steps**: Load phones and lexicon, load HMM, read features, run recognition per utterance, write results.

---

## Workflow

1. **Feature extraction**: `Soundwave-Visualization` → `Fourier_transition` → `Understanding+Cepstrum` → `Short_Time_Fourier_transformation` → `Compute+MFCC`
2. **Setup**: `Compute+Mean_Std`, `Create+Label`, `Create+Proto`
3. **Training**: `Train+SGM+HMM` (optionally after alignment via `Estimate+Alignment` and `Count+States`)
4. **Recognition**: `Recognize+SGM+HMM`
5. **Alignment / DTW**: `Dp-matching`, `Estimate+Alignment`, `Estimate+Alignment+Function`

---

## Dependencies

- Python 3
- `numpy`
- `wave` (stdlib)
- `matplotlib`
- `pandas` (for Compute+MFCC)
- `os`, `sys`, `json` (stdlib)

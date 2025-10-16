# Audio Feature Extraction Toolkit

Comprehensive toolkit for audio classification: generates spectrograms, MFCCs with delta/delta-delta features, and tabular datasets.

## Installation

```bash
pip install torch torchaudio matplotlib numpy pandas scipy
```

## Features

- **Visual Features**: Mel-spectrograms, MFCCs, deltas (Δ), and delta-deltas (ΔΔ)
- **Tabular Features**: Statistical and temporal features extracted from audio for ML models
- **Batch Processing**: Process entire classified datasets automatically
- **Flexible Output**: Save as images (PNG) or tensors (.pt)

## Usage

### 1. Generate Spectrograms & MFCCs with Deltas

```python
from spectrogram_generator import process_dataset

process_dataset(
    input_dir="./audio_dataset",
    output_dir="./processed_data",
    duration=10,                     # Duration in seconds (None = full)
    save_format="image",             # "image", "tensor", or "both"
    include_deltas=True,             # Generate Δ and ΔΔ features
    delta_on_spectrogram=False       # Apply deltas to spectrograms too
)
```

### 2. Create Tabular Dataset (CSV)

```python
from tabular_extractor import create_tabular_dataset

df = create_tabular_dataset(
    input_dir="./audio_dataset",
    output_csv="features.csv",
    duration=10,
    n_mfcc=20
)
```

## Input Structure

```
audio_dataset/
├── dog/
│   ├── bark1.wav
│   └── bark2.wav
└── cat/
    ├── meow1.wav
    └── meow2.wav
```

## Output Structure

**Visual features:**
```
processed_data/
├── dog_spectro/          # Mel-spectrograms
├── dog_mfcc/             # MFCC coefficients
├── dog_delta/            # Δ (first derivative)
├── dog_delta_delta/      # ΔΔ (second derivative)
├── dog_mfcc_combined/    # MFCC + Δ + ΔΔ stacked
└── ...
```

**Tabular features:**
```
features.csv with columns:
- class
- mfcc_mean, mfcc_std, ...
- delta_mean, delta_std, ...
- delta_delta_mean, delta_delta_std, ...
- spectral_centroid_*, energy_*, zcr, etc.
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_rate` | 16000 | Target sampling rate (Hz) |
| `n_mels` | 64 | Number of mel frequency bands |
| `n_mfcc` | 20 | Number of MFCC coefficients |
| `duration` | None | Fixed audio duration (None = full length) |
| `save_format` | "image" | "image", "tensor", or "both" |
| `include_deltas` | True | Generate Δ and ΔΔ features |
| `delta_on_spectrogram` | False | Apply deltas to spectrograms |

## What are Deltas?

- **Delta (Δ)**: First-order derivative capturing rate of change over time
- **Delta-Delta (ΔΔ)**: Second-order derivative capturing acceleration of change
- Improve classification by capturing temporal dynamics of audio features

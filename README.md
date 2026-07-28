# cesm1-extreme-weather-dl

Deep learning for extreme weather event classification using CESM1 Large Ensemble data.

## Overview

This project implements and compares multiple deep learning architectures for classifying
extreme weather events using climate model output from the CESM1 Large Ensemble (LENS) dataset.

## Project Structure

```
cesm1-extreme-weather-dl/
├── config.py                    # Centralized paths and constants
├── 01_preprocess.py             # CESM1 raw data -> labeled NetCDF
├── 02_train_logistic.py         # Train LogisticRegression baseline
├── 03_train_cnn.py              # Train CNN
├── 04_train_resnet.py           # Train ResNet (+ coordinate attention)
├── 05_train_capsule.py          # Train Capsule Network
├── 06_train_transformer.py      # Train Transformer
├── 07_drawfig.py                # Generate all result figures
├── data/
│   ├── README.md                # Data documentation and download links
│   ├── labeled/                 # Preprocessed NetCDF label files
│   └── land_sea_mask/           # Land/sea mask files
├── requirements.txt
└── .gitignore
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Data

Download CESM1 LENS data (see [data/README.md](data/README.md) for download links) and
configure the path:

```bash
export RAW_DATA_DIR=/path/to/your/cesm1/data
```

### 3. Preprocess

```bash
python 01_preprocess.py
```

### 4. Train Models

```bash
# Train individual models
python 02_train_logistic.py     # LogisticRegression baseline
python 03_train_cnn.py          # CNN
python 04_train_resnet.py       # ResNet
python 05_train_capsule.py      # Capsule Network
python 06_train_transformer.py  # Transformer
```

### 5. Generate Figures

```bash
python 07_drawfig.py
```

## Models

| Model | Description |
|-------|-------------|
| LogisticRegression | Baseline classifier |
| CNN | Convolutional Neural Network |
| ResNet | Residual Network with coordinate attention |
| Capsule | Capsule Network |
| Transformer | Vision Transformer |

## Configuration

All paths and hyperparameters are centralized in `config.py`. Key parameters include:

- `RAW_DATA_DIR` -- Path to raw CESM1 LENS data
- `LABELED_DATA_DIR` -- Path to preprocessed labeled NetCDF files
- `N_MEMBERS` -- Number of ensemble members (42)
- `ROLLING_MEAN_WINDOW` -- Rolling mean window size (15 days)
- `PERCENTILE_THRESHOLD` -- Extreme event percentile (99th)

## License

This project is released under the MIT License.

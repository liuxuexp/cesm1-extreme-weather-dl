# Data Documentation

## Overview

This project uses the **CESM1 Community Earth System Model Large Ensemble (LENS)** dataset
to train deep learning models for extreme weather event classification.

## Data Source

CESM1 LENS is a 42-member ensemble of historical climate simulations (1920--2005)
using the CESM1-CAM5 model. Each member uses identical radiative forcing but different
initial conditions, enabling the study of internal climate variability.

### Download URLs

| Source | URL |
|--------|-----|
| NCAR CESM1 LENS Project Page | https://www.cesm.ucar.edu/projects/community-projects/LENS/ |
| NCAR Earth System Data Portal | https://www.earthsystemcog.org/projects/lens/ |
| Amazon S3 (AWS) | https://registry.opendata.aws/ncar-cesm1-lens/ |

### Variables Used

| Variable | Description | Frequency | Level |
|----------|-------------|-----------|-------|
| TREFHT | Near-surface air temperature | Daily | 2 m |
| Z500 | 500 hPa geopotential height | Daily | 500 hPa |

### Preprocessing Steps

1. **Merge 42 ensemble members**: Combine all members into a single NetCDF file
2. **Spatial domain selection**: Crop to East Asia (70--135E, 10--55N for z500; 70--135E, 20--55N for t2m)
3. **Rolling mean anomaly**: Compute 15-day rolling mean and anomaly from climatological mean
4. **Extreme event labeling**: Identify extreme events exceeding 99th percentile threshold
5. **Labeled data generation**: Create labeled NetCDF files for summer and winter separately

### Output Files

The preprocessing step (`01_preprocess.py`) generates the following files:

- `summer-z500-labled-c5-7.nc` -- Summer z500 labeled data
- `summer-t2m-t2man-labled-c5-7.nc` -- Summer t2m labeled data
- `winter-z500-labled-c5-7.nc` -- Winter z500 labeled data
- `winter-t2m-t2man-labled-c5-7.nc` -- Winter t2m labeled data

These files are excluded from version control (see `.gitignore`).

## Land/Sea Mask

Land/sea mask files at various resolutions are provided in the `land_sea_mask/` directory:

- `Land-sea-mask-0.75.nc` -- Default mask (0.75 degree resolution)
- Other masks at 0.125--3.0 degree resolutions

These masks are used to distinguish land and sea grid points during feature engineering.

## Configuration

Paths can be configured via environment variables:

```bash
export RAW_DATA_DIR=/path/to/raw/cesm1/data
export LAND_SEA_MASK=/path/to/land-sea-mask.nc
```

Or by modifying the paths in `config.py`.

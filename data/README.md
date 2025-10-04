# Kepler KOI Dataset

## Overview
This directory contains the Kepler Objects of Interest (KOI) dataset from the NASA Exoplanet Archive, used for training the exoplanet detection model.

## Dataset Source
- **Source**: NASA Exoplanet Archive
- **Table**: `cumulative` (Kepler KOI cumulative table)
- **URL**: https://exoplanetarchive.ipac.caltech.edu/
- **Citation**: DOI: http://doi.org/10.17616/R3X31K

## Downloading the Dataset

### Option 1: Automatic Download (Recommended)
Use the provided download script to fetch the latest data:

```bash
python download_dataset.py
```

This will download the complete dataset with all 153 columns.

### Option 2: Manual Download
Visit the NASA Exoplanet Archive website:
1. Go to https://exoplanetarchive.ipac.caltech.edu/
2. Navigate to the Kepler KOI table
3. Select columns and download as CSV

## Dataset Comparison: Automatic vs Manual Download

### Automatic Download (via TAP API)
- **Method**: HTTP API query using Table Access Protocol (TAP)
- **Query**: `SELECT * FROM cumulative`
- **Columns**: 153 (all available columns)
- **Format**: Clean CSV, no comments
- **Advantages**:
  - Complete dataset with all metadata
  - Programmatic and reproducible
  - Easy to update with latest data
  - No manual column selection needed

### Manual Download (via Web Interface)
- **Method**: Interactive web interface
- **Query**: User-selected columns
- **Columns**: Typically 49 core columns (varies by user selection)
- **Format**: CSV with comment lines (starting with `#`)
- **Advantages**:
  - User can preview data
  - Selective column download
  - No coding required

### Key Difference
The automatic download uses `cumulative` table with **ALL 153 columns**, while manual downloads typically have **49 user-selected columns**. The extra 104 columns in the automatic download include:
- Additional stellar parameters
- Extended photometry bands
- Model fitting details
- Vetting flags and metadata
- Centroid offset measurements

For machine learning, the automatic download is preferred as it provides more features for the model to learn from (after removing data leakage columns).

## Dataset Structure

### Dimensions
- **Rows**: 9,564 KOI objects
- **Columns**: 153 features

### Key Columns
- `koi_disposition`: Classification (CONFIRMED, CANDIDATE, FALSE POSITIVE)
- `koi_period`: Orbital period (days)
- `koi_depth`: Transit depth (ppm)
- `koi_duration`: Transit duration (hours)
- `koi_steff`: Stellar effective temperature (K)
- `koi_srad`: Stellar radius (solar radii)
- `koi_kepmag`: Kepler magnitude

### Class Distribution
- FALSE POSITIVE: 4,839 (50.6%)
- CONFIRMED: 2,746 (28.7%)
- CANDIDATE: 1,979 (20.7%)

For binary classification:
- **Planets** (CONFIRMED + CANDIDATE): 4,725 (49.4%)
- **Non-planets** (FALSE POSITIVE): 4,839 (50.6%)

## Data Processing Pipeline

1. **Download** (this directory)
   - `download_dataset.py` - Automatic download script
   - `kepler_koi.csv` - Raw dataset (153 columns)

2. **Preprocessing** (`preprocessing/` directory)
   - Remove 63 data leakage columns
   - Handle missing values (median imputation)
   - Standardize features (StandardScaler)
   - Output: 77 clean features + 1 label

3. **Model Training** (`models/` directory)
   - Split data: 70/10/20 train/val/test
   - Train LightGBM classifier
   - Evaluate performance

## Files
- `download_dataset.py` - Script to download dataset from NASA API
- `kepler_koi.csv` - Raw downloaded dataset (153 columns)
- `preprocessing/kepler_koi_preprocessed.csv` - Cleaned dataset (78 columns: 77 features + label)

## Citation
If you use this dataset, please cite:

```
NASA Exoplanet Archive
DOI: 10.17616/R3X31K
URL: https://exoplanetarchive.ipac.caltech.edu/
```

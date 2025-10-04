# 🪐 NASA Exoplanet Detection with AI

**NASA Space Apps Challenge 2025 - A World Away: Hunting for Exoplanets with AI**

An AI-powered system for detecting exoplanets from Kepler Space Telescope transit data using LightGBM machine learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40.2-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🚀 Project Overview

This project automates the detection of exoplanets from transit signals in the Kepler Space Telescope data, reducing manual vetting by astronomers. Using supervised machine learning with LightGBM, we classify Kepler Objects of Interest (KOI) as either confirmed/candidate planets or false positives.

### Key Features
- **High-Performance ML Model**: LightGBM classifier achieving 90.3% F1-score and 97.1% ROC-AUC
- **Data Leakage Prevention**: Rigorous removal of 63 leakage columns for genuine learning
- **Interactive Web Application**: Streamlit-based UI for real-time predictions
- **Comprehensive Data Pipeline**: Automated download, preprocessing, and model training
- **Visualization Dashboard**: Interactive plots for data exploration and model interpretation

## 📊 Model Performance

| Metric | Test Set Score |
|--------|-------|
| **Accuracy** | 90.4% |
| **Precision** | 90.2% |
| **Recall** | 90.5% |
| **F1-Score** | 90.3% |
| **ROC-AUC** | 97.1% |

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/nasa-exoplanet-detection.git
cd nasa-exoplanet-detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
cd src
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
nasa-exoplanet-detection/
│
├── data/                              # Data directory
│   ├── download_dataset.py            # Automatic dataset downloader
│   ├── kepler_koi.csv                 # Raw Kepler KOI dataset (153 columns)
│   ├── README.md                      # Dataset documentation
│   └── preprocessing/                 # Processed data
│       ├── data_preprocessing.py      # Preprocessing pipeline
│       ├── kepler_koi_preprocessed.csv # Clean dataset (77 features + label)
│       ├── preprocessor.pkl           # Saved preprocessor
│       └── preprocessing_summary.txt  # Feature list
│
├── models/                            # Model directory
│   ├── train_model.py                 # LightGBM training script
│   ├── lightgbm_model.pkl             # Trained model
│   └── model_metrics.json             # Performance metrics
│
├── src/                               # Web application
│   └── app.py                         # Streamlit web application
│
├── CLAUDE.md                          # Project specifications
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## 🔬 Dataset

- **Source**: NASA Exoplanet Archive - Kepler Objects of Interest (KOI)
- **Download Method**: Automatic via TAP API (`cumulative` table)
- **Size**: 9,564 transit signals
- **Raw Columns**: 153 features
- **Clean Features**: 77 features (after removing 63 leakage columns)
- **Target**: Binary classification (Planet vs False Positive)
- **Class Distribution**: 49.4% planets, 50.6% non-planets

### Data Leakage Prevention

We carefully removed **63 columns** that could leak classification information:
- **False positive flags** (koi_fpflag_nt, koi_fpflag_ss, koi_fpflag_co, koi_fpflag_ec)
- **Derived planet properties** (koi_prad, koi_teq, koi_insol - computed assuming planet exists)
- **Disposition scores** (koi_score, koi_pdisposition - vetting outputs)
- **Model fitting outputs** (fitted parameters that use disposition)

This ensures the model learns from **genuine observational data only**.

### Feature Categories (77 total)

1. **Transit Signals**: Period, depth, duration, impact parameter
2. **Stellar Properties**: Temperature, radius, mass, surface gravity, metallicity
3. **Photometry**: Magnitudes in multiple bands (g, r, i, z, J, H, K, Kepler)
4. **Signal Statistics**: SNR, single/multiple event statistics
5. **Centroid Offsets**: Flux-weighted and difference image centroids

## 🎯 Usage

### Download Dataset

```bash
cd data
python download_dataset.py
```

This downloads the complete Kepler KOI dataset (153 columns) from NASA Exoplanet Archive.

### Web Application

1. **Launch the app**: Run `streamlit run app.py` from the `src` directory
2. **Choose input method**:
   - **Sample Data**: Test with preprocessed samples
   - **Upload CSV**: Upload preprocessed transit data (77 features required)

3. **View results**:
   - Classification (Planet/Non-Planet)
   - Confidence score
   - Probability gauge visualization
   - Model performance metrics

### Python API

```python
import pickle
import pandas as pd

# Load model and metrics
with open('models/lightgbm_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load preprocessor to get feature names
with open('data/preprocessing/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

feature_names = preprocessor['feature_columns']

# Prepare your data (77 features)
# Must be preprocessed (scaled) data
features = pd.DataFrame(..., columns=feature_names)

# Make prediction
prediction = model.predict(features)
probability = model.predict_proba(features)

print(f"Classification: {'Planet' if prediction[0] == 1 else 'Non-Planet'}")
print(f"Confidence: {probability[0][1]:.1%}")
```

## 🔧 Training the Model

To retrain the model with new data or parameters:

1. **Download data**: `python data/download_dataset.py`
2. **Preprocess**: `python data/preprocessing/data_preprocessing.py`
3. **Train model**: `python models/train_model.py`

The trained model will be saved to `models/lightgbm_model.pkl`

## 📈 Model Architecture

- **Algorithm**: LightGBM (Light Gradient Boosting Machine)
- **Type**: Binary classification
- **Data Split**: 70% train, 10% validation, 20% test (stratified)
- **Preprocessing**: StandardScaler + median imputation
- **Key Hyperparameters**:
  - n_estimators: 200
  - learning_rate: 0.05
  - max_depth: 7
  - num_leaves: 31
  - subsample: 0.8
  - colsample_bytree: 0.8

## 🏆 NASA Space Apps Challenge 2025

This project was developed for the NASA Space Apps Challenge 2025, addressing the challenge:
**"A World Away: Hunting for Exoplanets with AI"**

### Challenge Goals Met
✅ Automated exoplanet detection from transit data
✅ High-accuracy machine learning model (90.3% F1-score, 97.1% ROC-AUC)
✅ Rigorous data leakage prevention for genuine learning
✅ Complete data pipeline (download → preprocess → train → deploy)
✅ Interactive visualization and exploration tools
✅ Open-source implementation with documentation
✅ User-friendly web interface for predictions

## 📚 References

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [Kepler Mission](https://www.nasa.gov/mission_pages/kepler/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- Assessment of Ensemble-Based Machine Learning Algorithms for Exoplanet Identification

### Citation

If you use this work, please cite:
```
NASA Exoplanet Archive (2025)
DOI: http://doi.org/10.17616/R3X31K
```

## 📄 License

This project is open source and available under the MIT License.

## 📧 Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---

**Note**: This project achieves 90.3% F1-score and 97.1% ROC-AUC on clean, leakage-free data. The model learns from genuine observational features only, making it suitable for real-world exoplanet vetting workflows.
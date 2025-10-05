# 🪐 NASA Exoplanet Detection with AI

**NASA Space Apps Challenge 2025 - A World Away: Hunting for Exoplanets with AI**

An AI-powered system for detecting exoplanets from Kepler Space Telescope transit data using LightGBM machine learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40.2-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌐 Live Demo

**Try it now:** [https://spaceapps-exoplanet-detector.streamlit.app/](https://spaceapps-exoplanet-detector.streamlit.app/)

## 🚀 Project Overview

This project automates the detection of exoplanets from transit signals in the Kepler Space Telescope data, reducing manual vetting by astronomers. Using supervised machine learning with LightGBM, we classify Kepler Objects of Interest (KOI) as either confirmed/candidate planets or false positives.

### Key Features
- **🌐 Live Web Application**: Deployed on Streamlit Cloud for instant access
- **🎯 High-Performance ML Models**: Baseline (84% accuracy, 93% ROC-AUC) and Full (87% accuracy, 94% ROC-AUC)
- **🔮 Candidate Predictor**: Analyze unconfirmed KOI candidates with confidence scores
- **🌌 3D Visualizations**: Interactive Three.js solar system views with size comparisons
- **📊 SHAP Explainability**: Understand which features drive each prediction
- **🛡️ Data Leakage Prevention**: Rigorous removal of 88 leakage columns for genuine learning
- **📈 Interactive Dashboard**: Explore data, model performance, and make predictions

## 📊 Model Performance

### Full Model (52 features, optimized)
| Metric | Test Set Score |
|--------|-------|
| **Accuracy** | 87.4% |
| **Precision** | 86.9% |
| **Recall** | 88.0% |
| **F1-Score** | 87.4% |
| **ROC-AUC** | 94.3% |

### Baseline Model (9 core features)
| Metric | Test Set Score |
|--------|-------|
| **Accuracy** | 84.3% |
| **Precision** | 84.4% |
| **Recall** | 83.7% |
| **F1-Score** | 84.1% |
| **ROC-AUC** | 92.7% |

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
- **Clean Features**: 52 features (Full model) / 9 features (Baseline model)
- **Target**: Binary classification (Planet vs False Positive)
- **Class Distribution**: ~50% planets, ~50% non-planets (balanced)

### Data Leakage Prevention

We carefully removed **88 columns** that could leak classification information:
- **False positive flags** (koi_fpflag_nt, koi_fpflag_ss, koi_fpflag_co, koi_fpflag_ec)
- **Derived planet properties** (koi_prad, koi_teq, koi_insol, koi_dor, koi_incl, etc.)
- **Disposition scores** (koi_score, koi_pdisposition - vetting outputs)
- **Centroid offsets** (dikco_*, dicco_*, fwm_* - post-vetting features)
- **Model fitting outputs** (fitted parameters that use disposition)

This ensures the model learns from **genuine observational data only**.

### Feature Categories

**Baseline Model (9 features):**
- Core stellar properties: koi_steff, koi_slogg, koi_srad, koi_kepmag
- Core transit signals: koi_period, koi_depth, koi_duration, koi_impact, koi_model_snr

**Full Model (52 features):**

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

**Live Demo**: [https://spaceapps-exoplanet-detector.streamlit.app/](https://spaceapps-exoplanet-detector.streamlit.app/)

The web application includes 6 main tabs:

1. **🔍 Sample Explorer**: Interactive exploration of labeled KOIs with:
   - 3D solar system visualization (default view)
   - Star/planet size comparisons with Sun/Earth
   - Prediction results with confidence gauges
   - SHAP explainability analysis

2. **🔮 Candidate Predictor**: Analyze unconfirmed candidate KOIs:
   - Batch classification of all candidates
   - Confidence distribution histogram
   - High-confidence planet identification
   - Downloadable results

3. **📂 Import & Predict**: Upload custom CSV data for predictions

4. **📊 Data Explorer**: Explore dataset statistics and distributions

5. **📈 Model Performance**: View detailed performance metrics across train/val/test splits

6. **📚 Documentation**: Complete project documentation and methodology

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
✅ Automated exoplanet detection from transit data\
✅ High-accuracy machine learning models (87% accuracy, 94% ROC-AUC)\
✅ Rigorous data leakage prevention for genuine learning\
✅ Complete data pipeline (download → preprocess → train → deploy)\
✅ Interactive 3D visualizations with size comparisons\
✅ SHAP explainability for model interpretability\
✅ Candidate predictor for unconfirmed KOIs\
✅ Open-source implementation with documentation\
✅ **Deployed web application** on Streamlit Cloud

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

**Live Demo**: [https://spaceapps-exoplanet-detector.streamlit.app/](https://spaceapps-exoplanet-detector.streamlit.app/)

**Note**: This project achieves 87% accuracy and 94% ROC-AUC on clean, leakage-free data. The model learns from genuine observational features only, making it suitable for real-world exoplanet vetting workflows.
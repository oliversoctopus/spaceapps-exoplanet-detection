# 📊 Analysis Notebooks

This directory contains exploratory data analysis (EDA) and feature engineering notebooks for the NASA Exoplanet Detection project.

## Notebooks

### `Exoplanet_Detection_EDA_Feature_Engineering.ipynb`

**Author:** Collaborator  
**Purpose:** Comprehensive EDA and feature engineering for Kepler KOI dataset

**Contents:**
- **Data Exploration**: Initial analysis of the Kepler Objects of Interest (KOI) dataset
- **Feature Selection**: Identification of predictive features while avoiding data leakage
- **Leakage Prevention**: Analysis and removal of derived features that could leak classification information
- **Feature Engineering**: Creation of `is_multi_planet_system` feature with strong discriminative power
- **Data Preprocessing**: Log transformations and scaling for modeling
- **Initial Modeling**: Baseline comparison of Logistic Regression, Random Forest, and SVM

**Key Findings:**
- **11 features selected** from 49 original columns
- **Data leakage prevention**: Removed disposition-related columns (koi_pdisposition, koi_score, fpflags) and koi_teq
- **Multi-planet system feature**: 27.82% of CONFIRMED vs 1.07% of FALSE POSITIVES exist in multi-planet systems (Chi² p < 0.001)
- **Class balance**: ~50/50 split between CANDIDATE and FALSE POSITIVE after preprocessing
- **Initial model performance**: Random Forest achieved 85% accuracy, 92.9% ROC-AUC on test set

**Generated Files:**
- `kepler_modeling_clean.csv` - Cleaned dataset with original feature scales (for tree-based models)
- `kepler_modeling_log.csv` - Log-transformed features (for linear models)

⚠️ **Note:** Generated CSV files are not committed to version control (see `.gitignore`)

## Running the Notebooks

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install jupyter ydata-profiling
   ```

2. **Launch Jupyter:**
   ```bash
   jupyter notebook notebooks/
   ```

3. **Data Requirements:**
   - Ensure `kepler_data.csv` is available in the `data/` directory
   - The notebook expects the raw Kepler KOI dataset (49 columns, 9564 rows)

## Integration with Main Project

These notebooks support the main exoplanet detection pipeline:

- **Data Pipeline**: `data/` directory contains raw and preprocessed data
- **Models**: `models/` directory contains trained LightGBM models
- **Web App**: `src/app.py` - Streamlit application for predictions
- **Analysis**: `notebooks/` (this directory) - Exploratory analysis and feature engineering

The analysis here informed feature selection and preprocessing strategies used in the production model.

## Contributing

When adding new notebooks:
1. Use descriptive names following the pattern: `[Topic]_[Description].ipynb`
2. Include markdown cells documenting your analysis
3. Update this README with a summary of findings
4. Ensure notebooks run end-to-end without errors
5. Avoid committing large generated data files (use `.gitignore`)

---

**Project:** NASA Space Apps Challenge 2025 - Exoplanet Detection  
**Repository:** https://github.com/oliversoctopus/spaceapps-exoplanet-detection


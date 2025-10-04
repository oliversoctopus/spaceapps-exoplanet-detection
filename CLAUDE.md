# CLAUDE.md: NASA Space Apps Challenge - Exoplanet Detection Project

## Project Overview
This project is for the **NASA Space Apps Challenge 2025**, specifically the "A World Away: Hunting for Exoplanets with AI" challenge. The goal is to develop a supervised machine learning model to classify exoplanets vs. false positives using the **Kepler Objects of Interest (KOI)** dataset (`kepler_koi.csv`). The model will predict whether a transit signal is a planet (CANDIDATE/CONFIRMED) or not (FALSE POSITIVE) based on features like orbital period (`koi_period`) and transit depth (`koi_depth`). The chosen model is **LightGBM**, leveraging its speed, handling of imbalanced data (~50% false positives), and your familiarity with gradient boosting. The final deliverable is an open-source GitHub repo with a Streamlit app featuring interactive predictions, 3D visualizations, and explainability features.

## Purpose
The purpose is to automate exoplanet detection, reducing manual vetting by astronomers. The model will:
- Use labeled KOI data for supervised binary classification (planet vs. non-planet).
- Achieve high accuracy (>95% F1-score, per benchmarks) on 81 non-leakage features including transit signals, stellar properties, photometry, and false positive flags.
- Provide an interactive Streamlit demo to upload new data and display predictions with confidence scores and Plotly visualizations of transit patterns.

## Dataset
- **File**: `kepler_koi.csv` (downloaded from NASA Exoplanet Archive, ~9,564 rows, 153 columns).
- **Key Columns**:
  - **Label**: `koi_disposition` (CANDIDATE, CONFIRMED = positive; FALSE POSITIVE = negative). Encoded as binary: 1 for planet, 0 for non-planet.
  - **Features**: Transit signals (`koi_period`, `koi_depth`, `koi_duration`, `koi_impact`), stellar properties (`koi_steff`, `koi_srad`, `koi_slogg`, `koi_smass`), photometry (`koi_kepmag`, `koi_jmag`, `koi_hmag`), signal statistics (`koi_model_snr`), and centroid offsets.
  - **Excluded (Data Leakage)**: `koi_prad`, `koi_teq`, `koi_insol`, `koi_score`, `koi_pdisposition`, false positive flags (`koi_fpflag_nt`, `koi_fpflag_ss`, `koi_fpflag_co`, `koi_fpflag_ec`), and other derived planet properties or vetting outputs.
- **Source**: NASA Exoplanet Archive TAP query (`https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv`).
- **Preprocessing Needs**: Remove data leakage columns, handle missing values (median imputation), standardize numerical features (StandardScaler), encode labels (1 for planet, 0 for non-planet).

## Project Framework
### 1. Data Preprocessing
- **Input**: `data/kepler_koi.csv` (raw, 153 columns).
- **Tasks**:
  - Load with pandas: `pd.read_csv("data/kepler_koi.csv")`.
  - Create binary labels: `df['label'] = ((df['koi_disposition'] == 'CANDIDATE') | (df['koi_disposition'] == 'CONFIRMED')).astype(int)`.
  - **Remove data leakage**: Drop 63 columns including:
    - Direct answers: `koi_disposition`, `koi_pdisposition`, `koi_score`, `kepler_name`
    - Derived planet properties: `koi_prad`, `koi_teq`, `koi_insol`, `koi_dor`, `koi_incl`, `koi_ror`, `koi_sma`, `koi_eccen`, `koi_longp` (+ error columns)
    - False positive flags: `koi_fpflag_nt`, `koi_fpflag_ss`, `koi_fpflag_co`, `koi_fpflag_ec` (derived from vetting process)
    - Metadata: identifiers, coordinates, vetting status, processing info
  - Handle missing values: Median imputation for numerical features, drop columns with 100% missing values.
  - Standardize features: Use `sklearn.preprocessing.StandardScaler` (mean=0, std=1).
  - **Do NOT split data** - save complete preprocessed dataset for model training to handle splitting.
- **Output**: `data/preprocessing/kepler_koi_preprocessed.csv` (9,564 rows × 78 columns: 77 features + 1 label).

### 2. Model Development
- **Model**: LightGBM (`LGBMClassifier`), chosen for speed, handling of imbalanced data, and gradient boosting efficiency.
- **Location**: Store training code in `models/` directory, save trained models as `.pkl` files.
- **Tasks**:
  - Load preprocessed data: `pd.read_csv("data/preprocessing/kepler_koi_preprocessed.csv")`.
  - **Split data**: 70/10/20 train/validation/test using `train_test_split` with stratification and `random_state=42`.
  - Train binary classifier: Positive (CANDIDATE/CONFIRMED) vs. negative (FALSE POSITIVE).
  - Example code:
    ```python
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import train_test_split
    import pickle
    import pandas as pd

    # Load preprocessed data
    df = pd.read_csv("data/preprocessing/kepler_koi_preprocessed.csv")
    X = df.drop('label', axis=1)
    y = df['label']

    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp)

    # Train model
    model = LGBMClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    with open('models/lightgbm_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    ```
  - Tune hyperparameters (e.g., learning_rate, max_depth, num_leaves) via grid search or manual validation.
  - Evaluate: Accuracy, precision, recall, F1, ROC-AUC (prioritize recall for planet detection).
  - Goal: Achieve >95% F1-score, per benchmarks on KOI data.
  - **Data Balance**: Dataset is nearly balanced (49.4% planets, 50.6% non-planets), so no special handling needed.

### 3. Visualization & Deployment
- **Visualization**: Use Plotly for interactive plots (e.g., feature distributions, confusion matrix, ROC curves).
- **Deployment**: Build a Streamlit app in `src/app.py` to:
  - Upload new KOI-like data (CSV or manual input).
  - Load preprocessed data and trained model from `models/`.
  - Output predictions with confidence scores (`model.predict_proba`).
  - Display transit feature plots (e.g., `koi_depth` vs. `koi_period`, colored by prediction).
- **Example**:
    ```python
    import streamlit as st
    import plotly.express as px
    import pickle

    st.title("Exoplanet Detection")

    # Load model
    with open('models/lightgbm_model.pkl', 'rb') as f:
        model = pickle.load(f)

    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        # Preprocess df to match training features (81 features)
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]
        st.write(f"Prediction: {'Planet' if predictions[0] == 1 else 'Non-Planet'}")
        st.write(f"Confidence: {probabilities[0]:.2%}")
    ```
- **Output**: Interactive app hosted locally or on Streamlit Cloud.

### 4. Documentation & Submission
- **Repo Structure**:
  ```
  /data/
    kepler_koi.csv                          # Raw dataset (153 columns)
    /preprocessing/
      kepler_koi_preprocessed.csv           # Preprocessed (82 columns: 81 features + label)
      data_preprocessing.py                 # Preprocessing pipeline class
      preprocessor.pkl                      # Scaler and imputer for inference
      preprocessing_summary.txt             # Preprocessing summary
  /models/
    train_model.py                          # Model training code (handles train/val/test split)
    lightgbm_model.pkl                      # Trained LightGBM model
    model_metrics.json                      # Performance metrics
  /src/
    app.py                                  # Streamlit app
  /README.md                                # Project overview, results, instructions
  /CLAUDE.md                                # This file (instructions for Claude Code)
  ```
- **Tasks**:
  - Document preprocessing steps, model performance, and usage instructions.
  - Cite NASA Exoplanet Archive (DOI: http://doi.org/10.17616/R3X31K).
  - Submit as open-source GitHub repo with clear README.



## Guidance for Claude Code

**Focus**: Assist with Python code for:
- Preprocessing `kepler_koi.csv` (remove data leakage, handle NaNs, standardize, encode labels). **Do not split data during preprocessing.**
- Training/tuning LightGBM in `models/train_model.py` (load preprocessed data, split into train/val/test, hyperparameter optimization, cross-validation).
- Building Streamlit app in `src/app.py` with Plotly visualizations.
- Debugging errors (e.g., missing columns, NaN issues, feature mismatches).

**Constraints**:
- Use **only** `kepler_koi.csv` (no raw .fits light curves due to time/storage limits).
- Prioritize **LightGBM**; avoid complex models like CNN/TCN unless requested.
- Keep code **simple, modular**, and compatible with scikit-learn, LightGBM, pandas, Plotly, Streamlit.
- **Store models and training code in `models/` directory**, not `src/`.
- **Always remove data leakage features** (`koi_prad`, `koi_teq`, `koi_insol`, `koi_score`, etc.).

**Example Tasks**:
- Generate code to load and preprocess `kepler_koi.csv` with leakage removal (no splitting).
- Create model training script in `models/train_model.py` that loads preprocessed data, splits it, trains, and evaluates.
- Suggest LightGBM hyperparameter grid for GridSearchCV.
- Create Plotly plot for `koi_depth` vs. `koi_period` colored by prediction/actual label.
- Debug Streamlit app if prediction fails or features don't match.



Team & Context

Team: You (ML expert, experienced in Python, LightGBM, Streamlit) and Nilesh (UH CS student, AI developer, data pipelines).
Timeline: 48-hour hackathon (October 4-5, 2025).
Goal: High-accuracy model (>95% F1), user-friendly demo, and clear documentation to empower astronomers.

Resources

Dataset: kepler_koi.csv (local, from NASA Exoplanet Archive).
References: NASA Space Apps Challenge resources (e.g., "Assessment of Ensemble-Based Machine Learning Algorithms for Exoplanet Identification" for LightGBM tips, https://www.mdpi.com/2079-9292/13/19/3950/pdf).

### Explanation
- **Purpose**: The file clearly states the project’s goal (exoplanet classification using KOI data) and your choice of LightGBM, aligning with your skills and the challenge’s supervised learning focus.
- **Dataset**: Specifies `kepler_koi.csv` and key columns (`koi_disposition` for labels, features like `koi_period`), ensuring Claude focuses on the provided CSV.
- **Framework**: Outlines a streamlined workflow (preprocessing, modeling, visualization, deployment) based on your previous plan, tailored for LightGBM and hackathon constraints.
- **Guidance for Claude**: Directs the assistant to assist with specific tasks (e.g., preprocessing, tuning, Streamlit), ensuring relevant and actionable support.
- **Context**: Includes team roles and timeline to keep Claude aligned with your hackathon needs.
- Make the commit messages much simpler (1-2 sentences max).
- remove citation to yourself
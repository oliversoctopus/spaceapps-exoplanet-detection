# app.py (Final version with robust SHAP explainer)

from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import os

# --- Build absolute paths from the script's location ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_FILE = os.path.join(BASE_DIR, 'data', 'cumulative_2025.10.04_07.47.26.csv')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

try:
    model = joblib.load(os.path.join(MODELS_DIR, 'exoplanet_classifier.joblib'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, 'feature_columns.joblib'))
except FileNotFoundError:
    print("FATAL ERROR: Model files not found. Did you run train_model.py first?")
    exit()

# --- Initialize SHAP Explainer using the modern, unified interface ---
explainer = shap.Explainer(model, feature_names=feature_columns)

# Dynamically load and select candidates to guarantee they exist
try:
    data = pd.read_csv(DATA_FILE, comment='#')
    df_clean = data[feature_columns + ['kepoi_name', 'koi_disposition']].dropna()
    confirmed_planet = df_clean[df_clean['koi_disposition'] == 'CONFIRMED'].iloc[0]
    candidate_planet = df_clean[df_clean['koi_disposition'] == 'CANDIDATE'].iloc[1]
    false_positive = df_clean[df_clean['koi_disposition'] == 'FALSE POSITIVE'].iloc[0]
    df_candidates = pd.concat([confirmed_planet.to_frame().T, candidate_planet.to_frame().T, false_positive.to_frame().T])
    CANDIDATE_DATA = {
        row['kepoi_name']: row[feature_columns].to_dict()
        for index, row in df_candidates.iterrows()
    }
    print("Successfully loaded and dynamically selected candidate data from CSV.")
except Exception as e:
    print(f"WARNING: Could not load candidate data from CSV. Using dummy data. Error: {e}")
    CANDIDATE_DATA = {"error_loading": {col: 0 for col in feature_columns}}

@app.route('/')
def home():
    return render_template('index.html', candidates=CANDIDATE_DATA.keys())

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    candidate_id = data['candidate_id']
    features_dict = CANDIDATE_DATA[candidate_id]
    features_df = pd.DataFrame([features_dict]).reindex(columns=feature_columns, fill_value=0)
    features_scaled = scaler.transform(features_df)
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    confidence_score = probabilities[prediction]

    # --- Generate SHAP Waterfall Plot using the modern, robust method ---
    shap_explanation = explainer(features_scaled)

    plt.figure()
    # Select the explanations for the positive class (class 1, "Exoplanet Candidate")
    # The syntax [0, :, 1] selects the first sample, all features, and the second class.
    shap.plots.waterfall(shap_explanation[0, :, 1], max_display=10, show=False)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    explanation_text = "This plot shows how each feature contributes to the prediction. Features in <span style='color:red;'>red</span> push the prediction towards 'Exoplanet', while those in <span style='color:blue;'>blue</span> push it lower."
    response = {
        'prediction_text': 'Exoplanet Candidate' if prediction == 1 else 'Likely False Positive',
        'confidence': f"{confidence_score:.2%}",
        'explanation': explanation_text,
        'plot_b64': plot_base64
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
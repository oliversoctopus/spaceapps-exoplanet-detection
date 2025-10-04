# train_model.py (Corrected to avoid circular reasoning)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

print("Starting model training with Kepler data...")

# Point to the 'data' folder
DATA_FILE = os.path.join('data', 'cumulative_2025.10.04_07.47.26.csv')

os.makedirs('models', exist_ok=True)

try:
    data = pd.read_csv(DATA_FILE, comment='#')
    print("Data loaded successfully from 'data' folder.")
except FileNotFoundError:
    print(f"Error: '{DATA_FILE}' not found.")
    exit()

# --- CORRECTED FEATURE LIST ---
# The four 'koi_fpflag' columns have been removed to prevent data leakage.
feature_columns = [
    'koi_period',        # Orbital Period
    'koi_duration',      # Transit Duration
    'koi_depth',         # Transit Depth
    'koi_prad',          # Planetary Radius
    'koi_impact',        # Impact Parameter
    'koi_teq',           # Equilibrium Temperature
    'koi_insol',         # Insolation Flux
    'koi_steff',         # Stellar Effective Temperature
    'koi_slogg',         # Stellar Surface Gravity
    'koi_srad',          # Stellar Radius
]

target_column = 'koi_disposition'

# (The rest of the script is exactly the same)
df = data[data[target_column].isin(['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'])].copy()
df['is_exoplanet'] = df[target_column].apply(lambda x: 1 if x in ['CONFIRMED', 'CANDIDATE'] else 0)
df_clean = df[['is_exoplanet'] + feature_columns].dropna()
X = df_clean[feature_columns]
y = df_clean['is_exoplanet']

print(f"Data prepared. Using {len(X.columns)} features. Shape of data: {X.shape}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=y)

print("Training the Random Forest model...")
model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print(classification_report(y_test, predictions, target_names=['False Positive', 'Exoplanet Candidate']))
print("------------------------\n")

print("Saving model, scaler, and feature list to the 'models/' directory...")
joblib.dump(model, 'models/exoplanet_classifier.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(feature_columns, 'models/feature_columns.joblib')

print("Training complete and artifacts saved!")
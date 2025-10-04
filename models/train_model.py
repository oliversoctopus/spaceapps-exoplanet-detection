"""
LightGBM Model Training for NASA Kepler Exoplanet Detection
===========================================================
Trains a LightGBM binary classifier to detect exoplanets vs. false positives.
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

def load_preprocessed_data(filepath):
    """Load preprocessed dataset"""
    print("Loading preprocessed data...")
    df = pd.read_csv(filepath)
    print(f"  Loaded {df.shape[0]} samples with {df.shape[1]} columns")

    # Separate features and labels
    X = df.drop('label', axis=1)
    y = df['label']

    print(f"\nDataset summary:")
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {len(X)}")
    print(f"  Planets: {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.2f}%)")
    print(f"  Non-planets: {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.2f}%)")

    return X, y

def split_data(X, y, test_size=0.2, val_size=0.125, random_state=42):
    """Split data into train, validation, and test sets (70/10/20)"""
    print("\nSplitting data...")

    # First split: train+val (80%) and test (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: train (70%) and val (10%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )

    print(f"  Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

    # Check class distribution
    for name, y_set in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        planet_pct = (y_set == 1).sum() / len(y_set) * 100
        print(f"    {name} planet %: {planet_pct:.1f}%")

    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(X_train, y_train, X_val, y_val):
    """Train LightGBM classifier"""
    print("\nTraining LightGBM model...")

    # Initialize model with optimized hyperparameters
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )

    # Train model
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='binary_logloss',
        callbacks=[],  # No verbose callbacks
    )

    print("  Model training complete!")

    return model

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Evaluate model performance on all splits"""
    print("\nEvaluating model...")

    results = {}

    for name, X, y in [('Train', X_train, y_train),
                        ('Validation', X_val, y_val),
                        ('Test', X_test, y_test)]:

        # Predictions
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        # Metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_proba)
        }

        results[name.lower()] = metrics

        print(f"\n  {name} Set:")
        print(f"    Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1 Score:  {metrics['f1']:.4f}")
        print(f"    ROC-AUC:   {metrics['roc_auc']:.4f}")

    # Confusion matrix for test set
    cm = confusion_matrix(y_test, model.predict(X_test))
    print(f"\n  Test Set Confusion Matrix:")
    print(f"    TN: {cm[0][0]}, FP: {cm[0][1]}")
    print(f"    FN: {cm[1][0]}, TP: {cm[1][1]}")

    return results

def save_model_and_metrics(model, metrics, feature_names):
    """Save trained model and performance metrics"""
    print("\nSaving model and metrics...")

    # Save model
    model_path = 'lightgbm_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Saved model to {model_path}")

    # Save metrics
    metrics_data = {
        'metrics': metrics,
        'feature_count': len(feature_names),
        'feature_names': feature_names.tolist(),
        'model_params': model.get_params()
    }

    metrics_path = 'model_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"  Saved metrics to {metrics_path}")

def main():
    """Main training pipeline"""
    print("="*60)
    print("LIGHTGBM MODEL TRAINING PIPELINE")
    print("="*60)

    # Load preprocessed data
    X, y = load_preprocessed_data('../data/preprocessing/kepler_koi_preprocessed.csv')

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Train model
    model = train_model(X_train, y_train, X_val, y_val)

    # Evaluate model
    metrics = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test)

    # Save model and metrics
    save_model_and_metrics(model, metrics, X.columns)

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nFiles saved:")
    print("  - lightgbm_model.pkl (trained model)")
    print("  - model_metrics.json (performance metrics)")

if __name__ == "__main__":
    main()

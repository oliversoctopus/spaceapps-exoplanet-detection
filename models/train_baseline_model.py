"""
Baseline LightGBM Model Training for NASA Exoplanet Detection
==============================================================
Trains a baseline LightGBM model using 9 core features.
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
import json
from datetime import datetime

def load_baseline_data(filepath):
    """Load baseline preprocessed data"""
    print("Loading baseline dataset...")
    df = pd.read_csv(filepath)

    # Separate features and labels
    X = df.drop('label', axis=1)
    y = df['label']

    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {len(X)}")
    print(f"  Planets: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")

    return X, y

def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """Split data into train/val/test sets"""
    print("\nSplitting data...")

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )

    print(f"  Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test

def train_baseline_model(X_train, y_train, X_val, y_val):
    """Train baseline LightGBM model"""
    print("\nTraining baseline LightGBM model...")

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

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc'
    )

    print(f"  [OK] Model trained")
    return model

def evaluate_model(model, X, y, dataset_name):
    """Evaluate model on a dataset"""
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_pred_proba)
    }

    print(f"\n{dataset_name} Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

    return metrics

def save_model(model, feature_names, metrics, output_dir):
    """Save trained model and metrics"""

    # Save model
    model_path = f"{output_dir}/baseline_lightgbm_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n[OK] Model saved to: {model_path}")

    # Save metrics
    metrics_data = {
        'metrics': metrics,
        'feature_count': len(feature_names),
        'feature_names': feature_names,
        'model_params': model.get_params(),
        'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    metrics_path = f"{output_dir}/baseline_model_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[OK] Metrics saved to: {metrics_path}")

def main():
    print("=" * 60)
    print("BASELINE MODEL TRAINING")
    print("=" * 60)

    # Load data
    X, y = load_baseline_data('../data/preprocessing/kepler_koi_baseline.csv')

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Train model
    model = train_baseline_model(X_train, y_train, X_val, y_val)

    # Evaluate on all splits
    train_metrics = evaluate_model(model, X_train, y_train, "Train")
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    test_metrics = evaluate_model(model, X_test, y_test, "Test")

    # Combine metrics
    all_metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics
    }

    # Save model and metrics
    save_model(model, X.columns.tolist(), all_metrics, '.')

    print("\n" + "=" * 60)
    print("BASELINE MODEL TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nBaseline Test Performance:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.1%}")
    print(f"  F1-Score:  {test_metrics['f1']:.1%}")
    print(f"  ROC-AUC:   {test_metrics['roc_auc']:.1%}")

if __name__ == "__main__":
    main()

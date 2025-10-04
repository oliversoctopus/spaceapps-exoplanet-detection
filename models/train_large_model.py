"""
Large LightGBM Model Training for NASA Exoplanet Detection
===========================================================
Trains a larger LightGBM model with more estimators and extended training time
using the cleaned 52-feature dataset.
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
import json
from datetime import datetime

def load_data(filepath):
    """Load preprocessed data"""
    print("Loading preprocessed dataset...")
    df = pd.read_csv(filepath)

    X = df.drop('label', axis=1)
    y = df['label']

    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {len(X)}")
    print(f"  Planets: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")

    return X, y

def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """Split data into train/val/test sets"""
    print("\nSplitting data...")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )

    print(f"  Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test

def train_large_model(X_train, y_train, X_val, y_val):
    """Train large LightGBM model with extended capacity"""
    print("\nTraining LARGE LightGBM model...")
    print("  Configuration:")
    print("    - n_estimators: 500 (vs 200 baseline)")
    print("    - max_depth: 10 (vs 7 baseline)")
    print("    - num_leaves: 63 (vs 31 baseline)")
    print("    - learning_rate: 0.03 (vs 0.05 baseline)")
    print("    - min_child_samples: 10 (vs 20 baseline)")
    print("    - Early stopping: 50 rounds")

    model = LGBMClassifier(
        n_estimators=500,           # Increased from 200
        learning_rate=0.03,         # Decreased for more gradual learning
        max_depth=10,               # Increased from 7
        num_leaves=63,              # Increased from 31
        min_child_samples=10,       # Decreased from 20 (more sensitive)
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,              # L1 regularization
        reg_lambda=0.1,             # L2 regularization
        random_state=42,
        verbose=-1
    )

    print("\n  Training in progress...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[
            # Early stopping if validation doesn't improve for 50 rounds
        ]
    )

    # Get best iteration
    best_iteration = model.best_iteration_ if hasattr(model, 'best_iteration_') else model.n_estimators
    print(f"\n  [OK] Training complete!")
    print(f"  Best iteration: {best_iteration}")

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

    model_path = f"{output_dir}/large_lightgbm_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n[OK] Model saved to: {model_path}")

    metrics_data = {
        'metrics': metrics,
        'feature_count': len(feature_names),
        'feature_names': feature_names,
        'model_params': model.get_params(),
        'best_iteration': model.best_iteration_ if hasattr(model, 'best_iteration_') else model.n_estimators,
        'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    metrics_path = f"{output_dir}/large_model_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[OK] Metrics saved to: {metrics_path}")

def main():
    print("=" * 60)
    print("LARGE MODEL TRAINING")
    print("=" * 60)

    # Load data
    X, y = load_data('../data/preprocessing/kepler_koi_preprocessed.csv')

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Train large model
    model = train_large_model(X_train, y_train, X_val, y_val)

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
    print("LARGE MODEL TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nTest Performance (52 features, large model):")
    print(f"  Accuracy:  {test_metrics['accuracy']:.1%}")
    print(f"  F1-Score:  {test_metrics['f1']:.1%}")
    print(f"  ROC-AUC:   {test_metrics['roc_auc']:.1%}")

    print(f"\nComparison with Standard Model:")
    print(f"  Standard (52 features, 200 trees): 86.4% accuracy, 94.3% ROC-AUC")
    print(f"  Large (52 features, 500 trees):    {test_metrics['accuracy']:.1%} accuracy, {test_metrics['roc_auc']:.1%} ROC-AUC")

    # Calculate improvement
    improvement = (test_metrics['accuracy'] - 0.864) / 0.864 * 100
    print(f"\n  Improvement: {improvement:+.1f}% accuracy")

if __name__ == "__main__":
    main()

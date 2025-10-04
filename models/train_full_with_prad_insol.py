"""
Full LightGBM Model Training with koi_prad and koi_insol
=========================================================
Trains model with 58 features including verified non-leaking features:
koi_prad (planet radius) and koi_insol (insolation flux).
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

def train_model(X_train, y_train, X_val, y_val):
    """Train LightGBM model with optimized regularization"""
    print("\nTraining LightGBM model (58 features with koi_prad & koi_insol)...")
    print("  Configuration:")
    print("    - n_estimators: 1000")
    print("    - learning_rate: 0.02")
    print("    - max_depth: 6")
    print("    - num_leaves: 31")
    print("    - min_child_samples: 50")
    print("    - subsample: 0.6")
    print("    - colsample_bytree: 0.6")
    print("    - reg_alpha: 2.0 (L1)")
    print("    - reg_lambda: 2.0 (L2)")
    print("    - Early stopping: 100 rounds")

    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=6,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.6,
        colsample_bytree=0.6,
        reg_alpha=2.0,
        reg_lambda=2.0,
        min_split_gain=0.2,
        random_state=42,
        verbose=-1
    )

    print("\n  Training in progress...")
    from lightgbm import early_stopping

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[early_stopping(stopping_rounds=100, verbose=False)]
    )

    best_iteration = model.best_iteration_ if hasattr(model, 'best_iteration_') else model.n_estimators
    print(f"\n  [OK] Training complete!")
    print(f"  Best iteration: {best_iteration}/{model.n_estimators}")

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

    model_path = f"{output_dir}/full_with_prad_insol_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n[OK] Model saved to: {model_path}")

    metrics_data = {
        'metrics': metrics,
        'feature_count': len(feature_names),
        'feature_names': feature_names,
        'added_features': ['koi_prad', 'koi_prad_err1', 'koi_prad_err2',
                          'koi_insol', 'koi_insol_err1', 'koi_insol_err2'],
        'model_params': model.get_params(),
        'best_iteration': model.best_iteration_ if hasattr(model, 'best_iteration_') else model.n_estimators,
        'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    metrics_path = f"{output_dir}/full_with_prad_insol_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[OK] Metrics saved to: {metrics_path}")

def main():
    print("=" * 60)
    print("FULL MODEL WITH PRAD & INSOL")
    print("=" * 60)

    # Load data
    X, y = load_data('../data/preprocessing/kepler_koi_preprocessed.csv')

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Train model
    model = train_model(X_train, y_train, X_val, y_val)

    # Evaluate on all splits
    train_metrics = evaluate_model(model, X_train, y_train, "Train")
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    test_metrics = evaluate_model(model, X_test, y_test, "Test")

    # Calculate train/test gap
    train_test_gap = train_metrics['accuracy'] - test_metrics['accuracy']

    # Combine metrics
    all_metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics,
        'train_test_gap': train_test_gap
    }

    # Save model and metrics
    save_model(model, X.columns.tolist(), all_metrics, '.')

    print("\n" + "=" * 60)
    print("FULL MODEL TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nTest Performance (58 features with koi_prad & koi_insol):")
    print(f"  Accuracy:  {test_metrics['accuracy']:.1%}")
    print(f"  F1-Score:  {test_metrics['f1']:.1%}")
    print(f"  ROC-AUC:   {test_metrics['roc_auc']:.1%}")
    print(f"  Train/Test Gap: {train_test_gap:.1%}")

    print(f"\nComparison:")
    print(f"  Previous (52 features):        86.9% accuracy, 94.5% ROC-AUC, 7.3% gap")
    print(f"  New (58 features + prad/insol): {test_metrics['accuracy']:.1%} accuracy, {test_metrics['roc_auc']:.1%} ROC-AUC, {train_test_gap:.1%} gap")

    improvement = (test_metrics['accuracy'] - 0.869) * 100
    print(f"\n  Improvement: {improvement:+.1f}% accuracy")

if __name__ == "__main__":
    main()

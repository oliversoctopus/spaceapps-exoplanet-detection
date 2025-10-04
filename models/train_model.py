"""
Full LightGBM Model Training for NASA Exoplanet Detection
==========================================================
Uses advanced regularization and hyperparameter tuning
to reduce overfitting and improve generalization.
This is the main full model (52 features, optimized settings).
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
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
    """Train full LightGBM model with advanced regularization"""
    print("\nTraining full LightGBM model (optimized settings)...")
    print("  Advanced Configuration:")
    print("    - n_estimators: 1000 (with early stopping)")
    print("    - learning_rate: 0.02 (slower, more stable)")
    print("    - max_depth: 6 (conservative depth)")
    print("    - num_leaves: 31 (balanced complexity)")
    print("    - min_child_samples: 50 (strict minimum)")
    print("    - min_data_in_bin: 5 (reduce noise)")
    print("    - subsample: 0.6 (aggressive bagging)")
    print("    - colsample_bytree: 0.6 (aggressive feature sampling)")
    print("    - reg_alpha: 2.0 (strong L1)")
    print("    - reg_lambda: 2.0 (strong L2)")
    print("    - min_split_gain: 0.2 (require meaningful splits)")
    print("    - path_smooth: 1.0 (smooth leaf prediction)")
    print("    - Early stopping: 100 rounds (patient)")

    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.02,           # Slower learning
        max_depth=6,                  # Conservative depth
        num_leaves=31,                # Balanced
        min_child_samples=50,         # Very strict
        min_data_in_bin=5,            # Reduce bin noise
        subsample=0.6,                # Strong bagging
        colsample_bytree=0.6,         # Strong feature sampling
        subsample_freq=1,             # Apply bagging every iteration
        reg_alpha=2.0,                # Strong L1 regularization
        reg_lambda=2.0,               # Strong L2 regularization
        min_split_gain=0.2,           # Require meaningful gain
        min_child_weight=0.05,        # Higher minimum weight
        path_smooth=1.0,              # Smooth predictions
        max_bin=200,                  # Reduce bins for smoother splits
        random_state=42,
        verbose=-1
    )

    print("\n  Training in progress...")
    from lightgbm import early_stopping, log_evaluation

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=['train', 'valid'],
        eval_metric=['auc', 'binary_logloss'],
        callbacks=[
            early_stopping(stopping_rounds=100, verbose=False),
            log_evaluation(period=0)  # Suppress output
        ]
    )

    best_iteration = model.best_iteration_ if hasattr(model, 'best_iteration_') else model.n_estimators
    print(f"\n  [OK] Training complete!")
    print(f"  Best iteration: {best_iteration}/{model.n_estimators}")

    return model

def cross_validate_predictions(X_train_full, y_train_full, X_test, n_splits=5):
    """Perform stratified k-fold cross-validation for ensemble predictions"""
    print("\nPerforming 5-fold cross-validation ensemble...")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    test_predictions = np.zeros((len(X_test), n_splits))
    train_scores = []
    val_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full), 1):
        print(f"  Fold {fold}/{n_splits}...", end=' ')

        X_train = X_train_full.iloc[train_idx]
        X_val = X_train_full.iloc[val_idx]
        y_train = y_train_full.iloc[train_idx]
        y_val = y_train_full.iloc[val_idx]

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
            random_state=42 + fold,
            verbose=-1
        )

        from lightgbm import early_stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[early_stopping(stopping_rounds=100, verbose=False)]
        )

        # Get scores
        train_pred = model.predict_proba(X_train)[:, 1]
        val_pred = model.predict_proba(X_val)[:, 1]
        train_scores.append(roc_auc_score(y_train, train_pred))
        val_scores.append(roc_auc_score(y_val, val_pred))

        # Store test predictions
        test_predictions[:, fold-1] = model.predict_proba(X_test)[:, 1]

        print(f"ROC-AUC: {val_scores[-1]:.4f}")

    print(f"\n  CV Train ROC-AUC: {np.mean(train_scores):.4f} ± {np.std(train_scores):.4f}")
    print(f"  CV Valid ROC-AUC: {np.mean(val_scores):.4f} ± {np.std(val_scores):.4f}")
    print(f"  Train/Valid gap: {np.mean(train_scores) - np.mean(val_scores):.4f}")

    # Average predictions across folds
    avg_predictions = test_predictions.mean(axis=1)

    return avg_predictions

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

    model_path = f"{output_dir}/lightgbm_model.pkl"
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

    metrics_path = f"{output_dir}/model_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[OK] Metrics saved to: {metrics_path}")

def main():
    print("=" * 60)
    print("FULL MODEL TRAINING (OPTIMIZED SETTINGS)")
    print("=" * 60)

    # Load data
    X, y = load_data('../data/preprocessing/kepler_koi_preprocessed.csv')

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Train full model
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
    print(f"\nTest Performance (52 features):")
    print(f"  Accuracy:  {test_metrics['accuracy']:.1%}")
    print(f"  F1-Score:  {test_metrics['f1']:.1%}")
    print(f"  ROC-AUC:   {test_metrics['roc_auc']:.1%}")
    print(f"  Train/Test Gap: {train_test_gap:.1%}")

    print(f"\nComparison with baseline:")
    print(f"  Baseline (9 features):  84.3% accuracy, 92.7% ROC-AUC")
    print(f"  Full (52 features):     {test_metrics['accuracy']:.1%} accuracy, {test_metrics['roc_auc']:.1%} ROC-AUC, {train_test_gap:.1%} gap")

if __name__ == "__main__":
    main()

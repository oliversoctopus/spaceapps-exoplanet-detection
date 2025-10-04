"""
Baseline Preprocessing Pipeline for NASA Kepler Exoplanet Detection
====================================================================
Creates a minimal baseline dataset with 9 core features for initial model training.
Data splitting is handled during model training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle
import warnings
warnings.filterwarnings('ignore')

class BaselinePreprocessor:
    """Baseline preprocessing pipeline with 9 core features"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = [
            'koi_steff',      # Stellar effective temperature
            'koi_slogg',      # Stellar surface gravity
            'koi_srad',       # Stellar radius
            'koi_kepmag',     # Kepler magnitude
            'koi_period',     # Orbital period
            'koi_depth',      # Transit depth
            'koi_duration',   # Transit duration
            'koi_impact',     # Impact parameter
            'koi_model_snr'   # Model signal-to-noise ratio
        ]
        self.label_mapping = {
            'FALSE POSITIVE': 0,
            'CANDIDATE': 1,
            'CONFIRMED': 1
        }

    def load_data(self, filepath):
        """Load the Kepler KOI dataset"""
        print("Loading dataset...")
        df = pd.read_csv(filepath)
        print(f"  Loaded {df.shape[0]} samples with {df.shape[1]} features")
        return df

    def create_binary_labels(self, df):
        """Create binary labels from koi_disposition"""
        print("\nCreating binary labels...")

        # Map dispositions to binary labels
        df['label'] = df['koi_disposition'].map(self.label_mapping)

        # Check for any unmapped values
        unmapped = df[df['label'].isna()]['koi_disposition'].unique()
        if len(unmapped) > 0:
            print(f"  Warning: Unmapped dispositions found: {unmapped}")
            df = df[df['label'].notna()]

        df['label'] = df['label'].astype(int)

        # Print label distribution
        label_counts = df['label'].value_counts()
        print(f"  Planets (1): {label_counts.get(1, 0)} samples ({label_counts.get(1, 0)/len(df)*100:.2f}%)")
        print(f"  Non-planets (0): {label_counts.get(0, 0)} samples ({label_counts.get(0, 0)/len(df)*100:.2f}%)")

        return df

    def select_baseline_features(self, df):
        """Select 9 baseline features"""
        print(f"\nSelecting {len(self.feature_columns)} baseline features...")

        # Check which features exist
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            print(f"  ERROR: Missing features: {missing_features}")
            raise ValueError(f"Required features not found: {missing_features}")

        print(f"  Features selected:")
        for i, feat in enumerate(self.feature_columns, 1):
            print(f"    {i}. {feat}")

        # Keep only baseline features + label
        df_baseline = df[self.feature_columns + ['label']].copy()

        print(f"\n  Dataset shape: {df_baseline.shape}")
        return df_baseline

    def handle_missing_values(self, df):
        """Handle missing values using median imputation"""
        print("\nHandling missing values...")

        # Check missing values before imputation
        missing_before = df[self.feature_columns].isnull().sum()
        total_missing = missing_before.sum()

        if total_missing > 0:
            print(f"  Total missing values: {total_missing}")
            print(f"  Missing by feature:")
            for feat, count in missing_before[missing_before > 0].items():
                print(f"    {feat}: {count} ({count/len(df)*100:.1f}%)")

            # Impute missing values
            df[self.feature_columns] = self.imputer.fit_transform(df[self.feature_columns])
            print(f"  [OK] Imputation complete (median strategy)")
        else:
            print(f"  No missing values found")

        return df

    def scale_features(self, df):
        """Normalize features using StandardScaler"""
        print("\nScaling features (StandardScaler)...")

        # Fit scaler and transform features
        df[self.feature_columns] = self.scaler.fit_transform(df[self.feature_columns])

        print(f"  [OK] Features scaled to zero mean and unit variance")
        return df

    def save_preprocessor(self, output_dir):
        """Save preprocessor configuration"""
        preprocessor_data = {
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_columns': self.feature_columns,
            'label_mapping': self.label_mapping
        }

        output_path = f"{output_dir}/baseline_preprocessor.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(preprocessor_data, f)

        print(f"\n[OK] Preprocessor saved to: {output_path}")

    def save_summary(self, df, output_dir):
        """Save preprocessing summary"""
        summary = f"""BASELINE PREPROCESSING SUMMARY
========================================

Number of samples: {len(df)}
Number of features: {len(self.feature_columns)}
Planet percentage: {df['label'].sum()/len(df)*100:.2f}%

Baseline Features ({len(self.feature_columns)}):
"""
        for i, feat in enumerate(self.feature_columns, 1):
            summary += f"  {i}. {feat}\n"

        summary += "\nNote: Data splitting (train/val/test) is handled during model training.\n"

        output_path = f"{output_dir}/baseline_preprocessing_summary.txt"
        with open(output_path, 'w') as f:
            f.write(summary)

        print(f"[OK] Summary saved to: {output_path}")

    def preprocess(self, input_file, output_dir):
        """Run complete preprocessing pipeline"""
        print("=" * 60)
        print("BASELINE PREPROCESSING PIPELINE")
        print("=" * 60)

        # Load data
        df = self.load_data(input_file)

        # Create binary labels
        df = self.create_binary_labels(df)

        # Select baseline features
        df = self.select_baseline_features(df)

        # Handle missing values
        df = self.handle_missing_values(df)

        # Scale features
        df = self.scale_features(df)

        # Save preprocessed dataset
        output_file = f"{output_dir}/kepler_koi_baseline.csv"
        df.to_csv(output_file, index=False)
        print(f"\n[OK] Preprocessed dataset saved to: {output_file}")

        # Save preprocessor and summary
        self.save_preprocessor(output_dir)
        self.save_summary(df, output_dir)

        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE!")
        print("=" * 60)

        return df

if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = BaselinePreprocessor()

    # Run preprocessing
    df_preprocessed = preprocessor.preprocess(
        input_file='../kepler_koi.csv',
        output_dir='.'
    )

    print(f"\nNext step: Train baseline model with 9 features")

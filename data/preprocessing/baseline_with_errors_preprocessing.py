"""
Baseline + Errors Preprocessing Pipeline for NASA Kepler Exoplanet Detection
=============================================================================
Creates dataset with 9 baseline features + their error measurements (24 total).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle
import warnings
warnings.filterwarnings('ignore')

class BaselineWithErrorsPreprocessor:
    """Baseline preprocessing with error measurements"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')

        # 9 baseline features
        self.baseline_features = [
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

        # Error columns for baseline features (14 total)
        # Note: koi_kepmag_err is all NaN, so excluded
        #       koi_model_snr has no error columns
        self.error_features = [
            # Stellar property errors
            'koi_steff_err1', 'koi_steff_err2',
            'koi_slogg_err1', 'koi_slogg_err2',
            'koi_srad_err1', 'koi_srad_err2',
            # Transit signal errors
            'koi_period_err1', 'koi_period_err2',
            'koi_depth_err1', 'koi_depth_err2',
            'koi_duration_err1', 'koi_duration_err2',
            'koi_impact_err1', 'koi_impact_err2'
        ]

        # Combined features (23 total)
        self.feature_columns = self.baseline_features + self.error_features

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

        df['label'] = df['koi_disposition'].map(self.label_mapping)

        unmapped = df[df['label'].isna()]['koi_disposition'].unique()
        if len(unmapped) > 0:
            print(f"  Warning: Unmapped dispositions found: {unmapped}")
            df = df[df['label'].notna()]

        df['label'] = df['label'].astype(int)

        label_counts = df['label'].value_counts()
        print(f"  Planets (1): {label_counts.get(1, 0)} samples ({label_counts.get(1, 0)/len(df)*100:.2f}%)")
        print(f"  Non-planets (0): {label_counts.get(0, 0)} samples ({label_counts.get(0, 0)/len(df)*100:.2f}%)")

        return df

    def select_features(self, df):
        """Select baseline + error features"""
        print(f"\nSelecting {len(self.feature_columns)} features (9 baseline + 14 errors)...")

        # Check which features exist
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            print(f"  ERROR: Missing features: {missing_features}")
            raise ValueError(f"Required features not found: {missing_features}")

        print(f"\n  Baseline Features (9):")
        for i, feat in enumerate(self.baseline_features, 1):
            print(f"    {i}. {feat}")

        print(f"\n  Error Features (14):")
        for i, feat in enumerate(self.error_features, 1):
            print(f"    {i}. {feat}")

        # Keep only selected features + label
        df_selected = df[self.feature_columns + ['label']].copy()

        print(f"\n  Total features: {len(self.feature_columns)}")
        print(f"  Dataset shape: {df_selected.shape}")
        return df_selected

    def handle_missing_values(self, df):
        """Handle missing values using median imputation"""
        print("\nHandling missing values...")

        missing_before = df[self.feature_columns].isnull().sum()
        total_missing = missing_before.sum()

        if total_missing > 0:
            print(f"  Total missing values: {total_missing}")
            print(f"  Missing by feature (top 10):")
            for feat, count in missing_before[missing_before > 0].head(10).items():
                print(f"    {feat}: {count} ({count/len(df)*100:.1f}%)")

            # Fit and transform, then assign back properly
            imputed_values = self.imputer.fit_transform(df[self.feature_columns])
            df_imputed = pd.DataFrame(imputed_values, columns=self.feature_columns, index=df.index)
            df[self.feature_columns] = df_imputed
            print(f"  [OK] Imputation complete (median strategy)")
        else:
            print(f"  No missing values found")

        return df

    def scale_features(self, df):
        """Normalize features using StandardScaler"""
        print("\nScaling features (StandardScaler)...")

        # Fit and transform, then assign back properly
        scaled_values = self.scaler.fit_transform(df[self.feature_columns])
        df_scaled = pd.DataFrame(scaled_values, columns=self.feature_columns, index=df.index)
        df[self.feature_columns] = df_scaled

        print(f"  [OK] Features scaled to zero mean and unit variance")
        return df

    def save_preprocessor(self, output_dir):
        """Save preprocessor configuration"""
        preprocessor_data = {
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_columns': self.feature_columns,
            'baseline_features': self.baseline_features,
            'error_features': self.error_features,
            'label_mapping': self.label_mapping
        }

        output_path = f"{output_dir}/baseline_with_errors_preprocessor.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(preprocessor_data, f)

        print(f"\n[OK] Preprocessor saved to: {output_path}")

    def save_summary(self, df, output_dir):
        """Save preprocessing summary"""
        summary = f"""BASELINE + ERRORS PREPROCESSING SUMMARY
========================================

Number of samples: {len(df)}
Number of features: {len(self.feature_columns)} (9 baseline + 14 errors)
Planet percentage: {df['label'].sum()/len(df)*100:.2f}%

Baseline Features (9):
"""
        for i, feat in enumerate(self.baseline_features, 1):
            summary += f"  {i}. {feat}\n"

        summary += f"\nError Features (14):\n"
        for i, feat in enumerate(self.error_features, 1):
            summary += f"  {i}. {feat}\n"

        summary += "\nNote: Data splitting (train/val/test) is handled during model training.\n"

        output_path = f"{output_dir}/baseline_with_errors_summary.txt"
        with open(output_path, 'w') as f:
            f.write(summary)

        print(f"[OK] Summary saved to: {output_path}")

    def preprocess(self, input_file, output_dir):
        """Run complete preprocessing pipeline"""
        print("=" * 60)
        print("BASELINE + ERRORS PREPROCESSING PIPELINE")
        print("=" * 60)

        df = self.load_data(input_file)
        df = self.create_binary_labels(df)
        df = self.select_features(df)
        df = self.handle_missing_values(df)
        df = self.scale_features(df)

        output_file = f"{output_dir}/kepler_koi_baseline_with_errors.csv"
        df.to_csv(output_file, index=False)
        print(f"\n[OK] Preprocessed dataset saved to: {output_file}")

        self.save_preprocessor(output_dir)
        self.save_summary(df, output_dir)

        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE!")
        print("=" * 60)

        return df

if __name__ == "__main__":
    preprocessor = BaselineWithErrorsPreprocessor()

    df_preprocessed = preprocessor.preprocess(
        input_file='../kepler_koi.csv',
        output_dir='.'
    )

    print(f"\nNext step: Train baseline+errors model with {len(preprocessor.feature_columns)} features")

"""
Data Preprocessing Pipeline for NASA Kepler Exoplanet Detection
================================================================
Prepares the Kepler KOI dataset for machine learning by handling
missing values, encoding labels, and normalizing features.
Data splitting is handled during model training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
import pickle
import warnings
warnings.filterwarnings('ignore')

class ExoplanetPreprocessor:
    """Preprocessing pipeline for exoplanet detection"""

    def __init__(self, scaling_method='standard'):
        """
        Initialize the preprocessor

        Parameters:
        -----------
        scaling_method : str
            Method for scaling features ('minmax' or 'standard')
        """
        self.scaling_method = scaling_method
        self.scaler = StandardScaler() if scaling_method == 'standard' else MinMaxScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = None
        self.label_mapping = {
            'FALSE POSITIVE': 0,
            'CANDIDATE': 1,
            'CONFIRMED': 1
        }

    def load_data(self, filepath):
        """Load the Kepler KOI dataset"""
        print("Loading dataset...")
        # No comment parameter needed - raw dataset doesn't have comments
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
            # Drop rows with unmapped dispositions
            df = df[df['label'].notna()]

        df['label'] = df['label'].astype(int)

        # Print label distribution
        label_counts = df['label'].value_counts()
        print(f"  Planets (1): {label_counts.get(1, 0)} samples ({label_counts.get(1, 0)/len(df)*100:.2f}%)")
        print(f"  Non-planets (0): {label_counts.get(0, 0)} samples ({label_counts.get(0, 0)/len(df)*100:.2f}%)")

        return df

    def select_features(self, df):
        """Select relevant features for model training"""
        print("\nSelecting features and removing data leakage...")

        # Define leakage columns to drop
        leakage_columns = [
            # Direct answer columns
            'koi_disposition', 'koi_pdisposition', 'koi_score', 'kepler_name',

            # Derived planet properties (assume planet exists)
            # Note: koi_prad and koi_insol verified as non-leaking via statistical tests
            'koi_teq', 'koi_teq_err1', 'koi_teq_err2',
            'koi_dor', 'koi_dor_err1', 'koi_dor_err2',
            'koi_incl', 'koi_incl_err1', 'koi_incl_err2',
            'koi_ror', 'koi_ror_err1', 'koi_ror_err2',
            'koi_sma', 'koi_sma_err1', 'koi_sma_err2',
            'koi_eccen', 'koi_eccen_err1', 'koi_eccen_err2',
            'koi_longp', 'koi_longp_err1', 'koi_longp_err2',

            # Vetting/classification metadata
            'koi_vet_stat', 'koi_vet_date', 'koi_comment', 'koi_disp_prov',

            # False positive flags (derived from vetting process - data leakage)
            'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',

            # Centroid offset features (high mutual information - potential leakage)
            # Flux-weighted motion (FWM) features
            'koi_fwm_stat_sig', 'koi_fwm_sra', 'koi_fwm_sra_err',
            'koi_fwm_sdec', 'koi_fwm_sdec_err', 'koi_fwm_srao', 'koi_fwm_srao_err',
            'koi_fwm_sdeco', 'koi_fwm_sdeco_err', 'koi_fwm_prao', 'koi_fwm_prao_err',
            'koi_fwm_pdeco', 'koi_fwm_pdeco_err',
            # Difference image centroid offset (DICCO) features
            'koi_dicco_mra', 'koi_dicco_mra_err', 'koi_dicco_mdec', 'koi_dicco_mdec_err',
            'koi_dicco_msky', 'koi_dicco_msky_err',
            # Difference image Kepler band centroid offset (DIKCO) features
            'koi_dikco_mra', 'koi_dikco_mra_err', 'koi_dikco_mdec', 'koi_dikco_mdec_err',
            'koi_dikco_msky', 'koi_dikco_msky_err',

            # Data links and identifiers
            'koi_datalink_dvr', 'koi_datalink_dvs',
            'kepoi_name', 'kepid',
            'ra', 'dec', 'ra_str', 'dec_str', 'ra_err', 'dec_err',

            # Delivery/processing metadata
            'koi_tce_delivname', 'koi_delivname', 'koi_quarters', 'koi_count', 'koi_num_transits',

            # Model fitting metadata
            'koi_fittype', 'koi_trans_mod', 'koi_limbdark_mod',
            'koi_ldm_coeff1', 'koi_ldm_coeff2', 'koi_ldm_coeff3', 'koi_ldm_coeff4',
            'koi_parm_prov', 'koi_sparprov'
        ]

        # Get all columns except leakage and label
        all_columns = df.columns.tolist()
        leakage_present = [col for col in leakage_columns if col in all_columns]

        print(f"  Removing {len(leakage_present)} leakage columns")

        # Select features: everything except leakage and label
        self.feature_columns = [col for col in all_columns
                               if col not in leakage_columns and col != 'label']

        # Drop features with >100% missing values (completely empty)
        missing_threshold = 1.0
        features_to_keep = []

        for feature in self.feature_columns:
            missing_pct = df[feature].isnull().sum() / len(df)
            if missing_pct < missing_threshold:
                features_to_keep.append(feature)
            else:
                print(f"  Dropping {feature} due to {missing_pct*100:.1f}% missing values")

        self.feature_columns = features_to_keep
        print(f"  Final feature count: {len(self.feature_columns)}")

        return df

    def handle_missing_values(self, X):
        """Handle missing values using median imputation"""
        print("\nHandling missing values...")

        # Check missing values before imputation
        missing_before = X.isnull().sum().sum()
        print(f"  Missing values before imputation: {missing_before}")

        # Apply imputation
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        # Check missing values after imputation
        missing_after = X_imputed.isnull().sum().sum()
        print(f"  Missing values after imputation: {missing_after}")

        return X_imputed

    def scale_features(self, X):
        """Scale features using the specified method"""
        print(f"\nScaling features using {self.scaling_method}...")

        # Fit and transform all data
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        print(f"  Scaling complete. Feature ranges: [{X_scaled.min().min():.3f}, {X_scaled.max().max():.3f}]")

        return X_scaled

    def preprocess(self, filepath, save_processed=True):
        """
        Full preprocessing pipeline

        Parameters:
        -----------
        filepath : str
            Path to the raw CSV file
        save_processed : bool
            Whether to save processed data to file

        Returns:
        --------
        DataFrame containing preprocessed data with features and label
        """
        print("="*60)
        print("STARTING PREPROCESSING PIPELINE")
        print("="*60)

        # Load and prepare data
        df = self.load_data(filepath)
        df = self.create_binary_labels(df)
        df = self.select_features(df)

        # Separate features and labels
        X = df[self.feature_columns]
        y = df['label']

        # Handle missing values
        X = self.handle_missing_values(X)

        # Scale features
        X_scaled = self.scale_features(X)

        # Combine features and label
        df_processed = X_scaled.copy()
        df_processed['label'] = y.values

        # Create metadata dictionary
        metadata = {
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'label_mapping': self.label_mapping,
            'n_samples': len(df_processed),
            'n_features': len(self.feature_columns),
            'planet_percentage': (y == 1).sum() / len(y) * 100
        }

        if save_processed:
            self.save_processed_data(df_processed, metadata)

        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE!")
        print("="*60)

        return df_processed, metadata

    def save_processed_data(self, df_processed, metadata):
        """Save processed data to file"""
        print("\nSaving processed data...")

        # Save complete preprocessed dataset
        df_processed.to_csv('kepler_koi_preprocessed.csv', index=False)
        print("  Saved preprocessed dataset to kepler_koi_preprocessed.csv")

        # Save preprocessor components for inference
        preprocessor_components = {
            'scaler': metadata['scaler'],
            'imputer': metadata['imputer'],
            'feature_columns': metadata['feature_columns'],
            'label_mapping': metadata['label_mapping']
        }

        with open('preprocessor.pkl', 'wb') as f:
            pickle.dump(preprocessor_components, f)

        print("  Saved preprocessor components to preprocessor.pkl")

        # Save a summary file
        with open('preprocessing_summary.txt', 'w') as f:
            f.write("PREPROCESSING SUMMARY\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Number of samples: {metadata['n_samples']}\n")
            f.write(f"Number of features: {metadata['n_features']}\n")
            f.write(f"Planet percentage: {metadata['planet_percentage']:.2f}%\n\n")
            f.write(f"Features ({len(metadata['feature_columns'])}):\n")
            for i, feature in enumerate(metadata['feature_columns'], 1):
                f.write(f"  {i}. {feature}\n")
            f.write("\nNote: Data splitting (train/val/test) is handled during model training.\n")

        print("  Saved preprocessing summary to preprocessing_summary.txt")

def main():
    """Main execution function"""
    # Initialize preprocessor with StandardScaler
    preprocessor = ExoplanetPreprocessor(scaling_method='standard')

    # Run preprocessing pipeline
    df_processed, metadata = preprocessor.preprocess('../kepler_koi.csv', save_processed=True)

    # Display final summary
    print("\nFinal processed data shape:")
    print(f"  {df_processed.shape[0]} samples × {df_processed.shape[1]} columns ({metadata['n_features']} features + 1 label)")
    print(f"  Planet percentage: {metadata['planet_percentage']:.2f}%")

    print("\nProcessed data saved to:")
    print("  - kepler_koi_preprocessed.csv (complete dataset)")
    print("  - preprocessor.pkl (scaler, imputer, feature info)")
    print("  - preprocessing_summary.txt")
    print("\nNote: Data splitting will be performed during model training.")

if __name__ == "__main__":
    main()
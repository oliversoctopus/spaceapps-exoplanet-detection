"""
Feature Leakage Detection using Statistical Analysis
=====================================================
Performs comprehensive statistical analysis to detect potential data leakage
and circular reasoning in the full model's 77 features.
"""

import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

class LeakageDetector:
    """Statistical leakage detection for exoplanet features"""

    def __init__(self, correlation_threshold=0.7, perfect_separation_threshold=0.95):
        """
        Initialize leakage detector

        Parameters:
        -----------
        correlation_threshold : float
            Correlation above this indicates potential leakage
        perfect_separation_threshold : float
            Accuracy above this for single feature indicates leakage
        """
        self.correlation_threshold = correlation_threshold
        self.perfect_separation_threshold = perfect_separation_threshold
        self.suspicious_features = []

    def load_data(self, filepath):
        """Load preprocessed dataset"""
        print("Loading dataset...")
        df = pd.read_csv(filepath)

        X = df.drop('label', axis=1)
        y = df['label']

        print(f"  Features: {X.shape[1]}")
        print(f"  Samples: {len(X)}")
        print(f"  Planets: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")

        return X, y

    def point_biserial_correlation(self, X, y):
        """Calculate point-biserial correlation for each feature with label"""
        print("\n" + "="*60)
        print("POINT-BISERIAL CORRELATION ANALYSIS")
        print("="*60)
        print("Measures correlation between continuous features and binary label")
        print(f"Threshold: |r| > {self.correlation_threshold} indicates potential leakage\n")

        correlations = {}
        suspicious = []

        for col in X.columns:
            # Point-biserial correlation
            corr, p_value = pointbiserialr(y, X[col])
            correlations[col] = {
                'correlation': corr,
                'p_value': p_value,
                'abs_correlation': abs(corr)
            }

            # Check if suspicious
            if abs(corr) > self.correlation_threshold:
                suspicious.append({
                    'feature': col,
                    'correlation': corr,
                    'p_value': p_value
                })

        # Sort by absolute correlation
        sorted_corr = sorted(correlations.items(),
                            key=lambda x: x[1]['abs_correlation'],
                            reverse=True)

        print("Top 10 Highest Correlations:")
        print(f"{'Feature':<25} {'Correlation':>12} {'P-value':>12} {'Status':>12}")
        print("-" * 65)

        for feat, stats in sorted_corr[:10]:
            status = "SUSPICIOUS" if abs(stats['correlation']) > self.correlation_threshold else "OK"
            print(f"{feat:<25} {stats['correlation']:>12.4f} {stats['p_value']:>12.2e} {status:>12}")

        if suspicious:
            print(f"\n[WARNING] {len(suspicious)} features with correlation > {self.correlation_threshold}:")
            for item in suspicious:
                print(f"  - {item['feature']}: r={item['correlation']:.4f}, p={item['p_value']:.2e}")
                self.suspicious_features.append(item['feature'])
        else:
            print(f"\n[OK] No features exceed correlation threshold of {self.correlation_threshold}")

        return correlations

    def feature_separation_analysis(self, X, y):
        """Check if any single feature perfectly separates classes"""
        print("\n" + "="*60)
        print("FEATURE SEPARATION ANALYSIS")
        print("="*60)
        print("Checks if single features can perfectly separate planets/non-planets")
        print(f"Threshold: Accuracy > {self.perfect_separation_threshold*100}% indicates leakage\n")

        separations = []

        for col in X.columns:
            # Find optimal threshold for binary separation
            thresholds = np.percentile(X[col], [25, 50, 75])

            best_accuracy = 0
            best_threshold = None

            for threshold in thresholds:
                # Predict based on threshold
                pred_above = (X[col] > threshold).astype(int)
                pred_below = (X[col] <= threshold).astype(int)

                acc_above = (pred_above == y).mean()
                acc_below = (pred_below == y).mean()

                if acc_above > best_accuracy:
                    best_accuracy = acc_above
                    best_threshold = threshold

                if acc_below > best_accuracy:
                    best_accuracy = acc_below
                    best_threshold = threshold

            if best_accuracy > self.perfect_separation_threshold:
                separations.append({
                    'feature': col,
                    'accuracy': best_accuracy,
                    'threshold': best_threshold
                })

        if separations:
            separations.sort(key=lambda x: x['accuracy'], reverse=True)
            print(f"[WARNING] {len(separations)} features with accuracy > {self.perfect_separation_threshold*100}%:")
            print(f"\n{'Feature':<25} {'Accuracy':>12} {'Threshold':>15}")
            print("-" * 55)

            for item in separations:
                print(f"{item['feature']:<25} {item['accuracy']:>12.1%} {item['threshold']:>15.6f}")
                if item['feature'] not in self.suspicious_features:
                    self.suspicious_features.append(item['feature'])
        else:
            print(f"[OK] No single feature can separate classes with >{self.perfect_separation_threshold*100}% accuracy")

        return separations

    def distribution_analysis(self, X, y):
        """Analyze feature distributions by class"""
        print("\n" + "="*60)
        print("DISTRIBUTION ANALYSIS")
        print("="*60)
        print("Checks for features with zero/minimal overlap between classes\n")

        zero_overlap = []

        for col in X.columns:
            # Get values for each class
            planet_values = X[y == 1][col]
            non_planet_values = X[y == 0][col]

            # Check for zero overlap (ranges don't intersect)
            planet_min, planet_max = planet_values.min(), planet_values.max()
            non_planet_min, non_planet_max = non_planet_values.min(), non_planet_values.max()

            # Check if ranges overlap
            if planet_max < non_planet_min or non_planet_max < planet_min:
                overlap_percentage = 0.0
            else:
                # Calculate overlap
                overlap_min = max(planet_min, non_planet_min)
                overlap_max = min(planet_max, non_planet_max)
                overlap_range = overlap_max - overlap_min

                total_range = max(planet_max, non_planet_max) - min(planet_min, non_planet_min)
                overlap_percentage = overlap_range / total_range if total_range > 0 else 0

            if overlap_percentage < 0.1:  # Less than 10% overlap
                zero_overlap.append({
                    'feature': col,
                    'overlap_pct': overlap_percentage * 100,
                    'planet_range': (planet_min, planet_max),
                    'non_planet_range': (non_planet_min, non_planet_max)
                })

        if zero_overlap:
            print(f"[WARNING] {len(zero_overlap)} features with <10% range overlap:")
            print(f"\n{'Feature':<25} {'Overlap %':>12} {'Planet Range':>25} {'Non-Planet Range':>25}")
            print("-" * 90)

            for item in zero_overlap:
                p_range = f"[{item['planet_range'][0]:.2f}, {item['planet_range'][1]:.2f}]"
                np_range = f"[{item['non_planet_range'][0]:.2f}, {item['non_planet_range'][1]:.2f}]"
                print(f"{item['feature']:<25} {item['overlap_pct']:>12.1f} {p_range:>25} {np_range:>25}")

                if item['feature'] not in self.suspicious_features:
                    self.suspicious_features.append(item['feature'])
        else:
            print("[OK] All features have reasonable overlap between classes")

        return zero_overlap

    def mutual_information_check(self, X, y):
        """Check for suspiciously high information gain"""
        print("\n" + "="*60)
        print("INFORMATION GAIN ANALYSIS")
        print("="*60)
        print("Analyzes how much information each feature provides about the label\n")

        from sklearn.feature_selection import mutual_info_classif

        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)

        # Create dataframe
        mi_df = pd.DataFrame({
            'feature': X.columns,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)

        # Normalize to 0-1 scale
        max_mi = mi_df['mi_score'].max()
        mi_df['mi_normalized'] = mi_df['mi_score'] / max_mi if max_mi > 0 else 0

        print("Top 10 Features by Mutual Information:")
        print(f"{'Feature':<25} {'MI Score':>12} {'Normalized':>12} {'Status':>12}")
        print("-" * 65)

        for idx, row in mi_df.head(10).iterrows():
            status = "SUSPICIOUS" if row['mi_normalized'] > 0.8 else "OK"
            print(f"{row['feature']:<25} {row['mi_score']:>12.4f} {row['mi_normalized']:>12.4f} {status:>12}")

        # Check for suspiciously high MI
        suspicious_mi = mi_df[mi_df['mi_normalized'] > 0.8]
        if len(suspicious_mi) > 0:
            print(f"\n[WARNING] {len(suspicious_mi)} features with normalized MI > 0.8")
            for idx, row in suspicious_mi.iterrows():
                if row['feature'] not in self.suspicious_features:
                    self.suspicious_features.append(row['feature'])
        else:
            print("\n[OK] No features with suspiciously high mutual information")

        return mi_df

    def summary_report(self):
        """Generate summary report"""
        print("\n" + "="*60)
        print("LEAKAGE DETECTION SUMMARY")
        print("="*60)

        if len(self.suspicious_features) == 0:
            print("\n[OK] NO DATA LEAKAGE DETECTED")
            print("\nAll 77 features passed statistical analysis:")
            print("  - No high correlations with label (|r| < 0.7)")
            print("  - No perfect class separation")
            print("  - Reasonable distribution overlap")
            print("  - Normal mutual information scores")
            print("\nConclusion: Model uses genuine observational features only.")
        else:
            print(f"\n[WARNING] {len(self.suspicious_features)} SUSPICIOUS FEATURES DETECTED")
            print("\nFeatures flagged for potential leakage:")
            for i, feat in enumerate(self.suspicious_features, 1):
                print(f"  {i}. {feat}")

            print("\nRecommendation: Review these features for:")
            print("  - Derived planet properties")
            print("  - Vetting process outputs")
            print("  - Post-classification metadata")

    def analyze(self, filepath):
        """Run complete leakage analysis"""
        print("="*60)
        print("FEATURE LEAKAGE DETECTION - STATISTICAL ANALYSIS")
        print("="*60)

        # Load data
        X, y = self.load_data(filepath)

        # Run all tests
        correlations = self.point_biserial_correlation(X, y)
        separations = self.feature_separation_analysis(X, y)
        distributions = self.distribution_analysis(X, y)
        mi_scores = self.mutual_information_check(X, y)

        # Generate summary
        self.summary_report()

        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)

        return {
            'correlations': correlations,
            'separations': separations,
            'distributions': distributions,
            'mutual_information': mi_scores,
            'suspicious_features': self.suspicious_features
        }

if __name__ == "__main__":
    # Initialize detector
    detector = LeakageDetector(
        correlation_threshold=0.7,
        perfect_separation_threshold=0.95
    )

    # Run analysis on full model features
    results = detector.analyze('preprocessing/kepler_koi_preprocessed.csv')

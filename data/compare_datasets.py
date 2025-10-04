import pandas as pd
import requests
from io import StringIO

# Load manually downloaded dataset (skip comment lines starting with #)
manual_df = pd.read_csv("data/kepler_koi.csv", comment='#')

# Load dataset via NASA Exoplanet Archive TAP query
# Using cumulative table (same as manual download from archive)
tap_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv"
response = requests.get(tap_url)

# Check if response is valid
if response.status_code == 200 and 'xml' not in response.text[:100].lower():
    automatic_df = pd.read_csv(StringIO(response.text))
else:
    print(f"Error fetching data. Status: {response.status_code}")
    print(f"Response preview: {response.text[:200]}")
    automatic_df = None

# Compare datasets
if automatic_df is not None:
    print("Manual download columns:", manual_df.shape[1])
    print("Automatic download columns:", automatic_df.shape[1])
    print("\n" + "="*80)

    # Columns only in manual
    manual_only = set(manual_df.columns) - set(automatic_df.columns)
    if manual_only:
        print(f"\nColumns only in manual download ({len(manual_only)}):")
        print(sorted(manual_only))

    # Columns only in automatic
    automatic_only = set(automatic_df.columns) - set(manual_df.columns)
    if automatic_only:
        print(f"\nColumns only in automatic download ({len(automatic_only)}):")
        print(sorted(automatic_only))

    # Common columns
    common = set(manual_df.columns) & set(automatic_df.columns)
    print(f"\nCommon columns: {len(common)}")

    # Compare shapes
    print(f"\nManual dataset shape: {manual_df.shape}")
    print(f"Automatic dataset shape: {automatic_df.shape}")
else:
    print("\nCould not compare - automatic download failed")

"""
Download Kepler KOI Dataset from NASA Exoplanet Archive
========================================================
This script automatically downloads the Kepler Objects of Interest (KOI)
dataset from the NASA Exoplanet Archive using their TAP (Table Access Protocol) API.

Dataset Source: NASA Exoplanet Archive
Table: cumulative (Kepler KOI cumulative table)
URL: https://exoplanetarchive.ipac.caltech.edu/

Differences between manual and automatic download:
1. Manual download (from archive website):
   - Uses the Kepler TCE (Threshold Crossing Event) table
   - Typically has ~49 core columns selected by user
   - May have comments/metadata in the file
   - Downloaded via web interface with custom column selection

2. Automatic download (via TAP API):
   - Uses the 'cumulative' table with ALL columns (select * from cumulative)
   - Returns ~153 columns with complete metadata
   - No comment lines in CSV
   - Programmatic access via HTTP API

The cumulative table is more comprehensive and includes all available
features for each KOI, making it better for ML training.
"""

import pandas as pd
import requests
from io import StringIO
import os

def download_kepler_koi_dataset(output_path='kepler_koi.csv'):
    """
    Download the Kepler KOI dataset from NASA Exoplanet Archive

    Parameters:
    -----------
    output_path : str
        Path where the CSV file will be saved

    Returns:
    --------
    pd.DataFrame : The downloaded dataset
    """

    print("="*60)
    print("DOWNLOADING KEPLER KOI DATASET")
    print("="*60)

    # NASA Exoplanet Archive TAP query URL
    # This queries the 'cumulative' table which contains all KOI data
    tap_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv"

    print(f"\nFetching data from NASA Exoplanet Archive...")
    print(f"Table: cumulative (Kepler KOI)")
    print(f"Query: SELECT * FROM cumulative")
    print(f"Format: CSV")

    try:
        # Make HTTP request to download data
        response = requests.get(tap_url, timeout=30)

        # Check for errors
        if response.status_code != 200:
            raise Exception(f"HTTP error {response.status_code}")

        # Check if response is XML error message (common for API errors)
        if 'xml' in response.text[:100].lower() or 'error' in response.text[:100].lower():
            print(f"\nAPI Error Response:")
            print(response.text[:500])
            raise Exception("API returned error response")

        # Parse CSV
        df = pd.read_csv(StringIO(response.text))

        print(f"\n[OK] Download successful!")
        print(f"  Rows: {df.shape[0]:,}")
        print(f"  Columns: {df.shape[1]}")

        # Show key columns
        print(f"\nKey columns included:")
        key_cols = ['koi_disposition', 'koi_period', 'koi_depth', 'koi_prad',
                   'koi_teq', 'koi_steff', 'koi_srad']
        for col in key_cols:
            if col in df.columns:
                print(f"  [+] {col}")

        # Save to file
        df.to_csv(output_path, index=False)
        print(f"\n[OK] Dataset saved to: {output_path}")

        # Show disposition counts
        if 'koi_disposition' in df.columns:
            print(f"\nDisposition distribution:")
            for disp, count in df['koi_disposition'].value_counts().items():
                print(f"  {disp}: {count}")

        return df

    except requests.exceptions.Timeout:
        print("\n[ERROR] Request timed out")
        print("The NASA Exoplanet Archive may be slow. Try again later.")
        raise

    except Exception as e:
        print(f"\n[ERROR] Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify NASA Exoplanet Archive is accessible:")
        print("   https://exoplanetarchive.ipac.caltech.edu/")
        print("3. Try the manual download option instead")
        raise

def compare_with_manual_download(auto_df, manual_path):
    """Compare automatically downloaded data with manual download"""

    if not os.path.exists(manual_path):
        print(f"\nManual download file not found at: {manual_path}")
        return

    print("\n" + "="*60)
    print("COMPARING AUTO vs MANUAL DOWNLOAD")
    print("="*60)

    # Load manual download (may have comments)
    manual_df = pd.read_csv(manual_path, comment='#')

    print(f"\nAutomatic download: {auto_df.shape[0]} rows × {auto_df.shape[1]} columns")
    print(f"Manual download:    {manual_df.shape[0]} rows × {manual_df.shape[1]} columns")

    # Column differences
    auto_cols = set(auto_df.columns)
    manual_cols = set(manual_df.columns)

    common = auto_cols & manual_cols
    auto_only = auto_cols - manual_cols
    manual_only = manual_cols - auto_cols

    print(f"\nColumn comparison:")
    print(f"  Common columns: {len(common)}")
    print(f"  Only in automatic: {len(auto_only)}")
    print(f"  Only in manual: {len(manual_only)}")

    if auto_only:
        print(f"\n  Extra columns in automatic download (first 10):")
        for col in list(auto_only)[:10]:
            print(f"    - {col}")
        if len(auto_only) > 10:
            print(f"    ... and {len(auto_only) - 10} more")

if __name__ == "__main__":
    # Download dataset
    df = download_kepler_koi_dataset('kepler_koi.csv')

    # Compare with manual download if it exists
    # (Assuming manual download was saved as kepler_koi_manual.csv)
    # compare_with_manual_download(df, 'kepler_koi_manual.csv')

    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run preprocessing: python ../preprocessing/data_preprocessing.py")
    print("2. Train model: python ../models/train_model.py")

import kagglehub
import pandas as pd
import os
import shutil

# Create data folder if it doesn't exist
os.makedirs('data', exist_ok=True)

print("Downloading Bitcoin sentiment dataset from Kaggle...")
try:
    # Download latest version
    path = kagglehub.dataset_download("imadallal/sentiment-analysis-of-bitcoin-news-2021-2024")
    print(f"Dataset downloaded to: {path}")
    
    # List all files in the downloaded folder
    print("\nFiles in dataset:")
    files = os.listdir(path)
    for file in files:
        print(f"  - {file}")
    
    # Find CSV files
    csv_files = [f for f in files if f.endswith('.csv')]
    
    if csv_files:
        print(f"\nFound {len(csv_files)} CSV file(s)")
        
        # Copy CSV files to your data folder
        for csv_file in csv_files:
            source = os.path.join(path, csv_file)
            destination = os.path.join('data', csv_file)
            shutil.copy(source, destination)
            print(f"✓ Copied {csv_file} to data/")
            
            # Load and explore the CSV
            df = pd.read_csv(destination)
            print(f"\n--- {csv_file} ---")
            print(f"Rows: {len(df)}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"\nFirst 5 rows:")
            print(df.head())
            print(f"\nData types:")
            print(df.dtypes)
            print(f"\nMissing values:")
            print(df.isnull().sum())
            
    else:
        print("No CSV files found in dataset")
        
except Exception as e:
    print(f"Error downloading dataset: {e}")
    print("Make sure you have kagglehub installed: pip install kagglehub")
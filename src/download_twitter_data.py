import kagglehub
import pandas as pd
import os
import shutil

print("Downloading Twitter sentiment dataset...")
try:
    # Download Bitcoin tweets dataset
    path = kagglehub.dataset_download("kaushiksuresh147/bitcoin-tweets")
    print(f"Dataset downloaded to: {path}")
    
    files = os.listdir(path)
    print("\nFiles:", files)
    
    csv_files = [f for f in files if f.endswith('.csv')]
    
    if csv_files:
        for csv_file in csv_files:
            source = os.path.join(path, csv_file)
            destination = os.path.join('data', csv_file)
            shutil.copy(source, destination)
            print(f"✓ Copied {csv_file} to data/")
            
            df = pd.read_csv(destination)
            print(f"\nRows: {len(df)}")
            print(f"Columns: {df.columns.tolist()}")
            print(df.head())
    else:
        print("No CSV files found")
        
except Exception as e:
    print(f"Error: {e}")
    print("Trying alternative dataset...")
    try:
        path = kagglehub.dataset_download("alaix14/bitcoin-tweets-20160101-to-20190329")
        print(f"Alternative dataset downloaded to: {path}")
        # Same processing as above
    except Exception as e2:
        print(f"Alternative also failed: {e2}")
        print("\nManual alternative: Go to Kaggle.com and search 'bitcoin tweets'")

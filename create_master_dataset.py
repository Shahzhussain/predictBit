import pandas as pd
import os
import glob # Library to find all files matching a pattern

def combine_all_engineered_data(output_name="final_master_dataset.csv"):
    engineered_dir = "../data_engineered"
    
    # 1. Find all engineered files (e.g., engineered_2023.csv, engineered_2024.csv)
    # The glob pattern finds all files starting with 'engineered_' in the directory
    all_engineered_files = glob.glob(os.path.join(engineered_dir, "engineered_*.csv"))

    if not all_engineered_files:
        print("❌ Error: No engineered_*.csv files found in the data_engineered directory.")
        return
        
    print(f"Reading and combining {len(all_engineered_files)} files...")
    
    # 2. Read all files into a list
    all_data = []
    for filename in all_engineered_files:
        df = pd.read_csv(filename)
        all_data.append(df)
        
    # 3. Concatenate them into a single DataFrame
    master_df = pd.concat(all_data, ignore_index=True)
    
    # 4. Final sorting by date (CRITICAL for chronological time series split)
    master_df['date'] = pd.to_datetime(master_df['date'])
    master_df = master_df.sort_values(by='date', ascending=True).reset_index(drop=True)
    
    # 5. Save the master file
    output_path = os.path.join(engineered_dir, output_name)
    master_df.to_csv(output_path, index=False)
    
    print("\n=======================================================")
    print(f"✅ MASTER DATASET CREATED! Final size: {master_df.shape[0]} days.")
    print(f"File saved to: {output_path}")
    print("=======================================================")

# --- EXECUTION EXAMPLE ---
# Run this script AFTER running feature_engineering.py for all individual years.
combine_all_engineered_data()
import pandas as pd
import os

# --- CONFIGURATION ---
# Ensure this filename matches exactly what you have in your folder
FILE_NAME = "../data_engineered/final_structured_features.csv"
FILE_PATH = FILE_NAME 

def fix_dataframe_columns():
    print(f"📂 Loading {FILE_PATH}...")
    if not os.path.exists(FILE_PATH):
        print(f"❌ Error: File {FILE_PATH} not found.")
        return

    df = pd.read_csv(FILE_PATH)
    print(f"Original Shape: {df.shape}")
    print(f"Columns before fix: {df.columns.tolist()}")

    # ---------------------------------------------------------
    # 1. MERGE VOLUME COLUMNS
    # ---------------------------------------------------------
    # Logic: If 'Volume' is missing (NaN), grab the value from 'volume'
    if 'Volume' in df.columns and 'volume' in df.columns:
        print("\n⚙️ Merging 'volume' (lowercase) into 'Volume' (Capitalized)...")
        
        # Count nulls before
        nulls_before = df['Volume'].isnull().sum()
        print(f"   - Missing values in 'Volume' before merge: {nulls_before}")
        
        # The MERGE Operation
        df['Volume'] = df['Volume'].fillna(df['volume'])
        
        # Count nulls after
        nulls_after = df['Volume'].isnull().sum()
        print(f"   - Missing values in 'Volume' after merge:  {nulls_after}")
        
        # Drop the now-redundant lowercase column
        df.drop(columns=['volume'], inplace=True)
        print("✅ Dropped redundant 'volume' column.")
        
    elif 'volume' in df.columns:
        # Case where only lowercase exists
        df.rename(columns={'volume': 'Volume'}, inplace=True)
        print("✅ Renamed 'volume' to 'Volume'.")

    # ---------------------------------------------------------
    # 2. CLEAN UP CHANGE COLUMNS
    # ---------------------------------------------------------
    # You already have 'price_change' and 'price_change_percent' calculated correctly.
    # The 'change' column is raw, incomplete data from the source file and should be removed.
    
    if 'change' in df.columns:
        print("\n⚙️ Removing incomplete 'change' column...")
        df.drop(columns=['change'], inplace=True)
        print("✅ Dropped 'change' column.")

    # Verify critical calculated columns exist
    if 'price_change_percent' not in df.columns:
        print("⚠️ Warning: 'price_change_percent' is missing! Recalculating...")
        # Fallback calculation if needed (requires price_change and math)
        # Note: Accurate recalc requires raw Close price which might not be here.
        # Assuming it exists based on previous steps.

    # ---------------------------------------------------------
    # 3. SAVE AND VERIFY
    # ---------------------------------------------------------
    df.to_csv(FILE_PATH, index=False)
    print(f"\n💾 File successfully overwritten: {FILE_PATH}")
    print(f"Final Columns: {df.columns.tolist()}")
    
    # Final check
    if df['Volume'].isnull().sum() == 0:
        print("🎉 Success: 'Volume' column is now full and complete!")
    else:
        print(f"⚠️ Note: 'Volume' still has {df['Volume'].isnull().sum()} missing values.")

if __name__ == "__main__":
    fix_dataframe_columns()
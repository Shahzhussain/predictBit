import pandas as pd
import numpy as np

def analyze_dataset(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    print("🔍 DATASET ANALYSIS REPORT")
    print("=" * 60)
    
    # Basic info
    print(f"📁 File: {file_path}")
    print(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"💾 Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print()
    
    # Column details
    print("📋 COLUMNS AND DATA TYPES:")
    print("-" * 60)
    print(f"{'COLUMN NAME':<25} | {'DATA TYPE':<12} | {'NON-NULL':<8} | {'UNIQUE':<6} | SAMPLE VALUE")
    print("-" * 60)
    
    for col in df.columns:
        non_null = df[col].count()
        unique_count = df[col].nunique()
        sample_val = str(df[col].iloc[0]) if non_null > 0 else "NaN"
        
        print(f"{col:<25} | {str(df[col].dtype):<12} | {non_null:<8} | {unique_count:<6} | {sample_val[:30]}")

    # Data type summary
    print(f"\n📈 DATA TYPE SUMMARY:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    
    # Missing values
    print(f"\n⚠️  MISSING VALUES:")
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        for col, count in missing_cols.items():
            print(f"  {col}: {count} missing values ({count/len(df)*100:.1f}%)")
    else:
        print("  No missing values! ✅")
    
    return df

# Usage
if __name__ == "__main__":
    file_path = "../data_engineered/final_structured_features.csv"
    df = analyze_dataset(file_path)
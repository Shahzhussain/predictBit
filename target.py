import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# =======================================================
# CONFIGURATION
# =======================================================
ENGINEERED_DIR = "../data_engineered"
INPUT_FILENAME = "final_master_dataset.csv" 
INPUT_PATH = os.path.join(ENGINEERED_DIR, INPUT_FILENAME)

# Outputs
OUTPUT_PATH_CSV = os.path.join(ENGINEERED_DIR, "final_structured_features.csv") 
OUTPUT_DIR_NPY = os.path.join(ENGINEERED_DIR, "npy_data")
os.makedirs(OUTPUT_DIR_NPY, exist_ok=True)

# Settings
LOOK_BACK_DAYS = 10    
TEST_SIZE_RATIO = 0.20 

# =======================================================
# 1. CLEANING & TARGET CREATION
# =======================================================
def prepare_data(df):
    # --- FIX VOLUME ISSUES (The Logic from your fix script) ---
    if 'Volume' in df.columns and 'volume' in df.columns:
        df['Volume'] = df['Volume'].fillna(df['volume'])
        print("✅ Merged 'volume' into 'Volume'.")
    elif 'volume' in df.columns:
        df.rename(columns={'volume': 'Volume'}, inplace=True)
        
    # Drop redundant columns
    cols_to_drop = ['volume', 'change'] 
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # --- DROP NANs ---
    df = df.dropna().reset_index(drop=True)
    
    # --- CREATE TARGETS (Y) ---
    # Shift(-1) means we are pulling TOMORROW's value to TODAY's row
    df['Y_reg'] = df['close'].shift(-1)
    df['Y_class'] = (df['close'].shift(-1) > df['close']).astype(int)

    # Drop the last row (NaN targets)
    df = df.iloc[:-1].copy()
    
    return df

# =======================================================
# 2. SEQUENCE CREATION
# =======================================================
def create_sequences(data, targets_reg, targets_class, look_back):
    X, y_r, y_c = [], [], []
    
    for i in range(len(data) - look_back):
        # Feature window
        X.append(data[i:(i + look_back)])
        # Targets
        y_r.append(targets_reg[i + look_back])
        y_c.append(targets_class[i + look_back])
        
    return np.array(X), np.array(y_r), np.array(y_c)

# =======================================================
# 3. MAIN EXECUTION
# =======================================================
def run_structuring():
    print(f"📥 Loading master dataset from: {INPUT_PATH}")
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Error: File not found at {INPUT_PATH}")
        return

    df = pd.read_csv(INPUT_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Apply the fixes and create targets
    df = prepare_data(df)

    # --- Define Features (X) ---
    exclude_cols = ['date', 'Y_reg', 'Y_class', 'open', 'high', 'low', 'close']
    # Note: We KEEP 'Volume' now because it is fixed and valuable!
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    print(f"📝 Features Selected ({len(feature_cols)}): {feature_cols}")

    # --- Normalization ---
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(df[feature_cols])
    
    # --- SAVE ML DATA (CSV) ---
    df_ml = pd.DataFrame(X_scaled, columns=feature_cols)
    df_ml['date'] = df['date']
    df_ml['Y_reg'] = df['Y_reg']
    df_ml['Y_class'] = df['Y_class']
    df_ml.to_csv(OUTPUT_PATH_CSV, index=False)
    print(f"✅ Clean ML Data Saved: {OUTPUT_PATH_CSV}")

    # --- SAVE DL DATA (NPY) ---
    print("⚙️ Creating 3D sequences from CLEAN data...")
    
    split_idx = int(len(X_scaled) * (1 - TEST_SIZE_RATIO))
    seq_split_idx = split_idx - LOOK_BACK_DAYS
    
    X_seq, Y_reg_seq, Y_class_seq = create_sequences(
        X_scaled, df['Y_reg'].values, df['Y_class'].values, LOOK_BACK_DAYS
    )
    
    # Save arrays
    np.save(os.path.join(OUTPUT_DIR_NPY, "X_train.npy"), X_seq[:seq_split_idx])
    np.save(os.path.join(OUTPUT_DIR_NPY, "X_test.npy"), X_seq[seq_split_idx:])
    np.save(os.path.join(OUTPUT_DIR_NPY, "Y_train_class.npy"), Y_class_seq[:seq_split_idx])
    np.save(os.path.join(OUTPUT_DIR_NPY, "Y_test_class.npy"), Y_class_seq[seq_split_idx:])
    
    print(f"✅ Clean DL Arrays Saved to: {OUTPUT_DIR_NPY}")

if __name__ == "__main__":
    run_structuring()
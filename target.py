import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split # Used for splitting (though chronological split is preferred here)


# --- CONFIGURATION ---
ENGINEERED_DIR = "../data_engineered"
# Assuming you named your master file 'final_master_dataset.csv'
INPUT_FILENAME = "final_master_dataset.csv" 
INPUT_PATH = os.path.join(ENGINEERED_DIR, INPUT_FILENAME)
# Output path for the final structured data used by ML models
OUTPUT_PATH = os.path.join(ENGINEERED_DIR, "final_structured_features.csv") 

# Define the look-back window (N days of history for DL models)
LOOK_BACK_DAYS = 10 
TEST_SIZE_RATIO = 0.20 # 20% for testing (chronological split)

# =======================================================
# 1. LOAD, CLEAN, AND CREATE TARGETS
# =======================================================

def prepare_data(df):

    # --- 1. Create Target Variables (Y) ---

    # Regression Target (Y_reg): Next day's Close price
    df['Y_reg'] = df['close'].shift(-1)

    # Classification Target (Y_class): Next day's Price Direction (UP=1, DOWN/FLAT=0)
    # We compare the NEXT day's close price to the current day's close price
    df['Y_class'] = (df['close'].shift(-1) > df['close']).astype(int)

    # Drop rows with NaN in targets (expected to be only last row after shift)
    df = df.dropna(subset=['Y_reg', 'Y_class']).reset_index(drop=True)

    # --- 2. Create Core Features (X) ---

    # We will use the existing NLP features (avg_sentiment, NER_counts, Topic_Probs)
    # and the existing technical features (price_ma, volatility) as our base features.

    return df


# =======================================================
# 2. SEQUENCE/WINDOW CREATION (FOR LSTM/INFORMER)
# =======================================================

def create_sequences(data, look_back):
    """
    Creates 3D arrays (samples, timesteps, features) from 2D data.
    Used for LSTM, GRU, and Informer models.
    """
    X, Y_reg, Y_class = [], [], []
    
    # Iterate through the data, creating a window of 'look_back' size
    for i in range(len(data) - look_back):
        # X: The window of features from t to t + look_back - 1
        X_window = data.iloc[i:(i + look_back)]
        X.append(X_window.values)
        
        # Y: The targets are at time t + look_back (the next step after the window)
        Y_reg.append(data['Y_reg'].iloc[i + look_back])
        Y_class.append(data['Y_class'].iloc[i + look_back])
        
    return np.array(X), np.array(Y_reg), np.array(Y_class)


# =======================================================
# 3. MAIN EXECUTION & SPLIT
# =======================================================

def run_structuring():
    print(f"📥 Loading engineered master dataset from {INPUT_PATH}...")
    try:
        df = pd.read_csv(INPUT_PATH)
    except FileNotFoundError:
        print("❌ Master file not found. Ensure you ran the concatenation script.")
        return

    print(f"Initial dataframe shape: {df.shape}")
    print(f"Initial dataframe columns: {df.columns.tolist()}")

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date', ascending=True)

    df = prepare_data(df)
    print(f"Dataframe shape after prepare_data(): {df.shape}")
    print(f"Dataframe columns after prepare_data(): {df.columns.tolist()}")

    # --- Feature Selection and Normalization ---
    # Select all features created in the merging/engineering step, excluding the targets
    feature_cols = [col for col in df.columns if col not in ['date', 'Y_reg', 'Y_class', 'open', 'high', 'low', 'close']]
    print(f"Feature columns selected for scaling: {feature_cols}")

    df_features = df[feature_cols].copy()
    print(f"Feature dataframe shape before scaling: {df_features.shape}")

    # Check for missing values before scaling
    missing_features_rows = df_features.isnull().any(axis=1).sum()
    print(f"Number of rows with missing features before scaling: {missing_features_rows}")

    # Drop rows with missing features (optional: could also fill or impute)
    df_features.dropna(inplace=True)
    df = df.loc[df_features.index]  # Align df with filtered features

    print(f"Dataframe shape after dropping rows with missing features: {df.shape}")

    # Normalization (CRITICAL for DL models)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_features_scaled = scaler.fit_transform(df_features)
    df_features_scaled = pd.DataFrame(df_features_scaled, columns=feature_cols, index=df_features.index)

    # --- CHRONOLOGICAL TRAIN-TEST SPLIT (MANDATORY) ---
    
    # 1. Determine the split point
    split_index = int(len(df_features_scaled) * (1 - TEST_SIZE_RATIO))
    print(f"Total rows: {len(df_features_scaled)}. Splitting chronologically at row {split_index}.")
    
    # 2. Split the dataframes based on the index
    X_train_df = df_features_scaled.iloc[:split_index]
    X_test_df = df_features_scaled.iloc[split_index:]
    
    Y_train_reg = df['Y_reg'].iloc[:split_index]
    Y_test_reg = df['Y_reg'].iloc[split_index:]
    
    Y_train_class = df['Y_class'].iloc[:split_index]
    Y_test_class = df['Y_class'].iloc[split_index:]


    # 3. Create Sequences for DL/Transformer Models (3D Arrays)
    print("⚙️ Creating 3D sequences for LSTM/Informer...")

    # Combine scaled features with target columns for sequence creation
    data_for_sequences = df_features_scaled.copy()
    data_for_sequences['Y_reg'] = df['Y_reg']
    data_for_sequences['Y_class'] = df['Y_class']

    # NOTE: We use the full scaled data with targets for sequence creation
    X_sequences, Y_sequences_reg, Y_sequences_class = create_sequences(data_for_sequences, LOOK_BACK_DAYS)

    # The sequence split must account for the LOOK_BACK_DAYS shift
    X_train_seq = X_sequences[:split_index - LOOK_BACK_DAYS]
    X_test_seq = X_sequences[split_index - LOOK_BACK_DAYS:]
    
    # Y targets for the DL models must also be shifted to align with the sequence windows
    Y_train_reg_seq = Y_sequences_reg[:split_index - LOOK_BACK_DAYS]
    Y_test_reg_seq = Y_sequences_reg[split_index - LOOK_BACK_DAYS:]


    print("\n=============================================")
    print("✅ Data Structuring Complete. Ready for Models.")
    print("=============================================")
    print(f"Train Size (DL/Transformer): {X_train_seq.shape}")
    print(f"Test Size (DL/Transformer):  {X_test_seq.shape}")
    print(f"Train Target Y_class (ML/ARIMAX): {Y_train_class.shape}")
    
    # --- FINAL STEP: SAVE THE 1D DATASET FOR ML/ARIMAX ---
    # This 1D dataset is the direct input for XGBoost and ARIMAX (after differencing)
    df_final_ml = pd.DataFrame({
        'date': df['date'], # Use all valid rows after filtering
        'Y_class': df['Y_class'],
        'Y_reg': df['Y_reg'],
    })

    # Check for missing values before saving
    missing_before_save = df_final_ml.isnull().any(axis=1).sum()
    print(f"Number of rows with missing values before saving: {missing_before_save}")

    df_final_ml = pd.concat([df_final_ml.reset_index(drop=True), df_features_scaled.reset_index(drop=True)], axis=1)

    # Drop rows with any missing values if any remain
    missing_after_concat = df_final_ml.isnull().any(axis=1).sum()
    if missing_after_concat > 0:
        print(f"Dropping {missing_after_concat} rows with missing values after concatenation.")
        df_final_ml.dropna(inplace=True)

    df_final_ml.to_csv(OUTPUT_PATH, index=False)
    print(f"\n💾 1D Feature File (for ARIMAX/XGBoost) saved to: {OUTPUT_PATH}")
    
    # NOTE: The 3D arrays (X_train_seq, Y_train_reg_seq, etc.) must be passed directly
    # to your LSTM/GRU/Informer training scripts (which you will write next).


run_structuring()
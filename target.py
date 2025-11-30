# target_safe.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

BASE = ".."
ENG_DIR = os.path.join(BASE, "data_engineered")
INPUT = os.path.join(ENG_DIR, "final_master_dataset.csv")
OUTPUT_CSV = os.path.join(ENG_DIR, "final_structured_features.csv")
NPY_DIR = os.path.join(ENG_DIR, "npy_data")
os.makedirs(NPY_DIR, exist_ok=True)

LOOK_BACK = 10
TEST_RATIO = 0.20

def load_master():
    if not os.path.exists(INPUT):
        raise FileNotFoundError(INPUT)
    df = pd.read_csv(INPUT)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def prepare_targets(df):
    # Create next-day regression target and classification target
    df["Y_reg"] = df["close"].shift(-1)           # tomorrow's close
    df["Y_class"] = (df["close"].shift(-1) > df["close"]).astype(int)
    # drop last row (no target)
    df = df.iloc[:-1].copy().reset_index(drop=True)
    return df

def build_features_and_save(df):
    # Exclude raw price columns from features to avoid leakage via handcrafted features.
    exclude = ["date", "open", "high", "low", "close", "Y_reg", "Y_class"]
    feature_cols = [c for c in df.columns if c not in exclude]
    print("Feature columns:", feature_cols)

    # Fill any remaining NaNs conservatively (do NOT use future info)
    X = df[feature_cols].fillna(method="ffill").fillna(method="bfill").fillna(0)

    # Scale using train stats only
    split_idx = int(len(X) * (1 - TEST_RATIO))
    scaler = MinMaxScaler()
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    scaler.fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

    X_scaled = pd.concat([X_train_scaled, X_test_scaled]).sort_index()

    # produce final structured CSV: features scaled + date + targets
    df_out = pd.concat([X_scaled.reset_index(drop=True),
                        df[["date", "Y_reg", "Y_class"]].reset_index(drop=True)], axis=1)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print("Saved final structured features:", OUTPUT_CSV)

    # Create sequences for DL
    arr = X_scaled.values
    y_reg = df["Y_reg"].values
    y_class = df["Y_class"].values

    X_seq = []
    Y_reg_seq = []
    Y_class_seq = []
    for i in range(len(arr) - LOOK_BACK):
        X_seq.append(arr[i:i+LOOK_BACK])
        Y_reg_seq.append(y_reg[i+LOOK_BACK])
        Y_class_seq.append(y_class[i+LOOK_BACK])

    X_seq = np.array(X_seq)
    Y_reg_seq = np.array(Y_reg_seq)
    Y_class_seq = np.array(Y_class_seq)

    # split chronologically on sequences
    seq_split = split_idx - LOOK_BACK
    if seq_split <= 0:
        raise ValueError("Not enough data for LOOK_BACK with given TEST_RATIO.")
    np.save(os.path.join(NPY_DIR, "X_train.npy"), X_seq[:seq_split])
    np.save(os.path.join(NPY_DIR, "X_test.npy"),  X_seq[seq_split:])
    np.save(os.path.join(NPY_DIR, "Y_train_reg.npy"), Y_reg_seq[:seq_split])
    np.save(os.path.join(NPY_DIR, "Y_test_reg.npy"),  Y_reg_seq[seq_split:])
    np.save(os.path.join(NPY_DIR, "Y_train_class.npy"), Y_class_seq[:seq_split])
    np.save(os.path.join(NPY_DIR, "Y_test_class.npy"),  Y_class_seq[seq_split:])
    print("Saved NPY sequences to:", NPY_DIR)
    print("Shapes X_train_seq:", X_seq[:seq_split].shape, "X_test_seq:", X_seq[seq_split:].shape)

if __name__ == "__main__":
    df = load_master()
    df = prepare_targets(df)
    build_features_and_save(df)

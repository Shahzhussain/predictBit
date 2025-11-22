import pandas as pd
import numpy as np
import os
import re

# ---------------------------------------------------
# Convert price columns to numeric
# ---------------------------------------------------
def convert_price_columns(df):

    price_cols = ["open", "high", "low", "close"]

    for col in price_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(" ", "", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------
# Convert volume column (e.g., 74.85K, 1.2M)
# ---------------------------------------------------
def convert_volume(df):
    def convert(v):
        v = str(v).strip().upper()
        if v.endswith("K"):
            return float(v[:-1]) * 1000
        elif v.endswith("M"):
            return float(v[:-1]) * 1_000_000
        else:
            # remove commas if plain number
            v = v.replace(",", "")
            return pd.to_numeric(v, errors="coerce")

    df["volume"] = df["volume"].apply(convert)
    return df


# ---------------------------------------------------
# Load merged dataset
# ---------------------------------------------------
def load_merged(file_name):
    base_path = ".."
    path = os.path.join(base_path, "data_merged", file_name)
    print(f"📥 Loading merged dataset: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()  # Strip whitespace characters from columns

    # Drop rows where essential columns are missing to prevent NaNs downstream
    essential_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in essential_cols:
        if col in df.columns:
            before_drop = df.shape[0]
            df = df[df[col].notnull()]
            after_drop = df.shape[0]
            print(f"Dropped {before_drop - after_drop} rows with missing '{col}'")

    df["date"] = pd.to_datetime(df["date"])

    # Convert prices to numeric
    df = convert_price_columns(df)

    # Convert volume to numeric
    if "volume" in df.columns:
        df = convert_volume(df)

    return df


# ---------------------------------------------------
# Add price features
# ---------------------------------------------------
def add_price_features(df):
    df["price_change"] = df["close"] - df["close"].shift(1)
    df["price_change_percent"] = df["price_change"] / df["close"].shift(1)
    df["price_direction"] = (df["close"].shift(-1) > df["close"]).astype(int)
    return df


# ---------------------------------------------------
# Sentiment–price relationship
# ---------------------------------------------------
def add_sentiment_impact_features(df):

    df["sentiment_negative"] = (df["avg_sentiment"] < 0).astype(int)
    df["sentiment_positive"] = (df["avg_sentiment"] > 0).astype(int)
    df["price_down"] = (df["price_change"] < 0).astype(int)
    df["price_up"] = (df["price_change"] > 0).astype(int)

    df["sentiment_price_agreement"] = (
        (df["sentiment_negative"] & df["price_down"]) |
        (df["sentiment_positive"] & df["price_up"])
    ).astype(int)

    df["sentiment_price_impact"] = df["avg_sentiment"] * df["price_change_percent"]

    return df


# ---------------------------------------------------
# Rolling window features
# ---------------------------------------------------
def add_rolling_features(df):

    df["sentiment_ma3"] = df["avg_sentiment"].rolling(3).mean()
    df["sentiment_ma7"] = df["avg_sentiment"].rolling(7).mean()

    df["price_ma3"] = df["close"].rolling(3).mean()
    df["price_ma7"] = df["close"].rolling(7).mean()

    df["volatility7"] = df["close"].rolling(7).std()

    return df


# ---------------------------------------------------
# Save engineered dataset
# ---------------------------------------------------
def save_engineered(df, output_name):
    base_path = ".."
    out_dir = os.path.join(base_path, "data_engineered")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, output_name)
    df.to_csv(out_path, index=False)

    print(f"✅ Engineered dataset saved:\n{out_path}")


# ---------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------
def run_feature_engineering(merged_file, output_file):

    df = load_merged(merged_file)

    df = add_price_features(df)
    df = add_sentiment_impact_features(df)
    df = add_rolling_features(df)

    df = df.dropna()

    save_engineered(df, output_file)


# ---------------------------------------------------
# CALL HERE
# ---------------------------------------------------
# Example:
run_feature_engineering("merged_2024.csv", "engineered_2024.csv")

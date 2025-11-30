# feature_engineering_safe.py
import pandas as pd
import numpy as np
import os

BASE = ".."  # one folder up
MERGED_DIR = os.path.join(BASE, "data_merged")
OUT_DIR = os.path.join(BASE, "data_engineered")
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Helpers for numeric conversion
# -------------------------
def to_numeric_prices(df, cols=["open","high","low","close"]):
    for c in cols:
        if c in df.columns:
            df[c] = (
                df[c].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace(" ", "", regex=False)
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def convert_volume_col(df, col="volume"):
    if col not in df.columns:
        return df
    def conv(v):
        v = str(v).strip().upper()
        if v.endswith("K"):
            return float(v[:-1]) * 1_000
        if v.endswith("M"):
            return float(v[:-1]) * 1_000_000
        # remove commas
        v = v.replace(",", "")
        try:
            return float(v)
        except:
            return np.nan
    df[col] = df[col].apply(conv)
    return df

def clean_percent_col(df, col="change_percent"):
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace("%","",regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")/100.0
    return df

# -------------------------
# Technical indicators
# -------------------------
def SMA(series, window):
    return series.rolling(window, min_periods=1).mean()

def EMA(series, window):
    return series.ewm(span=window, adjust=False).mean()

def RSI(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window, min_periods=1).mean()
    ma_down = down.rolling(window, min_periods=1).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # neutral when insufficient data

def MACD(series, fast=12, slow=26, signal=9):
    ema_fast = EMA(series, fast)
    ema_slow = EMA(series, slow)
    macd = ema_fast - ema_slow
    macd_signal = EMA(macd, signal)
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def bollinger_bands(series, window=20, n_std=2):
    ma = series.rolling(window, min_periods=1).mean()
    std = series.rolling(window, min_periods=1).std().fillna(0)
    upper = ma + (n_std * std)
    lower = ma - (n_std * std)
    return upper, lower

# -------------------------
# Per-file safe feature engineering
# -------------------------
def process_year(merged_filename, out_filename):
    in_path = os.path.join(MERGED_DIR, merged_filename)
    if not os.path.exists(in_path):
        print(f"File not found: {in_path}")
        return

    print("Processing:", in_path)
    df = pd.read_csv(in_path)
    df.columns = df.columns.str.strip()

    # ensure date exists and convert
    if "date" not in df.columns:
        raise ValueError("date column missing in merged file")
    df["date"] = pd.to_datetime(df["date"])

    # Convert numeric columns
    df = to_numeric_prices(df, ["open","high","low","close"])
    df = convert_volume_col(df, "volume")
    df = clean_percent_col(df, "change_percent")

    # Sort by date (should already be per-year but safe)
    df = df.sort_values("date").reset_index(drop=True)

    # -------------------------
    # Sentiment numeric mapping (safe)
    # -------------------------
    if "sentiment_label" in df.columns:
        mapping = {"positive":1, "neutral":0, "negative":-1}
        df["sentiment_numeric"] = df["sentiment_label"].map(mapping).fillna(0)
    else:
        # if your processed file already has avg_sentiment, keep it
        if "avg_sentiment" not in df.columns:
            df["sentiment_numeric"] = 0.0

    # If sentiment_score exists, keep it as confidence feature (safe)
    # Compute aggregated daily sentiment only if original tweets present per-row
    # But merged files should already be daily; assume avg_sentiment exists or use sentiment_numeric
    if "avg_sentiment" not in df.columns:
        df["avg_sentiment"] = df.get("sentiment_numeric", 0.0)

    # -------------------------
    # Price features (safe: use only past & present)
    # -------------------------
    # price_change = close(t) - close(t-1)
    df["price_change"] = df["close"] - df["close"].shift(1)
    df["price_change_percent"] = df["price_change"] / df["close"].shift(1)

    # NOTE: DO NOT compute price_direction (future) here
    # we will compute targets later in target.py (using shift(-1))

    # -------------------------
    # Technical indicators (computed per-year safely)
    # -------------------------
    df["sma_3"] = SMA(df["close"], 3)
    df["sma_7"] = SMA(df["close"], 7)
    df["ema_7"] = EMA(df["close"], 7)
    df["ema_14"] = EMA(df["close"], 14)
    df["rsi_14"] = RSI(df["close"], 14)
    macd, macd_sig, macd_hist = MACD(df["close"])
    df["macd"] = macd
    df["macd_signal"] = macd_sig
    df["macd_hist"] = macd_hist
    df["bb_upper"], df["bb_lower"] = bollinger_bands(df["close"], window=20, n_std=2)

    # volatility (7-day std)
    df["volatility7"] = df["close"].rolling(7, min_periods=1).std()

    # -------------------------
    # Sentiment rolling (safe per-year)
    # -------------------------
    df["sentiment_ma3"] = df["avg_sentiment"].rolling(3, min_periods=1).mean()
    df["sentiment_ma7"] = df["avg_sentiment"].rolling(7, min_periods=1).mean()

    # -------------------------
    # Safe fill for short windows (do not leak, but fill with prior/neutral)
    # -------------------------
    # Use forward fill for technical indicators where appropriate, then backfill minimal
    df[["price_change", "price_change_percent", "sma_3", "sma_7",
        "ema_7", "ema_14", "rsi_14", "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_lower", "volatility7", "sentiment_ma3", "sentiment_ma7"]] = \
        df[["price_change", "price_change_percent", "sma_3", "sma_7",
            "ema_7", "ema_14", "rsi_14", "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_lower", "volatility7", "sentiment_ma3", "sentiment_ma7"]].ffill().bfill().fillna(0)

    # -------------------------
    # Keep only SAFE columns (no future info)
    # -------------------------
    # Choose the safe features to export
    safe_cols = [
        "date", "open", "high", "low", "close", "volume", "change_percent",
        "avg_sentiment", "sentiment_numeric", "price_change", "price_change_percent",
        "sma_3", "sma_7", "ema_7", "ema_14", "rsi_14",
        "macd", "macd_signal", "macd_hist", "bb_upper", "bb_lower",
        "volatility7", "sentiment_ma3", "sentiment_ma7"
    ]

    # Keep whatever of safe_cols exists in df
    cols_to_save = [c for c in safe_cols if c in df.columns]
    df_out = df[cols_to_save].copy()

    # Save
    out_path = os.path.join(OUT_DIR, out_filename)
    df_out.to_csv(out_path, index=False)
    print("Saved engineered (safe):", out_path)

# -------------------------
# Quick CLI: process all merged_*.csv in data_merged
# -------------------------
if __name__ == "__main__":
    files = [f for f in os.listdir(MERGED_DIR) if f.lower().startswith("merged_") and f.lower().endswith(".csv")]
    if not files:
        print("No merged_*.csv found in", MERGED_DIR)
    for f in files:
        year = f.split("_")[-1].split(".")[0]
        out_name = f"engineered_{year}.csv"
        process_year(f, out_name)

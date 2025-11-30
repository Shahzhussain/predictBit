import pandas as pd
import numpy as np
import os
import sys

# ---------------------------------------------------
# UTILITY: Aggregate tweet sentiment per day
# ---------------------------------------------------
def aggregate_tweet_sentiment(df_tweets):
    # Ensure date column is datetime
    df_tweets['date'] = pd.to_datetime(df_tweets['date']).dt.date

    # Map sentiment labels to numeric values
    label_map = {
        'positive': 1,
        'neutral': 0,
        'negative': -1
    }

    df_tweets["sentiment_numeric"] = df_tweets["sentiment_label"].map(label_map)

    # Group by date
    daily = df_tweets.groupby('date').agg(
        avg_sentiment=("sentiment_score", "mean"),
        pos_count=("sentiment_label", lambda x: (x == "positive").sum()),
        neg_count=("sentiment_label", lambda x: (x == "negative").sum()),
        neu_count=("sentiment_label", lambda x: (x == "neutral").sum()),
        total_tweets=("sentiment_label", "count")
    ).reset_index()

    return daily


# ---------------------------------------------------
# MAIN MERGING FUNCTION
# ---------------------------------------------------
def merge_price_and_tweets(price_file, tweet_file, output_name):
    def load_file(filepath, is_price_file=False):
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.csv':
            return pd.read_csv(filepath)
        elif ext in ['.xls', '.xlsx']:
            if is_price_file:
                # For price files, load excel with header row (no manual column assignment)
                df = pd.read_excel(filepath, header=0)
                return df
            else:
                # For tweet files, load excel with header
                return pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    # Correct folder paths
    base_path = ".."  # one folder out
    price_path = os.path.join(base_path, "data_prices", price_file)
    tweet_path = os.path.join(base_path, "data_tweets", tweet_file)
    output_dir = os.path.join(base_path, "data_merged")
    os.makedirs(output_dir, exist_ok=True)

    try:
        print(f"📥 Loading price data: {price_path}")
        df_price = load_file(price_path, is_price_file=True)
    except Exception as e:
        print(f"❌ Failed to load price data file: {e}")
        return

    try:
        print(f"📥 Loading tweet data: {tweet_path}")
        df_tweets = load_file(tweet_path, is_price_file=False)
    except Exception as e:
        print(f"❌ Failed to load tweet data file: {e}")
        return

    # Clean dates
    df_price["date"] = pd.to_datetime(df_price["date"]).dt.date
    df_tweets["date"] = pd.to_datetime(df_tweets["date"]).dt.date

    print("📊 Aggregating tweets by date...")
    daily_sentiment = aggregate_tweet_sentiment(df_tweets)

    print("🔗 Merging price + sentiment data...")
    merged = df_price.merge(daily_sentiment, on="date", how="left")

    # Fill missing sentiment for days with 0 tweets
    merged["avg_sentiment"].fillna(0, inplace=True)
    merged["pos_count"].fillna(0, inplace=True)
    merged["neg_count"].fillna(0, inplace=True)
    merged["neu_count"].fillna(0, inplace=True)
    merged["total_tweets"].fillna(0, inplace=True)

    # Sorted By Date (Ascending)
    merged = merged.sort_values(by="date", ascending=True)

    # Save output
    output_path = os.path.join(output_dir, output_name)
    merged.to_csv(output_path, index=False)

    print(f"✅ Merged dataset saved at:\n{output_path}")


# ---------------------------------------------------
# MAIN EXECUTION GUARD
# ---------------------------------------------------
if __name__ == "__main__":
    # Hardcoded filenames
    price_file = "prices_2022.csv"
    tweet_file = "tweets_2022.csv"
    output_name = "merged_2022.csv"
    merge_price_and_tweets(price_file, tweet_file, output_name)

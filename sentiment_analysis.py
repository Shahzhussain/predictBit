import pandas as pd
from transformers import pipeline
import torch
import os
import time

def analyze_csv_batch(csv_filename, sample_size=None):
    # INPUT path
    input_path = os.path.join("..", "data", "processed", csv_filename)
    
    # OUTPUT path
    output_dir = os.path.join("..", "data", "sentiment")
    os.makedirs(output_dir, exist_ok=True)

    print(f"📥 Reading input file: {input_path}")
    print(f"📊 File size: {os.path.getsize(input_path)/1024/1024:.1f} MB")

    df = pd.read_csv(input_path)
    print(f"✅ Loaded DataFrame: {df.shape}")

    # Take sample if specified, otherwise process ALL data
    if sample_size and sample_size < len(df):
        df = df.head(sample_size)
        print(f"🔬 Using sample of {len(df)} rows")
    else:
        print(f"🚀 Processing ALL {len(df)} rows")

    # Find text column
    text_col = None
    for col in df.columns:
        if col.lower() in ["text", "tweet", "content", "text_lemmatized"]:
            text_col = col
            break

    if text_col is None:
        print("❌ No text column found. Available columns:", df.columns.tolist())
        return

    print(f"📝 Using text column: '{text_col}'")
    print(f"🚀 Setting up sentiment analysis pipeline...")

    # Check for GPU
    device = 0 if torch.cuda.is_available() else -1
    if torch.cuda.is_available():
        print("🎯 Using GPU acceleration!")
    else:
        print("⚡ Using CPU (GPU recommended for large datasets)")

    # Initialize pipeline with batch processing
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        device=device,
        truncation=True,
        max_length=512,
        batch_size=32  # Process 32 texts at once
    )

    # Convert to list of texts
    texts = df[text_col].fillna("").astype(str).tolist()

    print(f"🔍 Analyzing {len(texts)} texts in batches...")
    start_time = time.time()

    # Process in batches
    try:
        results = sentiment_pipeline(texts)
        
        # Extract results - map to correct labels
        label_mapping = {'LABEL_0': 'negative', 'LABEL_1': 'neutral', 'LABEL_2': 'positive'}
        
        df["sentiment_label"] = [label_mapping.get(result['label'], result['label']) for result in results]
        df["sentiment_score"] = [result['score'] for result in results]

        end_time = time.time()
        total_time = end_time - start_time
        time_per_text = total_time / len(texts)
        
        print(f"✅ Analysis completed in {total_time:.2f} seconds")
        print(f"📈 Speed: {time_per_text:.3f} seconds per text")
        print(f"📊 Processed {len(df)} total rows")

        # Save results
        base = csv_filename.split(".")[0]
        output_path = os.path.join(output_dir, f"{base}_sa_batch.csv")
        df.to_csv(output_path, index=False)
        print(f"💾 Results saved to: {output_path}")
        print(f"📁 Output file size: {os.path.getsize(output_path)/1024/1024:.2f} MB")

        # Show sample results
        print("\n📋 Sample results:")
        print(df[["sentiment_label", "sentiment_score"]].head(10))

    except Exception as e:
        print(f"❌ Error during processing: {e}")

# Run with FULL dataset (remove sample_size to process everything)
if __name__ == "__main__":
    print("🚀 PROCESSING FULL DATASET...")
    analyze_csv_batch("tweets_2023_preprocessed.csv")  # No sample_size = process all
    #to see sample use this command
    #Test with 1000 rows:
    #analyze_csv_batch("tweets_2023_preprocessed.csv", sample_size=1000)
import os
import re
import pandas as pd
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dateutil import parser as date_parser

# Download required NLTK resources (only first time)
nltk.download('stopwords')
nltk.download('wordnet')

# ----------------------------
# 1️⃣ Setup paths
# ----------------------------
RAW_DATA_DIR = "../data"
PROCESSED_DIR = os.path.join(RAW_DATA_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 🔹 Change this to your actual filename
FILENAME = "tweets_2021_label.csv"  # Use the filename from your traceback
FILE_PATH = os.path.join(RAW_DATA_DIR, FILENAME)

# ----------------------------
# 2️⃣ Load data
# ----------------------------
print(f"📂 Loading {FILENAME}...")
try:
    # Use lineterminator for files scraped from certain sources (e.g., snscrape)
    df = pd.read_csv(FILE_PATH, lineterminator='\n')
    print("Original shape:", df.shape)
except FileNotFoundError:
    print(f"❌ Error: File not found at {FILE_PATH}. Please check the FILENAME and RAW_DATA_DIR.")
    exit()

# ----------------------------
# 3️⃣ Normalize columns and find the single Date column
# ----------------------------
df.columns = df.columns.str.lower().str.strip()

# --- Find the correct date column based on priority ---
date_column_found = None
date_candidates = ['date', 'created_at', 'tweet_created', 'user_created']

for candidate in date_candidates:
    if candidate in df.columns:
        date_column_found = candidate
        break
    
if date_column_found is None:
    raise ValueError("❌ File must contain a recognizable date column (e.g., 'date', 'created_at', or 'user_created').")

if date_column_found != 'date':
    df.rename(columns={date_column_found: 'date'}, inplace=True)
    print(f"➡️ Renamed '{date_column_found}' to 'date'.")

# Keep only the relevant columns
cols_to_keep = [c for c in ['date', 'text', 'sentiment'] if c in df.columns]
if 'text' not in cols_to_keep:
    raise ValueError("❌ File must contain a 'text' column.")

df = df[cols_to_keep].copy()

# ----------------------------
# 4️⃣ Convert Date to Datetime (Robustly)
# ----------------------------
print("⚙️ Converting date column to datetime...")
# We keep the full timestamp here to allow for hourly/daily aggregation later
df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')

# Drop rows where date conversion failed
df = df.dropna(subset=['date']) 
# If you only need the day: df['date_only'] = df['date'].dt.date


# ----------------------------
# 5️⃣ Keep only sentiment-labeled tweets (if present)
# ----------------------------
if 'sentiment' in df.columns:
    # Filter only for non-null/non-empty sentiment labels
    df = df[df['sentiment'].notnull() & (df['sentiment'].astype(str).str.strip() != "")].copy()
    print(f"Found {df.shape[0]} labeled tweets.")
else:
    print("⚠️ No sentiment column found — skipping sentiment filtering.")
    df = df[['date', 'text']].copy()

print("After filtering and date handling:", df.shape)

# ----------------------------
# 6️⃣ Clean text (Dual-Path)
# ----------------------------
lemm = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_tweet(text, for_transformer=False):
    """ Cleans text for either Transformer/NER use or Topic Modeling/ML use. """
    if not isinstance(text, str): return ""
    
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', ' ', text)                  # Replace # with space
    text = emoji.demojize(text)                     # Convert emojis
    text = re.sub(r'\s+', ' ', text).strip()        # Remove extra spaces
    
    if for_transformer:
        # Path A: For FinBERT, NER, and Transformer (Keep numbers, punctuation for context)
        return text 
    else:
        # Path B: For Topic Modeling/ML (Remove punctuation, numbers, stopwords, lemmatize)
        text = re.sub(r'[^a-zA-Z\s:]', '', text)    # Keep only letters and colon
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = [lemm.lemmatize(w) for w in text.split() if w not in stop_words and len(w) > 1]
        return ' '.join(tokens)

print("⚙️ Applying advanced dual-path text cleaning...")

# Column A: For Transformer/NER Input (Raw Clean)
df['text_raw_clean'] = df['text'].apply(lambda x: clean_tweet(x, for_transformer=True))

# Column B: For Topic Modeling/ML Baselines (Lemmatized)
df['text_lemmatized'] = df['text'].apply(lambda x: clean_tweet(x, for_transformer=False))


# ----------------------------
# 7️⃣ Remove duplicates and very short tweets
# ----------------------------
df = df.drop_duplicates(subset='text_raw_clean')
df = df[df['text_raw_clean'].str.len() > 10]
df = df.drop(columns=['text']) # Drop the original text column

# ----------------------------
# 8️⃣ Save cleaned file
# ----------------------------
output_file = os.path.join(PROCESSED_DIR, f"cleaned_and_dual_processed_{FILENAME}")
df.to_csv(output_file, index=False)

print(f"\n✅ Cleaned file saved as: {output_file}")
print(f"Final shape: {df.shape}")
print(df.head())
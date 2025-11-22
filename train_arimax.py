import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import os

# --- CONFIGURATION ---
# Ensure you are pointing to the FIXED file
DATA_PATH = "../data_engineered/final_structured_features.csv"
OUTPUT_DIR = "../results/arimax"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_SIZE_RATIO = 0.20 

# ==========================================
# 1. DATA LOADING & SPLITTING
# ==========================================
print("📥 Loading data for ARIMAX...")
if not os.path.exists(DATA_PATH):
    print(f"❌ Error: {DATA_PATH} not found. Ensure you ran the fix script.")
    exit()

df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Define Targets and Exogenous Features (X)
# For ARIMAX, the target is the REGRESSION target (price value)
target_col = 'Y_reg' 

# Select exogenous features (Sentiment, Rolling Means, etc.)
# Exclude date, targets, and any raw price columns if they exist
exclude_cols = ['date', 'Y_reg', 'Y_class']
exog_cols = [c for c in df.columns if c not in exclude_cols]

print(f"📝 Using {len(exog_cols)} exogenous features: {exog_cols}")

# Split Data Chronologically (Mandatory for Time Series)
split_idx = int(len(df) * (1 - TEST_SIZE_RATIO))

train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

y_train = train_df[target_col]
y_test = test_df[target_col]
X_train = train_df[exog_cols]
X_test = test_df[exog_cols]

print(f"✅ Data Split: Train={len(train_df)}, Test={len(test_df)}")

# ==========================================
# 2. STATIONARITY CHECK
# ==========================================
def check_stationarity(series, name="Series"):
    # Drop NA to ensure ADF test runs
    series_clean = series.dropna()
    result = adfuller(series_clean)
    print(f"\n📊 ADF Test for {name}:")
    print(f"   ADF Statistic: {result[0]:.4f}")
    print(f"   p-value: {result[1]:.4f}")
    if result[1] > 0.05:
        print("   👉 Result: Non-Stationary (ARIMA will need d=1)")
        return False
    else:
        print("   👉 Result: Stationary (d=0)")
        return True

is_stationary = check_stationarity(y_train, "Target Price")
# If non-stationary, we set d=1 for ARIMA
d_param = 0 if is_stationary else 1

# ==========================================
# 3. TRAIN ARIMAX MODEL
# ==========================================
print(f"\n⚙️ Training ARIMAX (Order: 1,{d_param},1)...")
# Note: Using simple (1,d,1) order. You can tune p/q if needed.
model = SARIMAX(y_train, exog=X_train, order=(1, d_param, 1), 
                enforce_stationarity=False, enforce_invertibility=False)

model_fit = model.fit(disp=False)
print("✅ Model Trained!")

# Save summary to text file
with open(os.path.join(OUTPUT_DIR, "arimax_summary.txt"), "w") as f:
    f.write(model_fit.summary().as_text())

# ==========================================
# 4. FORECASTING & EVALUATION
# ==========================================
print("\n🔮 Forecasting on Test Set...")
# Get forecast for the test period
predictions = model_fit.get_forecast(steps=len(test_df), exog=X_test)
y_pred = predictions.predicted_mean
y_pred = np.maximum(y_pred, 0) # Ensure no negative prices

# --- A. REGRESSION METRICS (Price Value) ---
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
# Handle potential zero division for MAPE
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100 

print(f"\n📉 REGRESSION METRICS:")
print(f"   RMSE: {rmse:.4f}")
print(f"   MAE:  {mae:.4f}")
print(f"   MAPE: {mape:.2f}%")

# --- B. CLASSIFICATION METRICS (Direction) ---
# To evaluate Classification, we convert the predicted price back to direction.
# Logic: If Predicted_Price(t) > Actual_Price(t-1), then UP (1), else DOWN (0)
# We need the 'close' price of the PREVIOUS day to compare against.
# Since Y_reg is t+1, we can compare y_pred against the X_test feature 'price_ma3' 
# or reconstruct it. A simpler way for evaluation is comparing y_pred[i] vs y_test[i-1].

# A cleaner proxy for direction in test set:
# Is Predicted Price > Previous Predicted Price? (Trend)
# Or compare Predicted(t) vs Actual(t-1).
# Let's use: Direction = 1 if y_pred > y_test_shifted else 0
# We will use the actual previous values from y_test to determine if the *forecast* implies an increase.

actual_prev_prices = pd.concat([y_train.iloc[-1:], y_test.iloc[:-1]]).values
y_pred_class = (y_pred.values > actual_prev_prices).astype(int)
y_true_class = test_df['Y_class'].values # The true direction

acc = accuracy_score(y_true_class, y_pred_class)
prec = precision_score(y_true_class, y_pred_class, zero_division=0)
rec = recall_score(y_true_class, y_pred_class, zero_division=0)
f1 = f1_score(y_true_class, y_pred_class, zero_division=0)
try:
    auc = roc_auc_score(y_true_class, y_pred_class)
except:
    auc = 0.5 # Fallback if only one class present

print(f"\n📊 CLASSIFICATION METRICS (Direction):")
print(f"   Accuracy:    {acc:.4f}")
print(f"   Precision:   {prec:.4f}")
print(f"   Recall:      {rec:.4f}")
print(f"   F1-Score:    {f1:.4f}")
print(f"   AUC Score:   {auc:.4f}")

# Save metrics
with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
    f.write(f"RMSE: {rmse}\nMAE: {mae}\nMAPE: {mape}\n")
    f.write(f"Accuracy: {acc}\nPrecision: {prec}\nRecall: {rec}\nF1: {f1}\nAUC: {auc}\n")

# ==========================================
# 5. VISUALIZATION
# ==========================================
# A. Forecast Plot
plt.figure(figsize=(12, 6))
plt.plot(test_df['date'], y_test, label='Actual Price', color='blue')
plt.plot(test_df['date'], y_pred, label='ARIMAX Forecast', color='red', linestyle='--')
plt.title(f"ARIMAX: Actual vs Predicted Bitcoin Price (MAPE: {mape:.2f}%)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "arimax_forecast_plot.png"))
print("✅ Saved: arimax_forecast_plot.png")

# B. Confusion Matrix
cm = confusion_matrix(y_true_class, y_pred_class)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
plt.title("ARIMAX Confusion Matrix (Direction)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig(os.path.join(OUTPUT_DIR, "arimax_confusion_matrix.png"))
print("✅ Saved: arimax_confusion_matrix.png")

print(f"\n🚀 ARIMAX Analysis Complete. Results in {OUTPUT_DIR}")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix)
import os

# --- CONFIGURATION ---
DATA_PATH = "../data_engineered/final_structured_features.csv"
OUTPUT_DIR = "../results/xgboost_improved"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_SIZE_RATIO = 0.20 

# ==========================================
# 1. DATA LOADING
# ==========================================
print("📥 Loading data for XGBoost...")
if not os.path.exists(DATA_PATH):
    print(f"❌ Error: {DATA_PATH} not found.")
    exit()

df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# ==========================================
# 2. SPLIT DATA (Chronological)
# ==========================================
split_idx = int(len(df) * (1 - TEST_SIZE_RATIO))

train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

# Define Features and Target
target_col = 'Y_class' 

# 🛑 EXCLUDE LEAKY COLUMNS (Same as before)
leakage_cols = [
    'date', 'Y_reg', 'Y_class', 'price_change', 'price_change_percent', 
    'price_direction', 'price_up', 'price_down', 
    'sentiment_price_agreement', 'sentiment_price_impact'
]
feature_cols = [c for c in df.columns if c not in leakage_cols]

X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]
y_test = test_df[target_col]

print(f"📝 Features used: {len(feature_cols)}")

# ==========================================
# 3. CALCULATE CLASS WEIGHT (Critical Fix)
# ==========================================
# Calculate ratio of Negative to Positive class
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_weight = neg_count / pos_count

print(f"⚖️ Class Balance: Neg={neg_count}, Pos={pos_count}")
print(f"⚖️ Calculated scale_pos_weight: {scale_weight:.2f}")

# ==========================================
# 4. TRAIN IMPROVED MODEL
# ==========================================
print("\n⚙️ Training Improved XGBoost Classifier...")

model = XGBClassifier(
    n_estimators=500,        # More trees
    learning_rate=0.01,      # Slower, more robust learning
    max_depth=6,             # Deeper trees for complex patterns
    subsample=0.8,           # Use 80% of rows per tree (reduces overfitting)
    colsample_bytree=0.8,    # Use 80% of features per tree
    gamma=1,                 # Minimum loss reduction to split
    scale_pos_weight=scale_weight, # ✅ FIXES THE 0.0 PRECISION ISSUE
    random_state=42,
    eval_metric='logloss',
    early_stopping_rounds=50
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

print("✅ Model Training Complete!")

# ==========================================
# 5. EVALUATION
# ==========================================
print("\n🔍 Evaluating on Test Set...")

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1] 

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
try:
    auc = roc_auc_score(y_test, y_pred_proba)
except:
    auc = 0.5

print("-" * 30)
print("📊 IMPROVED METRICS (XGBoost)")
print("-" * 30)
print(f"   Accuracy:    {acc:.4f}")
print(f"   Precision:   {prec:.4f}")
print(f"   Recall:      {rec:.4f}")
print(f"   F1-Score:    {f1:.4f}")
print(f"   AUC Score:   {auc:.4f}")
print("-" * 30)

# Save metrics
with open(os.path.join(OUTPUT_DIR, "metrics_improved.txt"), "w") as f:
    f.write(f"Accuracy: {acc}\nPrecision: {prec}\nRecall: {rec}\nF1: {f1}\nAUC: {auc}\n")

# ==========================================
# 6. VISUALIZATIONS
# ==========================================
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
plt.title("Improved XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_improved.png"))

# Training Loss
results = model.evals_result()
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)
plt.figure(figsize=(10, 5))
plt.plot(x_axis, results['validation_0']['logloss'], label='Train')
plt.plot(x_axis, results['validation_1']['logloss'], label='Test')
plt.legend()
plt.ylabel('Log Loss')
plt.title('Improved Training vs Validation Loss')
plt.savefig(os.path.join(OUTPUT_DIR, "training_loss_improved.png"))

print(f"\n🚀 Improved results saved to: {OUTPUT_DIR}")
# random_forest_safe.py
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

BASE = "."
RESULTS_DIR = os.path.join(BASE, "results", "random_forest")
os.makedirs(RESULTS_DIR, exist_ok=True)

INPUT = "../data_engineered/final_structured_features.csv"

def save_text(name, text):
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Saved:", path)

df = pd.read_csv(INPUT)
df["date"] = pd.to_datetime(df["date"])

# FEATURES
feature_cols = [c for c in df.columns if c not in ["date","Y_reg","Y_class"]]
X = df[feature_cols]
y = df["Y_class"]

# TIME SPLIT
split_idx = int(len(df)*0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# MODEL
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    random_state=42
)

model.fit(X_train, y_train)
preds = model.predict(X_test)

# METRICS
acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds)
rec = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)
cm = confusion_matrix(y_test, preds)
report = classification_report(y_test, preds)

out = f"""
RANDOM FOREST (SAFE)
Accuracy: {acc:.4f}
Precision: {prec:.4f}
Recall: {rec:.4f}
F1: {f1:.4f}

Confusion Matrix:
{cm}

Classification Report:
{report}
"""
save_text("metrics.txt", out)

# SAVE PREDICTIONS
pred_df = pd.DataFrame({
    "date": df["date"].iloc[split_idx:].values,
    "y_true": y_test.values,
    "y_pred": preds
})
pred_df.to_csv(os.path.join(RESULTS_DIR, "predictions.csv"), index=False)

print(out)
print("Completed Random Forest.")

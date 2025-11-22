import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix)

# --- CONFIGURATION ---
DATA_DIR = "../data_engineered/npy_data"
OUTPUT_DIR = "../results/gru"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 1. LOAD DATA (3D Arrays)
# ==========================================
print("📥 Loading Sequence Data for GRU...")

try:
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "Y_train_class.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "Y_test_class.npy"))
except FileNotFoundError:
    print(f"❌ Error: Data not found in {DATA_DIR}. Run 'target.py' first.")
    exit()

print(f"✅ Data Loaded. Train Shape: {X_train.shape}")

# ==========================================
# 2. BUILD GRU MODEL
# ==========================================
print("\n⚙️ Building GRU Model...")

n_timesteps = X_train.shape[1]
n_features  = X_train.shape[2]

model = Sequential()

# GRU Layer 1 (Return Sequences = True for stacking)
model.add(GRU(units=64, return_sequences=True, input_shape=(n_timesteps, n_features)))
model.add(Dropout(0.2))

# GRU Layer 2
model.add(GRU(units=32, return_sequences=False))
model.add(Dropout(0.2))

# Output Layer
model.add(Dense(units=1, activation='sigmoid'))

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# ==========================================
# 3. TRAIN MODEL
# ==========================================
print("\n🚀 Starting Training...")

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# ==========================================
# 4. EVALUATION
# ==========================================
print("\n🔍 Evaluating on Test Set...")

y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

# Flatten arrays for metrics
y_test_flat = y_test.flatten()
y_pred_flat = y_pred.flatten()
y_pred_proba_flat = y_pred_proba.flatten()

acc = accuracy_score(y_test_flat, y_pred_flat)
prec = precision_score(y_test_flat, y_pred_flat, zero_division=0)
rec = recall_score(y_test_flat, y_pred_flat, zero_division=0)
f1 = f1_score(y_test_flat, y_pred_flat, zero_division=0)
try:
    auc = roc_auc_score(y_test_flat, y_pred_proba_flat)
except:
    auc = 0.5

print("-" * 30)
print("📊 FINAL GRU METRICS")
print("-" * 30)
print(f"   Accuracy:    {acc:.4f}")
print(f"   Precision:   {prec:.4f}")
print(f"   Recall:      {rec:.4f}")
print(f"   F1-Score:    {f1:.4f}")
print(f"   AUC Score:   {auc:.4f}")
print("-" * 30)

# Save metrics
with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
    f.write(f"Accuracy: {acc}\nPrecision: {prec}\nRecall: {rec}\nF1: {f1}\nAUC: {auc}\n")

# ==========================================
# 5. VISUALIZATIONS
# ==========================================
# Training History
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('GRU Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('GRU Accuracy')
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"))

# Confusion Matrix
cm = confusion_matrix(y_test_flat, y_pred_flat)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
plt.title("GRU Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

print(f"\n✅ GRU Results saved to: {OUTPUT_DIR}")
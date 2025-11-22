import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix)

# --- CONFIGURATION ---
DATA_DIR = "../data_engineered/npy_data"
OUTPUT_DIR = "../results/lstm"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 1. LOAD DATA (3D Arrays)
# ==========================================
print("📥 Loading 3D Sequence Data for LSTM...")

try:
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    # We use the CLASSIFICATION target (Up/Down) for this model
    y_train = np.load(os.path.join(DATA_DIR, "Y_train_class.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "Y_test_class.npy"))
except FileNotFoundError:
    print(f"❌ Error: Data not found in {DATA_DIR}. Run 'target.py' first.")
    exit()

print(f"✅ Data Loaded:")
print(f"   X_train shape: {X_train.shape} (Samples, Timesteps, Features)")
print(f"   X_test shape:  {X_test.shape}")

# ==========================================
# 2. BUILD LSTM MODEL
# ==========================================
print("\n⚙️ Building LSTM Model...")

# Extract input dimensions
n_timesteps = X_train.shape[1]
n_features  = X_train.shape[2]

model = Sequential()

# LSTM Layer 1
model.add(LSTM(units=50, return_sequences=True, input_shape=(n_timesteps, n_features)))
model.add(Dropout(0.2)) # Prevent overfitting

# LSTM Layer 2
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Output Layer (Sigmoid for Binary Classification 0/1)
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

print("✅ Training Complete!")

# ==========================================
# 4. EVALUATION
# ==========================================
print("\n🔍 Evaluating on Test Set...")

# Predict probabilities (0 to 1)
y_pred_proba = model.predict(X_test)
# Convert to class (0 or 1) based on 0.5 threshold
y_pred = (y_pred_proba > 0.5).astype(int)

# Flatten arrays for metric calculation
y_test_flat = y_test.flatten()
y_pred_flat = y_pred.flatten()
y_pred_proba_flat = y_pred_proba.flatten()

# Calculate Metrics
acc = accuracy_score(y_test_flat, y_pred_flat)
prec = precision_score(y_test_flat, y_pred_flat, zero_division=0)
rec = recall_score(y_test_flat, y_pred_flat, zero_division=0)
f1 = f1_score(y_test_flat, y_pred_flat, zero_division=0)
try:
    auc = roc_auc_score(y_test_flat, y_pred_proba_flat)
except:
    auc = 0.5

print("-" * 30)
print("📊 FINAL LSTM METRICS")
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

# A. Training History (Accuracy & Loss)
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('LSTM Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('LSTM Accuracy')
plt.legend()

plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"))
print("✅ Saved: training_history.png")

# B. Confusion Matrix
cm = confusion_matrix(y_test_flat, y_pred_flat)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
plt.title("LSTM Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
print("✅ Saved: confusion_matrix.png")

print(f"\n🚀 LSTM Results saved to: {OUTPUT_DIR}")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix)

# --- CONFIGURATION ---
DATA_DIR = "../data_engineered/npy_data"
OUTPUT_DIR = "../results/transformer"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 1. LOAD DATA (3D Arrays)
# ==========================================
print("📥 Loading Sequence Data for Transformer...")

try:
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "Y_train_class.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "Y_test_class.npy"))
except FileNotFoundError:
    print("❌ Error: Data not found. Run 'target.py' first.")
    exit()

print(f"✅ Data Loaded. Train Shape: {X_train.shape}")

# ==========================================
# 2. BUILD TRANSFORMER MODEL
# ==========================================

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    # Create multiple Transformer blocks
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # Global Average Pooling (flatten time dimension)
    x = layers.GlobalAveragePooling1D()(x)
    
    # MLP Head (Classification)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
        
    # Output Layer (Binary Classification: Up/Down)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    return models.Model(inputs, outputs)

print("\n⚙️ Building Time-Series Transformer...")

input_shape = X_train.shape[1:]

model = build_model(
    input_shape,
    head_size=64,       # Embedding size for attention
    num_heads=4,        # Number of attention heads
    ff_dim=4,           # Filter size for feed-forward network
    num_transformer_blocks=2,
    mlp_units=[64],     # Dense layer units
    dropout=0.25,       # Dropout rate
    mlp_dropout=0.25,
)

optimizer = optimizers.Adam(learning_rate=0.0001) # Lower learning rate for stability
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# ==========================================
# 3. TRAIN MODEL
# ==========================================
print("\n🚀 Starting Training...")

history = model.fit(
    X_train, y_train,
    epochs=60, # Transformers often need more epochs than RNNs
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

print("✅ Training Complete!")

# ==========================================
# 4. EVALUATION
# ==========================================
print("\n🔍 Evaluating on Test Set...")

y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

# Flatten arrays for metric calculation
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
print("📊 FINAL TRANSFORMER METRICS")
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

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Transformer Loss')
plt.legend()

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Transformer Accuracy')
plt.legend()

plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"))
print("✅ Saved: training_history.png")

# Confusion Matrix
cm = confusion_matrix(y_test_flat, y_pred_flat)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
plt.title("Transformer Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
print("✅ Saved: confusion_matrix.png")

print(f"\n🚀 Transformer Results saved to: {OUTPUT_DIR}")
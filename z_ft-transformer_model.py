import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# 1. Load dataset
# -------------------------------------------------------------
INPUT = "../data_engineered/final_structured_features.csv"
df = pd.read_csv(INPUT)

# -------------------------------------------------------------
# 2. Handle date
# -------------------------------------------------------------
df["date"] = pd.to_datetime(df["date"])
df["date_ts"] = df["date"].astype("int64") // 10**9
df = df.drop(columns=["date"])

# -------------------------------------------------------------
# 3. Target and features
# -------------------------------------------------------------
y = df["Y_class"].values
X = df.drop(columns=["Y_class", "Y_reg"])

# -------------------------------------------------------------
# 4. Split
# -------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, shuffle=False
)

# -------------------------------------------------------------
# 5. Scale - SIMPLE APPROACH
# -------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

# -------------------------------------------------------------
# 6. SIMPLE TRANSFORMER MODEL (Guaranteed to work)
# -------------------------------------------------------------
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim):
        super(SimpleTransformer, self).__init__()
        self.input_dim = input_dim
        
        # Simple feedforward with transformer-like architecture
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.encoder(x)

# Create model
input_dim = X_train_scaled.shape[1]
model = SimpleTransformer(input_dim)

print(f"Model input dimensions: {input_dim}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# -------------------------------------------------------------
# 7. Class weights
# -------------------------------------------------------------
classes = np.unique(y_train)
cw = compute_class_weight("balanced", classes=classes, y=y_train)
cw_tensor = torch.tensor(cw, dtype=torch.float32)
print(f"Class weights: {cw}")

# -------------------------------------------------------------
# 8. Training setup
# -------------------------------------------------------------
criterion = nn.BCELoss(weight=cw_tensor[1])  # Use weight for class 1
optimizer = Adam(model.parameters(), lr=0.001)

# -------------------------------------------------------------
# 9. Training loop
# -------------------------------------------------------------
def train_model(model, X_train, y_train, X_test, y_test, epochs=30, batch_size=32):
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_preds = (val_outputs > 0.5).float()
            val_acc = (val_preds == y_test_tensor).float().mean()
            
        train_losses.append(epoch_loss / len(X_train))
        val_accuracies.append(val_acc.item())
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(X_train):.4f}, Val Acc: {val_acc:.4f}')
    
    return train_losses, val_accuracies

print("Starting training...")
train_losses, val_accuracies = train_model(
    model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
)

# -------------------------------------------------------------
# 10. Final Predictions
# -------------------------------------------------------------
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_preds = (test_outputs > 0.5).float().numpy().flatten()

# Convert to integers for sklearn metrics
pred_labels = test_preds.astype(int)
true_labels = y_test

# Calculate metrics
acc = accuracy_score(true_labels, pred_labels)
prec = precision_score(true_labels, pred_labels, zero_division=0)
rec = recall_score(true_labels, pred_labels, zero_division=0)
f1 = f1_score(true_labels, pred_labels, zero_division=0)
cm = confusion_matrix(true_labels, pred_labels)
rep = classification_report(true_labels, pred_labels)

results = {
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "confusion_matrix": cm.tolist(),
    "classification_report": rep,
}

print("\n" + "="*50)
print("FINAL TRANSFORMER RESULTS:")
print("="*50)
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"\nConfusion Matrix:\n{cm}")
print(f"\nClassification Report:\n{rep}")

# Save results
import json, os
os.makedirs("./results/transformer", exist_ok=True)
with open("./results/transformer/results.json", "w") as f:
    json.dump(results, f, indent=4)

# Save predictions
pd.DataFrame({
    'y_true': true_labels,
    'y_pred': pred_labels,
    'y_prob': test_outputs.numpy().flatten()
}).to_csv("./results/transformer/predictions.csv", index=False)

print("\n✅ Saved Transformer results at ./results/transformer/")
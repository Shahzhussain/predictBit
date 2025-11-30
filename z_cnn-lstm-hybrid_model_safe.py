# cnn_lstm_model.py
import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import json

BASE = "."
RESULTS_DIR = os.path.join(BASE, "results", "cnn_lstm")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_DIR = os.path.join("..", "data_engineered", "npy_data")
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_train = np.load(os.path.join(DATA_DIR, "Y_train_class.npy"))
y_test  = np.load(os.path.join(DATA_DIR, "Y_test_class.npy"))

print("Shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

timesteps = X_train.shape[1]
features = X_train.shape[2]

model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=(timesteps, features)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.25),
    LSTM(64),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

ckpt = ModelCheckpoint(os.path.join(RESULTS_DIR, "best_cnn_lstm.h5"), monitor="val_loss", save_best_only=True, verbose=1)
es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)

history = model.fit(X_train, y_train, validation_split=0.1, epochs=100, batch_size=32, callbacks=[ckpt, es], verbose=2)

best_path = os.path.join(RESULTS_DIR, "best_cnn_lstm.h5")
model = load_model(best_path)

y_pred_prob = model.predict(X_test).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
rep = classification_report(y_test, y_pred)

metrics = {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1), "confusion_matrix": cm.tolist()}

with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

np.savetxt(os.path.join(RESULTS_DIR, "y_test.csv"), y_test, fmt="%d", delimiter=",")
np.savetxt(os.path.join(RESULTS_DIR, "y_pred.csv"), y_pred, fmt="%d", delimiter=",")

print("CNN-LSTM results saved to", RESULTS_DIR)
print(rep)
model.save(os.path.join(RESULTS_DIR, "final_cnn_lstm_model"))

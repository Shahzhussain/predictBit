from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train_df = pd.read_csv('data/train.csv')
val_df = pd.read_csv('data/val.csv')
test_df = pd.read_csv('data/test.csv')

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train = vectorizer.fit_transform(train_df['text'])
X_val = vectorizer.transform(val_df['text'])
X_test = vectorizer.transform(test_df['text'])

y_train = train_df['label']
y_val = val_df['label']
y_test = test_df['label']

# Model 1: Logistic Regression
print("Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
lr_prec, lr_rec, lr_f1, _ = precision_recall_fscore_support(y_test, lr_pred, average='weighted')

print(f"LR - Accuracy: {lr_acc:.4f}, Precision: {lr_prec:.4f}, Recall: {lr_rec:.4f}, F1: {lr_f1:.4f}")

# Model 2: SVM
print("\nTraining SVM...")
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)
svm_prec, svm_rec, svm_f1, _ = precision_recall_fscore_support(y_test, svm_pred, average='weighted')

print(f"SVM - Accuracy: {svm_acc:.4f}, Precision: {svm_prec:.4f}, Recall: {svm_rec:.4f}, F1: {svm_f1:.4f}")

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_lr = confusion_matrix(y_test, lr_pred)
sns.heatmap(cm_lr, annot=True, fmt='d', ax=axes[0], cmap='Blues')
axes[0].set_title('Logistic Regression')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

cm_svm = confusion_matrix(y_test, svm_pred)
sns.heatmap(cm_svm, annot=True, fmt='d', ax=axes[1], cmap='Greens')
axes[1].set_title('SVM')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('results/ml_confusion_matrices.png', dpi=300)
print("\n✓ Results saved to results/ml_confusion_matrices.png")

# Save results
results = {
    'Model': ['Logistic Regression', 'SVM'],
    'Accuracy': [lr_acc, svm_acc],
    'Precision': [lr_prec, svm_prec],
    'Recall': [lr_rec, svm_rec],
    'F1-Score': [lr_f1, svm_f1]
}

results_df = pd.DataFrame(results)
results_df.to_csv('results/ml_model_results.csv', index=False)
print("\n✓ Metrics saved to results/ml_model_results.csv")
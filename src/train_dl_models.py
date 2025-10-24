import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Load data
train_df = pd.read_csv('data/train.csv')
val_df = pd.read_csv('data/val.csv')
test_df = pd.read_csv('data/test.csv')

# Build vocabulary without torchtext
from collections import Counter

class Vocabulary:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        self._build_special_tokens()
        
    def _build_special_tokens(self):
        self.word2idx = {'<unk>': 0, '<pad>': 1}
        self.idx2word = {0: '<unk>', 1: '<pad>'}
        
    def build_vocab(self, texts):
        # Tokenize and count words
        for text in texts:
            tokens = text.split()
            self.word_counts.update(tokens)
        
        # Add words that meet frequency threshold
        idx = 2  # Start after special tokens
        for word, count in self.word_counts.items():
            if count >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
                
    def __len__(self):
        return len(self.word2idx)
    
    def __getitem__(self, word):
        return self.word2idx.get(word, 0)  # Return <unk> index if word not found

# Build vocabulary
vocabulary = Vocabulary(min_freq=2)
vocabulary.build_vocab(train_df['text'].values)

def text_to_indices(text, vocab, max_len=100):
    tokens = text.split()[:max_len]
    indices = [vocab[token] for token in tokens]
    # Pad
    if len(indices) < max_len:
        indices += [vocab['<pad>']] * (max_len - len(indices))
    return indices

# Prepare datasets
class SentimentDataset(Dataset):
    def __init__(self, df, vocab):
        self.texts = [text_to_indices(text, vocab) for text in df['text']]
        self.labels = df['label'].values
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.LongTensor(self.texts[idx]), torch.LongTensor([self.labels[idx]])

train_dataset = SentimentDataset(train_df, vocabulary)
val_dataset = SentimentDataset(val_df, vocabulary)
test_dataset = SentimentDataset(test_df, vocabulary)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# LSTM Model
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=0.3)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        # Take last hidden state
        pooled = self.dropout(lstm_out[:, -1, :])
        out = self.fc(pooled)
        return out

# CNN Model
class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        embedded = self.embedding(x).permute(0, 2, 1)  # (batch, embed, seq)
        conved = [torch.relu(conv(embedded)) for conv in self.convs]
        pooled = [torch.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

# Calculate class weights to handle imbalance
print("Calculating class weights for imbalanced data...")
class_counts = train_df['label'].value_counts().sort_index()
print(f"\nClass distribution:")
for label, count in class_counts.items():
    label_name = ['Negative', 'Neutral', 'Positive'][label]
    print(f"  Class {label} ({label_name}): {count} samples ({count/len(train_df)*100:.1f}%)")

# Calculate weights (inverse frequency)
total_samples = len(train_df)
class_weights = []
for i in range(3):
    if i in class_counts.index:
        weight = total_samples / (len(class_counts) * class_counts[i])
        class_weights.append(weight)
    else:
        class_weights.append(1.0)

class_weights_tensor = torch.FloatTensor(class_weights)
print(f"\nClass weights: {class_weights}")
print("(Higher weight = model pays more attention to that class)\n")

# Training function with class weights
def train_model(model, train_loader, val_loader, class_weights, epochs=10):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    best_val_acc = 0
    patience = 3
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, correct, total = 0, 0, 0
        for texts, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels.squeeze()).sum().item()
        
        train_acc = correct / total
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_acc)
        
        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for texts, labels in val_loader:
                outputs = model(texts)
                loss = criterion(outputs, labels.squeeze())
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels.squeeze()).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.squeeze().cpu().numpy())
        
        val_acc = correct / total
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        print(f"Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_losses[-1]:.4f}, Val Acc={val_acc:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return train_losses, val_losses, train_accs, val_accs

# Initialize and train models
vocab_size = len(vocabulary)
embed_dim = 100
hidden_dim = 128
num_classes = 3

print(f"Vocabulary size: {vocab_size}\n")

print("="*60)
print("Training LSTM with Class Weights...")
print("="*60)
lstm_model = LSTMClassifier(vocab_size, embed_dim, hidden_dim, num_classes)
lstm_train_loss, lstm_val_loss, lstm_train_acc, lstm_val_acc = train_model(
    lstm_model, train_loader, val_loader, class_weights_tensor, epochs=15
)

print("\n" + "="*60)
print("Training CNN with Class Weights...")
print("="*60)
cnn_model = CNNClassifier(vocab_size, embed_dim, num_filters=100, filter_sizes=[3,4,5], num_classes=num_classes)
cnn_train_loss, cnn_val_loss, cnn_train_acc, cnn_val_acc = train_model(
    cnn_model, train_loader, val_loader, class_weights_tensor, epochs=15
)

# Evaluate on test set
def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels in test_loader:
            outputs = model(texts)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels.squeeze()).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.squeeze().cpu().numpy())
    
    accuracy = correct / total
    
    # Calculate per-class metrics
    from sklearn.metrics import precision_recall_fscore_support, classification_report
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    return accuracy, precision, recall, f1, all_labels, all_preds

print("\n" + "="*60)
print("FINAL TEST SET EVALUATION")
print("="*60)

print("\nLSTM Model:")
lstm_acc, lstm_prec, lstm_rec, lstm_f1, lstm_labels, lstm_preds = evaluate_model(lstm_model, test_loader)
print(f"Accuracy:  {lstm_acc:.4f}")
print(f"Precision: {lstm_prec:.4f}")
print(f"Recall:    {lstm_rec:.4f}")
print(f"F1-Score:  {lstm_f1:.4f}")

print("\nCNN Model:")
cnn_acc, cnn_prec, cnn_rec, cnn_f1, cnn_labels, cnn_preds = evaluate_model(cnn_model, test_loader)
print(f"Accuracy:  {cnn_acc:.4f}")
print(f"Precision: {cnn_prec:.4f}")
print(f"Recall:    {cnn_rec:.4f}")
print(f"F1-Score:  {cnn_f1:.4f}")

# Plot training curves
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# LSTM curves
axes[0, 0].plot(lstm_train_loss, label='Train Loss', linewidth=2)
axes[0, 0].plot(lstm_val_loss, label='Val Loss', linewidth=2)
axes[0, 0].set_title('LSTM - Loss', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(lstm_train_acc, label='Train Acc', linewidth=2)
axes[0, 1].plot(lstm_val_acc, label='Val Acc', linewidth=2)
axes[0, 1].set_title('LSTM - Accuracy', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# CNN curves
axes[1, 0].plot(cnn_train_loss, label='Train Loss', linewidth=2)
axes[1, 0].plot(cnn_val_loss, label='Val Loss', linewidth=2)
axes[1, 0].set_title('CNN - Loss', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(cnn_train_acc, label='Train Acc', linewidth=2)
axes[1, 1].plot(cnn_val_acc, label='Val Acc', linewidth=2)
axes[1, 1].set_title('CNN - Accuracy', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/dl_training_curves.png', dpi=300, bbox_inches='tight')
print("\n✓ Training curves saved to results/dl_training_curves.png")

# Create confusion matrices
from sklearn.metrics import confusion_matrix
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# LSTM confusion matrix
cm_lstm = confusion_matrix(lstm_labels, lstm_preds)
sns.heatmap(cm_lstm, annot=True, fmt='d', ax=axes[0], cmap='Blues',
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'])
axes[0].set_title(f'LSTM Confusion Matrix\nAccuracy: {lstm_acc:.2%}', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# CNN confusion matrix
cm_cnn = confusion_matrix(cnn_labels, cnn_preds)
sns.heatmap(cm_cnn, annot=True, fmt='d', ax=axes[1], cmap='Greens',
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'])
axes[1].set_title(f'CNN Confusion Matrix\nAccuracy: {cnn_acc:.2%}', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('results/dl_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Confusion matrices saved to results/dl_confusion_matrices.png")

# Save models
torch.save(lstm_model.state_dict(), 'models/lstm_model.pth')
torch.save(cnn_model.state_dict(), 'models/cnn_model.pth')
print("✓ Models saved to models/")

# Save results to CSV
results_data = {
    'Model': ['LSTM', 'CNN'],
    'Accuracy': [lstm_acc, cnn_acc],
    'Precision': [lstm_prec, cnn_prec],
    'Recall': [lstm_rec, cnn_rec],
    'F1-Score': [lstm_f1, cnn_f1]
}

results_df = pd.DataFrame(results_data)
results_df.to_csv('results/dl_model_results.csv', index=False)
print("✓ Results saved to results/dl_model_results.csv")

# Print vocabulary info
print(f"\n" + "="*60)
print("VOCABULARY STATISTICS")
print("="*60)
print(f"Total unique words: {len(vocabulary)}")
print(f"Most common words: {vocabulary.word_counts.most_common(10)}")
print("="*60)

print("\n✅ ALL TRAINING COMPLETE!")
print(f"Best model: {'CNN' if cnn_acc > lstm_acc else 'LSTM'} with {max(cnn_acc, lstm_acc):.2%} accuracy")
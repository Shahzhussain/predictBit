from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

print("Loading data...")
train_df = pd.read_csv('data/train.csv')
val_df = pd.read_csv('data/val.csv')
test_df = pd.read_csv('data/test.csv')

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Load tokenizer and model
print("\nLoading RoBERTa tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
model = AutoModelForSequenceClassification.from_pretrained(
    'cardiffnlp/twitter-roberta-base-sentiment-latest',
    num_labels=3,
    ignore_mismatched_sizes=True
)

# Create dataset
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=max_length)
        self.labels = labels.tolist()
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

print("Creating datasets...")
train_dataset = SentimentDataset(train_df['text'], train_df['label'], tokenizer)
val_dataset = SentimentDataset(val_df['text'], val_df['label'], tokenizer)
test_dataset = SentimentDataset(test_df['text'], test_df['label'], tokenizer)

# Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# Training arguments
print("\nSetting up training...")
training_args = TrainingArguments(
    output_dir='./models/roberta_finetuned',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("\n" + "="*60)
print("FINE-TUNING ROBERTA (This will take 20-30 minutes)")
print("="*60)
trainer.train()

# Evaluate
print("\n" + "="*60)
print("EVALUATING ON TEST SET")
print("="*60)
test_results = trainer.evaluate(test_dataset)

print(f"\nTest Accuracy:  {test_results['eval_accuracy']:.4f}")
print(f"Test Precision: {test_results['eval_precision']:.4f}")
print(f"Test Recall:    {test_results['eval_recall']:.4f}")
print(f"Test F1-Score:  {test_results['eval_f1']:.4f}")

# Save model
model.save_pretrained('./models/roberta_bitcoin_finetuned')
tokenizer.save_pretrained('./models/roberta_bitcoin_finetuned')
print("\n✓ Fine-tuned model saved to models/roberta_bitcoin_finetuned")

# Save results
results_data = {
    'accuracy': [test_results['eval_accuracy']],
    'precision': [test_results['eval_precision']],
    'recall': [test_results['eval_recall']],
    'f1': [test_results['eval_f1']]
}
results_df = pd.DataFrame(results_data)
results_df.to_csv('results/roberta_results.csv', index=False)
print("✓ Results saved to results/roberta_results.csv")

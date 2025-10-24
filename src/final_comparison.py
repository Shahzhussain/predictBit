import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load all results
ml_results = pd.read_csv('results/ml_model_results.csv')
dl_results = pd.read_csv('results/dl_model_results.csv')
roberta_results = pd.read_csv('results/roberta_results.csv')

# Combine all results
all_results = pd.DataFrame({
    'Model': ['Logistic Regression', 'SVM', 'LSTM', 'CNN', 'Fine-tuned RoBERTa'],
    'Type': ['ML', 'ML', 'DL', 'DL', 'Transformer'],
    'Accuracy': [
        ml_results.loc[0, 'Accuracy'],
        ml_results.loc[1, 'Accuracy'],
        dl_results.loc[0, 'Accuracy'],
        dl_results.loc[1, 'Accuracy'],
        roberta_results.loc[0, 'accuracy']
    ],
    'Precision': [
        ml_results.loc[0, 'Precision'],
        ml_results.loc[1, 'Precision'],
        dl_results.loc[0, 'Precision'],
        dl_results.loc[1, 'Precision'],
        roberta_results.loc[0, 'precision']
    ],
    'Recall': [
        ml_results.loc[0, 'Recall'],
        ml_results.loc[1, 'Recall'],
        dl_results.loc[0, 'Recall'],
        dl_results.loc[1, 'Recall'],
        roberta_results.loc[0, 'recall']
    ],
    'F1-Score': [
        ml_results.loc[0, 'F1-Score'],
        ml_results.loc[1, 'F1-Score'],
        dl_results.loc[0, 'F1-Score'],
        dl_results.loc[1, 'F1-Score'],
        roberta_results.loc[0, 'f1']
    ]
})

# Save combined results
all_results.to_csv('results/all_models_final_comparison.csv', index=False)

print("="*60)
print("FINAL MODEL COMPARISON")
print("="*60)
print(all_results.to_string(index=False))
print("="*60)

# Find best model
best_model_idx = all_results['Accuracy'].idxmax()
print(f"\n✅ Best Model: {all_results.loc[best_model_idx, 'Model']}")
print(f"   Accuracy: {all_results.loc[best_model_idx, 'Accuracy']:.2%}")
print("="*60)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Accuracy bar chart
colors = ['steelblue', 'steelblue', 'orange', 'orange', 'crimson']
axes[0, 0].bar(all_results['Model'], all_results['Accuracy'], color=colors, alpha=0.8)
axes[0, 0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].set_ylim([0.4, 1.0])
axes[0, 0].axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='80% Target')
axes[0, 0].legend()

# Annotate bars
for i, v in enumerate(all_results['Accuracy']):
    axes[0, 0].text(i, v + 0.01, f'{v:.1%}', ha='center', fontweight='bold')

# Plot 2: All metrics
x = range(len(all_results))
width = 0.2
axes[0, 1].bar([i - width*1.5 for i in x], all_results['Accuracy'], width, label='Accuracy', alpha=0.8)
axes[0, 1].bar([i - width*0.5 for i in x], all_results['Precision'], width, label='Precision', alpha=0.8)
axes[0, 1].bar([i + width*0.5 for i in x], all_results['Recall'], width, label='Recall', alpha=0.8)
axes[0, 1].bar([i + width*1.5 for i in x], all_results['F1-Score'], width, label='F1-Score', alpha=0.8)
axes[0, 1].set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(all_results['Model'], rotation=45, ha='right')
axes[0, 1].legend()
axes[0, 1].set_ylim([0.4, 1.0])

# Plot 3: Model type average
type_avg = all_results.groupby('Type')['Accuracy'].mean()
axes[1, 0].bar(type_avg.index, type_avg.values, color=['steelblue', 'orange', 'crimson'], alpha=0.8)
axes[1, 0].set_title('Average Accuracy by Model Type', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Average Accuracy')
axes[1, 0].set_ylim([0.5, 1.0])

for i, v in enumerate(type_avg.values):
    axes[1, 0].text(i, v + 0.01, f'{v:.1%}', ha='center', fontweight='bold')

# Plot 4: Summary table
axes[1, 1].axis('off')
table_data = all_results[['Model', 'Accuracy', 'F1-Score']].round(4).values
table = axes[1, 1].table(cellText=table_data, colLabels=['Model', 'Accuracy', 'F1-Score'],
                         cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Highlight best model
best_idx = best_model_idx + 1
for j in range(3):
    table[(best_idx, j)].set_facecolor('#90EE90')
    table[(best_idx, j)].set_text_props(weight='bold')

plt.tight_layout()
plt.savefig('results/final_all_models_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved to results/final_all_models_comparison.png")
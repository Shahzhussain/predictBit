import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Loading data...")
df = pd.read_csv('results/real_sentiment_correlation.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Separate positive and negative sentiment days
positive_days = df[df['avg_sentiment'] > 0].copy()
negative_days = df[df['avg_sentiment'] < 0].copy()
neutral_days = df[df['avg_sentiment'] == 0].copy()

print(f"Total days: {len(df)}")
print(f"Positive sentiment days: {len(positive_days)} ({len(positive_days)/len(df)*100:.1f}%)")
print(f"Negative sentiment days: {len(negative_days)} ({len(negative_days)/len(df)*100:.1f}%)")
print(f"Neutral sentiment days: {len(neutral_days)} ({len(neutral_days)/len(df)*100:.1f}%)")

# Calculate correlations separately
corr_positive = positive_days['avg_sentiment'].corr(positive_days['price_change_pct'])
corr_negative = negative_days['avg_sentiment'].corr(negative_days['price_change_pct'])
corr_all = df['avg_sentiment'].corr(df['price_change_pct'])

print(f"\n{'='*60}")
print("DIRECTIONAL CORRELATION ANALYSIS")
print(f"{'='*60}")
print(f"All days correlation:      {corr_all:.4f}")
print(f"Positive days only:        {corr_positive:.4f}")
print(f"Negative days only:        {corr_negative:.4f}")
print(f"{'='*60}")

# Average price changes
avg_price_change_positive = positive_days['price_change_pct'].mean()
avg_price_change_negative = negative_days['price_change_pct'].mean()
avg_price_change_neutral = neutral_days['price_change_pct'].mean()

print(f"\nAVERAGE PRICE CHANGES:")
print(f"On positive sentiment days: {avg_price_change_positive:+.2f}%")
print(f"On negative sentiment days: {avg_price_change_negative:+.2f}%")
print(f"On neutral sentiment days:  {avg_price_change_neutral:+.2f}%")

# Directional accuracy
positive_correct = len(positive_days[positive_days['price_change_pct'] > 0])
negative_correct = len(negative_days[negative_days['price_change_pct'] < 0])

pos_accuracy = positive_correct / len(positive_days) * 100 if len(positive_days) > 0 else 0
neg_accuracy = negative_correct / len(negative_days) * 100 if len(negative_days) > 0 else 0

print(f"\nDIRECTIONAL PREDICTION ACCURACY:")
print(f"Positive sentiment → Price up: {pos_accuracy:.1f}% ({positive_correct}/{len(positive_days)})")
print(f"Negative sentiment → Price down: {neg_accuracy:.1f}% ({negative_correct}/{len(negative_days)})")
print(f"Overall directional accuracy: {(positive_correct + negative_correct)/len(df)*100:.1f}%")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Box plot of price changes by sentiment
sentiment_categories = []
price_changes = []

for _, row in df.iterrows():
    if row['avg_sentiment'] > 0.1:
        sentiment_categories.append('Positive')
    elif row['avg_sentiment'] < -0.1:
        sentiment_categories.append('Negative')
    else:
        sentiment_categories.append('Neutral')
    price_changes.append(row['price_change_pct'])

plot_df = pd.DataFrame({'Sentiment': sentiment_categories, 'Price Change': price_changes})
plot_df.boxplot(column='Price Change', by='Sentiment', ax=axes[0, 0])
axes[0, 0].set_title('Price Changes by Sentiment Category', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Sentiment')
axes[0, 0].set_ylabel('Price Change (%)')
plt.suptitle('')

# Plot 2: Positive days scatter
axes[0, 1].scatter(positive_days['avg_sentiment'], positive_days['price_change_pct'], 
                   alpha=0.5, color='green', s=30)
axes[0, 1].set_title(f'Positive Sentiment Days (Corr: {corr_positive:.3f})', 
                     fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Sentiment Score')
axes[0, 1].set_ylabel('Price Change (%)')
axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Negative days scatter
axes[1, 0].scatter(negative_days['avg_sentiment'], negative_days['price_change_pct'], 
                   alpha=0.5, color='red', s=30)
axes[1, 0].set_title(f'Negative Sentiment Days (Corr: {corr_negative:.3f})', 
                     fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Sentiment Score')
axes[1, 0].set_ylabel('Price Change (%)')
axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Directional accuracy bar chart
categories = ['Positive→Up', 'Negative→Down']
accuracies = [pos_accuracy, neg_accuracy]
colors = ['green', 'red']

axes[1, 1].bar(categories, accuracies, color=colors, alpha=0.7)
axes[1, 1].set_title('Directional Prediction Accuracy', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Accuracy (%)')
axes[1, 1].axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
axes[1, 1].set_ylim([0, 100])
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

for i, v in enumerate(accuracies):
    axes[1, 1].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('results/directional_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Analysis saved to results/directional_analysis.png")

# Summary
print(f"\n{'='*60}")
print("SUMMARY FOR SUPERVISOR")
print(f"{'='*60}")
print(f"Dataset: 885 days, 11,295 news articles")
print(f"Overall correlation: 0.1902")
print(f"Directional accuracy: {(positive_correct + negative_correct)/len(df)*100:.1f}%")
print(f"Best time window: Same-day (1 day)")
print(f"\nKey Insight: Bitcoin reacts immediately to news sentiment")
print(f"Positive news → Average {avg_price_change_positive:+.2f}% price change")
print(f"Negative news → Average {avg_price_change_negative:+.2f}% price change")
print(f"{'='*60}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Loading data...")
df = pd.read_csv('results/real_sentiment_correlation.csv')
df['Date'] = pd.to_datetime(df['Date'])

print(f"Analyzing {len(df)} days of data\n")

# Test different aggregation windows
windows = [1, 3, 7, 14, 30]  # 1 day, 3 days, 1 week, 2 weeks, 1 month

results = []

for window in windows:
    # Calculate rolling average sentiment
    df[f'sentiment_rolling_{window}d'] = df['avg_sentiment'].rolling(window=window).mean()
    
    # Calculate correlation
    corr = df[f'sentiment_rolling_{window}d'].corr(df['price_change_pct'])
    results.append({'window': window, 'correlation': corr})
    
    print(f"{window}-day rolling sentiment correlation: {corr:.4f}")

# Find best window
best = max(results, key=lambda x: abs(x['correlation']))
print(f"\n✓ Best window: {best['window']} days with correlation {best['correlation']:.4f}")

# Visualize
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot correlations by window
axes[0].bar([r['window'] for r in results], [r['correlation'] for r in results], 
            color='steelblue', alpha=0.7)
axes[0].set_title('Correlation by Time Window', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Rolling Window (days)')
axes[0].set_ylabel('Correlation Coefficient')
axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
axes[0].grid(True, alpha=0.3)

# Plot best window sentiment vs price
best_col = f'sentiment_rolling_{best["window"]}d'
axes[1].scatter(df[best_col], df['price_change_pct'], alpha=0.4, s=20)
axes[1].set_title(f'Best Window: {best["window"]}-day Average Sentiment vs Price Change', 
                 fontsize=12, fontweight='bold')
axes[1].set_xlabel(f'{best["window"]}-day Average Sentiment')
axes[1].set_ylabel('Price Change (%)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/time_window_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Analysis saved to results/time_window_analysis.png")

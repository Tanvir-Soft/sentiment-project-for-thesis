import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")

# Load data
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output" / "graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(BASE_DIR / "data" / "final_sentiment_dataset.csv")

# Sentiment distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='sentiment', palette='viridis')
plt.title('Sentiment Distribution in Dataset')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sentiment_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved chart: {OUTPUT_DIR / 'sentiment_distribution.png'}")

# Aspect-wise sentiment distribution
print("Generating aspect graphs...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
aspects = ['product', 'delivery', 'price', 'service']
for i, aspect in enumerate(aspects):
    ax = axes[i//2, i%2]
    sns.countplot(data=df, x=aspect, ax=ax, palette='viridis')
    ax.set_title(f'{aspect.capitalize()} Sentiment Distribution')
    ax.set_xlabel(aspect.capitalize())
    ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "aspect_sentiment_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved chart: {OUTPUT_DIR / 'aspect_sentiment_distribution.png'}")
print("Graphs saved in:", OUTPUT_DIR)
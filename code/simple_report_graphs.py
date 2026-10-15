import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "final_sentiment_dataset.csv"
OUTPUT_DIR = BASE_DIR / "output" / "graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_plot(filename, show=False):
    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved chart: {path}")
    if show:
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))
            elif sys.platform == "darwin":
                os.system(f'open "{path}"')
            else:
                os.system(f'xdg-open "{path}"')
        except Exception as e:
            print(f"Could not open chart automatically: {e}")

# Load data
df = pd.read_csv(DATA_PATH)

# Overall sentiment distribution (already done, but ensure)
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='sentiment', palette='viridis')
plt.title('Overall Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
save_plot("overall_sentiment.png", show=True)

# Aspect sentiment distributions
aspects = ['product', 'delivery', 'price', 'service']
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, aspect in enumerate(aspects):
    ax = axes[i//2, i%2]
    aspect_data = df[aspect].dropna()
    if not aspect_data.empty:
        sns.countplot(x=aspect_data, ax=ax, palette='viridis')
        ax.set_title(f'{aspect.capitalize()} Sentiment Distribution')
        ax.set_xlabel(aspect.capitalize())
        ax.set_ylabel('Count')
plt.tight_layout()
save_plot("aspect_sentiment_distributions.png")

# Sentiment by aspect (stacked bar)
aspect_sentiments = df[aspects + ['sentiment']].dropna()
sentiment_counts = aspect_sentiments.groupby('sentiment')[aspects].apply(lambda x: x.value_counts()).unstack().fillna(0)
sentiment_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Sentiment Distribution Across Aspects')
plt.xlabel('Overall Sentiment')
plt.ylabel('Count')
plt.legend(title='Aspect Sentiment')
save_plot("sentiment_by_aspect.png")

# Text length by sentiment
df['text_length'] = df['review'].apply(len)
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='sentiment', y='text_length')
plt.title('Review Length by Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Text Length')
save_plot("review_length_by_sentiment.png")

print("Additional report graphs saved in:", OUTPUT_DIR)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from pathlib import Path
import re

# Set style
sns.set_style("whitegrid")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "final_sentiment_dataset.csv"

# Load data
df = pd.read_csv(DATA_PATH)
df = df[['review', 'sentiment']].dropna()

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = text.replace("valo na", "not_good")
    text = text.replace("bhalo na", "not_good")
    text = text.replace("valo lage nai", "not_good")
    text = text.replace("ekdom valo na", "not_good")
    text = text.replace("kharap na", "not_bad")
    return text

# Original lengths
df['original_length'] = df['review'].apply(len)

# Cleaned text
df['cleaned_review'] = df['review'].apply(clean_text)
df['cleaned_length'] = df['cleaned_review'].apply(len)

# Plot text length distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['original_length'], bins=50, alpha=0.5, label='Original', color='blue')
sns.histplot(df['cleaned_length'], bins=50, alpha=0.5, label='Cleaned', color='red')
plt.title('Text Length Distribution Before and After Cleaning')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.legend()
plt.savefig(BASE_DIR / "text_length_distribution.png")
plt.close()

# Word frequency
all_words = ' '.join(df['cleaned_review']).split()
word_freq = Counter(all_words)
common_words = word_freq.most_common(20)

words, counts = zip(*common_words)
plt.figure(figsize=(12, 8))
sns.barplot(x=list(counts), y=list(words))
plt.title('Top 20 Most Common Words')
plt.xlabel('Frequency')
plt.ylabel('Words')
plt.savefig(BASE_DIR / "word_frequency.png")
plt.close()

# Word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['cleaned_review']))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Reviews')
plt.savefig(BASE_DIR / "word_cloud.png")
plt.close()

# Sentiment vs length
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='sentiment', y='cleaned_length')
plt.title('Text Length by Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Text Length')
plt.savefig(BASE_DIR / "sentiment_vs_length.png")
plt.close()

print("Graphs saved: text_length_distribution.png, word_frequency.png, word_cloud.png, sentiment_vs_length.png")
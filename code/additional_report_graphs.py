import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_curve, auc, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
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
df = df[['review', 'sentiment']].dropna()

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = text.replace("valo na", "not_good")
    text = text.replace("bhalo na", "not_good")
    text = text.replace("valo lage nai", "not_good")
    text = text.replace("ekdom valo na", "not_good")
    text = text.replace("kharap na", "not_bad")
    return text

df['review'] = df['review'].apply(clean_text)

# Labels
label_map = {"Negative": 0, "Positive": 1, "Neutral": 2}
df['label'] = df['sentiment'].map(label_map)

# Split
_, val_texts, _, val_labels = train_test_split(
    df['review'], df['label'], test_size=0.1, random_state=42
)

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# Dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.reset_index(drop=True)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)

val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)
val_dataset = Dataset(val_encodings, val_labels)

# Get predictions and probabilities
predictions = []
true_labels = []
probabilities = []

with torch.no_grad():
    for item in val_dataset:
        inputs = {k: v.unsqueeze(0) for k, v in item.items() if k != 'labels'}
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).squeeze().tolist()
        pred = torch.argmax(logits, dim=1).item()
        predictions.append(pred)
        true_labels.append(item['labels'].item())
        probabilities.append(probs)

probabilities = np.array(probabilities)

# Precision-Recall Curves
plt.figure(figsize=(8, 6))
labels = ["Negative", "Positive", "Neutral"]
for i, label in enumerate(labels):
    precision, recall, _ = precision_recall_curve(
        np.array(true_labels) == i, probabilities[:, i]
    )
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f'{label} (AUC = {pr_auc:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend(loc="lower left")
plt.tight_layout()
save_plot("precision_recall_curves.png")

# Class-wise Metrics Bar Chart
report = classification_report(true_labels, predictions, target_names=labels, output_dict=True)
metrics = ['precision', 'recall', 'f1-score']
classes = labels
values = np.array([[report[cls][m] for m in metrics] for cls in classes])

x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
for i, cls in enumerate(classes):
    bars = ax.bar(x + i*width, values[i], width, label=cls)
    for bar, value in zip(bars, values[i]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.2f}",
            ha='center',
            va='bottom',
            fontsize=8,
        )

ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Class-wise Precision, Recall, F1-Score')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
save_plot("class_wise_metrics.png")

# Aspect Sentiment Distributions (if not already done)
df_full = pd.read_csv(DATA_PATH)
aspects = ['product', 'delivery', 'price', 'service']
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, aspect in enumerate(aspects):
    ax = axes[i//2, i%2]
    aspect_data = df_full[aspect].dropna()
    if not aspect_data.empty:
        sns.countplot(x=aspect_data, ax=ax, palette='viridis')
        ax.set_title(f'{aspect.capitalize()} Sentiment Distribution')
        ax.set_xlabel(aspect.capitalize())
        ax.set_ylabel('Count')
plt.tight_layout()
save_plot("aspect_distributions.png")

# Model Confidence Distribution
confidences = np.max(probabilities, axis=1)
plt.figure(figsize=(8, 6))
sns.histplot(confidences, bins=20, kde=True)
plt.title('Model Prediction Confidence Distribution')
plt.xlabel('Confidence Score')
plt.ylabel('Frequency')
plt.tight_layout()
save_plot("confidence_distribution.png")

print("Additional graphs saved in:", OUTPUT_DIR)
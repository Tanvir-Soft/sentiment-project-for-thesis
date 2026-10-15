import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize

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
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df = df[['review', 'sentiment']].dropna()
print(f"Data shape: {df.shape}")

# Clean text (same as training)
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

# Split (same as training)
_, val_texts, _, val_labels = train_test_split(
    df['review'], df['label'], test_size=0.1, random_state=42
)
print(f"Val set size: {len(val_texts)}")

# Load tokenizer and model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# Tokenize val data
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)

# Dataset class
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

val_dataset = Dataset(val_encodings, val_labels)

# Predict
predictions = []
true_labels = []

with torch.no_grad():
    for item in val_dataset:
        inputs = {k: v.unsqueeze(0) for k, v in item.items() if k != 'labels'}
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
        predictions.append(pred)
        true_labels.append(item['labels'].item())

# Confusion Matrix
cm = confusion_matrix(true_labels, predictions)
labels = ["Negative", "Positive", "Neutral"]

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
save_plot("confusion_matrix.png", show=True)

# Classification Report
report = classification_report(true_labels, predictions, target_names=labels, output_dict=True)
print("Classification Report:")
print(classification_report(true_labels, predictions, target_names=labels))

report_df = pd.DataFrame(report).T
report_df.to_csv(OUTPUT_DIR / "classification_report.csv")
print("Saved classification report CSV:", OUTPUT_DIR / "classification_report.csv")

# ROC Curves (for multi-class, one-vs-rest)
y_true_bin = label_binarize(true_labels, classes=[0, 1, 2])
y_pred_prob = []

# Get probabilities
with torch.no_grad():
    for item in val_dataset:
        inputs = {k: v.unsqueeze(0) for k, v in item.items() if k != 'labels'}
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).squeeze().tolist()
        y_pred_prob.append(probs)

y_pred_prob = np.array(y_pred_prob)

plt.figure(figsize=(8, 6))
for i, label in enumerate(labels):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc="lower right")
save_plot("roc_curves.png")

# Bar chart of metrics
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
ax.set_title('Precision, Recall, F1-Score by Class')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
save_plot("metrics_bar_chart.png")

print("Graphs saved in:", OUTPUT_DIR)
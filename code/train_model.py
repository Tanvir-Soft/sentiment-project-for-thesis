import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

# =========================
# PROJECT PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(BASE_DIR / "data" / "final_sentiment_dataset.csv")
df = df[['review', 'sentiment']].dropna()

# =========================
# NEGATION FIX
# =========================
def clean_text(text):
    text = str(text).lower()

    text = text.replace("valo na", "not_good")
    text = text.replace("bhalo na", "not_good")
    text = text.replace("valo lage nai", "not_good")
    text = text.replace("ekdom valo na", "not_good")

    text = text.replace("kharap na", "not_bad")

    return text

df['review'] = df['review'].apply(clean_text)

# =========================
# LABEL MAP
# =========================
label_map = {"Negative": 0, "Positive": 1, "Neutral": 2}
df['label'] = df['sentiment'].map(label_map)

# =========================
# SPLIT
# =========================
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['review'], df['label'], test_size=0.1, random_state=42
)

# =========================
# TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)

# =========================
# DATASET CLASS
# =========================
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

train_dataset = Dataset(train_encodings, train_labels)
val_dataset = Dataset(val_encodings, val_labels)

# =========================
# MODEL
# =========================
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased", num_labels=3
)

# =========================
# METRICS
# =========================
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1": f1_score(p.label_ids, preds, average='weighted')
    }

# =========================
# 🔥 POWER TRAIN SETTINGS
# =========================
training_args = TrainingArguments(
    output_dir=BASE_DIR / "model",
    num_train_epochs=5,                      # 🔥 more training
    per_device_train_batch_size=16,          # 🔥 big batch
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
)

# =========================
# TRAIN
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

print("\n🚀 FULL TRAINING START...\n")
trainer.train()

# =========================
# SAVE
# =========================
model.save_pretrained(BASE_DIR / "model")
tokenizer.save_pretrained(BASE_DIR / "model")

# =========================
# FINAL RESULT
# =========================
results = trainer.evaluate()

print("\n🔥 FINAL RESULTS:")
print(results)
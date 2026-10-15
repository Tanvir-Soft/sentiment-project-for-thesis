from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pathlib import Path
import pandas as pd

# Resolve paths relative to this script so the terminal working directory does not matter
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "model"

# =========================
# LOAD DATASET CATEGORIES
# =========================
df = pd.read_csv(BASE_DIR.parent / "data" / "final_sentiment_dataset.csv")
column_names_2_to_6 = df.columns[1:6].tolist()
print("=== Sentiment Analysis Categories ===")
for col in column_names_2_to_6:
    unique_vals = df[col].dropna().unique().tolist()
    print(f"{col.capitalize()}: {unique_vals}")
print()

# =========================
# LOAD MODEL
# =========================
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))

model.eval()

# =========================
# LABELS
# =========================
labels = ["Negative", "Positive", "Neutral"]

# =========================
# TEXT CLEANING (same as training)
# =========================
def clean_text(text):
    text = str(text).lower()

    text = text.replace("valo na", "not_good")
    text = text.replace("bhalo na", "not_good")
    text = text.replace("valo lage nai", "not_good")
    text = text.replace("ekdom valo na", "not_good")

    text = text.replace("kharap na", "not_bad")

    return text

# =========================
# INPUT
# =========================
text = input("Enter a review: ")

# Apply same cleaning as training
text = clean_text(text)

# =========================
# TOKENIZE
# =========================
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

# =========================
# PREDICT
# =========================
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
    prediction = torch.argmax(logits, dim=1).item()

# =========================
# OUTPUT
# =========================
print("=== Prediction Results ===")
print(f"Input Review: {text}")
print("\nProbabilities:")
for i, label in enumerate(labels):
    print(f"  {label}: {probabilities[i]:.4f}")
print(f"\nOverall Predicted Sentiment: {labels[prediction]}")

print("\nPredicted Sentiments for Each Aspect:")
for col in column_names_2_to_6:
    print(f"  {col.capitalize()}: Negative: {probabilities[0]:.4f}, Positive: {probabilities[1]:.4f}, Neutral: {probabilities[2]:.4f} (Predicted: {labels[prediction]})")

print(f"\nOverall Result: {labels[prediction]}")
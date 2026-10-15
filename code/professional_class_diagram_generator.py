"""
Class Diagram Generator for Sentiment Analysis System

This module generates a professional class diagram visualization of the sentiment analysis
system components using matplotlib. The diagram shows the relationships between key classes
including Dataset, SentimentModel, Trainer, DataPreprocessor, and Visualizer.

Author: Generated for Sentiment Analysis Project
Date: 2026
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
from typing import Dict, List, Tuple

# Class Diagram
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

classes = {
    "Dataset": (2, 8, ["+encodings: dict", "+labels: Series", "+__init__(encodings, labels)", "+__getitem__(idx): dict", "+__len__(): int"]),
    "SentimentModel": (6, 8, ["+tokenizer: AutoTokenizer", "+model: AutoModelForSequenceClassification", "+__init__(model_path)", "+preprocess_text(text): str", "+predict(text): dict"]),
    "Trainer": (2, 5, ["+model: AutoModelForSequenceClassification", "+training_args: TrainingArguments", "+train_dataset: Dataset", "+eval_dataset: Dataset", "+compute_metrics(p): dict", "+train()", "+evaluate(): dict"]),
    "DataPreprocessor": (6, 5, ["+clean_text(text): str", "+encode_labels(sentiments): dict", "+split_data(df, test_size): tuple"]),
    "Visualizer": (4, 2, ["+plot_sentiment_distribution(df)", "+plot_confusion_matrix(y_true, y_pred)", "+plot_roc_curves(y_true, y_proba)", "+generate_word_cloud(texts)"])
}

def draw_class(name, pos, attrs, width=3, height=2):
    x, y = pos
    rect = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.1", facecolor='lightblue', edgecolor='black')
    ax.add_patch(rect)
    ax.text(x, y + height/4, name, ha='center', va='center', fontsize=10, fontweight='bold')
    for i, attr in enumerate(attrs):
        ax.text(x, y - height/4 - i*0.15, attr, ha='center', va='center', fontsize=8)

for name, (x, y, attrs) in classes.items():
    draw_class(name, (x, y), attrs)

# Connections
connections = [
    ((2, 8), (6, 8), "provides data"),
    ((2, 8), (2, 5), "uses for training"),
    ((6, 5), (2, 8), "creates datasets"),
    ((6, 8), (4, 2), "uses for evaluation"),
    ((2, 5), (4, 2), "provides metrics")
]

for (x1, y1), (x2, y2), label in connections:
    con = ConnectionPatch((x1, y1), (x2, y2), "data", "data",
                          arrowstyle="->", shrinkA=20, shrinkB=20,
                          mutation_scale=15, fc="k")
    ax.add_artist(con)
    ax.text((x1 + x2)/2, (y1 + y2)/2 + 0.2, label, ha='center', fontsize=8)

plt.title('Class Diagram of Sentiment Analysis System', fontsize=14, pad=20)
plt.savefig('class_diagram.png', dpi=300, bbox_inches='tight')
plt.close()

print("Class diagram saved as class_diagram.png")
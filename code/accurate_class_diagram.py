import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import textwrap

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Classes based on actual code
classes = {
    "Dataset": (2, 6, ["Attributes:", "- encodings: dict", "- labels: pd.Series", "Methods:", "+ __init__(encodings, labels)", "+ __getitem__(idx): dict", "+ __len__(): int"]),
    "AutoTokenizer": (6, 6, ["From transformers", "Methods:", "+ from_pretrained(model_name)", "+ __call__(texts): encodings"]),
    "AutoModelForSequenceClassification": (10, 6, ["From transformers", "Attributes:", "- num_labels: int", "Methods:", "+ from_pretrained(model_name)", "+ forward(inputs): outputs"]),
    "Trainer": (6, 3, ["From transformers", "Attributes:", "- model", "- args: TrainingArguments", "- train_dataset", "- eval_dataset", "- compute_metrics", "Methods:", "+ train()", "+ evaluate(): dict"]),
    "TrainingArguments": (2, 3, ["From transformers", "Attributes:", "- output_dir: str", "- num_train_epochs: int", "- per_device_train_batch_size: int", "- ..."]),
    "pd.DataFrame": (10, 3, ["From pandas", "Methods:", "+ read_csv(path)", "+ apply(func)", "+ map(dict)"])
}

def draw_class(name, pos, details, width=3, height=2.5):
    x, y = pos
    rect = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.1", facecolor='lightblue', edgecolor='black')
    ax.add_patch(rect)
    ax.text(x, y + height/3, name, ha='center', va='center', fontsize=10, fontweight='bold')
    for i, detail in enumerate(details):
        ax.text(x, y + height/3 - 0.2 - i*0.15, detail, ha='center', va='center', fontsize=8)

for name, (x, y, details) in classes.items():
    draw_class(name, (x, y), details)

# Connections
connections = [
    ((2, 6), (6, 6), "tokenizes data"),
    ((6, 6), (10, 6), "provides input"),
    ((2, 6), (6, 3), "training data"),
    ((10, 6), (6, 3), "model to train"),
    ((2, 3), (6, 3), "training args"),
    ((10, 3), (2, 6), "loads data")
]

for (x1, y1), (x2, y2), label in connections:
    con = ConnectionPatch((x1, y1), (x2, y2), "data", "data",
                          arrowstyle="->", shrinkA=20, shrinkB=20,
                          mutation_scale=15, fc="k")
    ax.add_artist(con)
    ax.text((x1 + x2)/2, (y1 + y2)/2 + 0.1, label, ha='center', fontsize=8)

plt.title('Accurate Class Diagram Based on Code Implementation', fontsize=14, pad=20)
plt.savefig('accurate_class_diagram.png', dpi=300, bbox_inches='tight')
plt.close()

print("Accurate class diagram saved as accurate_class_diagram.png")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import textwrap

fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 16)
ax.axis('off')

# Define positions for subgraphs
subgraphs = {
    "Data Collection": (2, 14),
    "Preprocessing": (2, 10),
    "Model Architecture": (2, 6),
    "Training Setup": (8, 10),
    "Evaluation": (8, 6),
    "Tools": (8, 2)
}

# Draw subgraphs
def draw_subgraph(title, pos, items, width=3.5, height=3):
    x, y = pos
    # Title box
    title_rect = FancyBboxPatch((x - width/2, y + height/2 - 0.3), width, 0.6,
                                boxstyle="round,pad=0.1", facecolor='lightblue', edgecolor='black')
    ax.add_patch(title_rect)
    ax.text(x, y + height/2, title, ha='center', va='center', fontsize=10, fontweight='bold')

    # Item boxes
    for i, item in enumerate(items):
        item_y = y + height/2 - 0.8 - i * 0.5
        item_rect = FancyBboxPatch((x - width/2 + 0.2, item_y - 0.2), width - 0.4, 0.4,
                                   boxstyle="round,pad=0.05", facecolor='white', edgecolor='gray')
        ax.add_patch(item_rect)
        ax.text(x, item_y, '\n'.join(textwrap.wrap(item, 20)), ha='center', va='center', fontsize=8)

# Subgraph contents
contents = {
    "Data Collection": ["Bengali Sentiment Dataset", "Reviews with Overall & Aspect Sentiments"],
    "Preprocessing": ["Text Cleaning", "Negation Handling (e.g., 'valo na' → 'not_good')", "Lowercasing", "Label Encoding (Neg=0, Pos=1, Neu=2)"],
    "Model Architecture": ["BERT-base-multilingual-cased", "12 Layers, 768 Hidden", "Classification Head (3 Classes)", "Max Seq Length: 128"],
    "Training Setup": ["Train/Val Split: 90/10", "Batch Size: 16", "Epochs: 5", "Optimizer: Adam", "Loss: Cross-Entropy"],
    "Evaluation": ["Accuracy", "F1-Score (Weighted)", "Precision/Recall per Class", "Confusion Matrix", "ROC Curves"],
    "Tools": ["Python 3.14", "Transformers 5.5.4", "PyTorch", "Scikit-learn", "Pandas, Matplotlib, Seaborn"]
}

for title, pos in subgraphs.items():
    draw_subgraph(title, pos, contents[title])

# Draw connections
connections = [
    (subgraphs["Data Collection"], subgraphs["Preprocessing"]),
    (subgraphs["Preprocessing"], subgraphs["Model Architecture"]),
    (subgraphs["Model Architecture"], subgraphs["Training Setup"]),
    (subgraphs["Training Setup"], subgraphs["Evaluation"]),
    (subgraphs["Evaluation"], subgraphs["Tools"])
]

for start, end in connections:
    con = ConnectionPatch(start, end, "data", "data",
                          arrowstyle="->", shrinkA=30, shrinkB=30,
                          mutation_scale=15, fc="k")
    ax.add_artist(con)

plt.title('Detailed Methodology Architecture of Sentiment Analysis System', fontsize=14, pad=20)
plt.savefig('detailed_methodology_architecture.png', dpi=300, bbox_inches='tight')
plt.close()

print("Detailed methodology architecture saved as detailed_methodology_architecture.png")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

# Entities
entities = {
    "REVIEW": (2, 6, ["review_id (PK)", "review_text", "overall_sentiment (FK)", "created_at"]),
    "SENTIMENT": (6, 6, ["sentiment_id (PK)", "label (UK)", "type"]),
    "ASPECT_SENTIMENT": (6, 2, ["aspect_id (PK)", "aspect_name (UK)", "sentiment_label"])
}

def draw_entity(name, pos, attrs, width=3, height=2):
    x, y = pos
    rect = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.1", facecolor='lightgreen', edgecolor='black')
    ax.add_patch(rect)
    ax.text(x, y + height/4, name, ha='center', va='center', fontsize=10, fontweight='bold')
    for i, attr in enumerate(attrs):
        ax.text(x, y - height/4 - i*0.15, attr, ha='center', va='center', fontsize=8)

for name, (x, y, attrs) in entities.items():
    draw_entity(name, (x, y), attrs)

# Relationships
# REVIEW to SENTIMENT
con1 = ConnectionPatch((2, 6), (6, 6), "data", "data",
                       arrowstyle="<->", shrinkA=20, shrinkB=20,
                       mutation_scale=15, fc="k")
ax.add_artist(con1)
ax.text(4, 6.5, "has (1:1)", ha='center', fontsize=8)

# REVIEW to ASPECT_SENTIMENT
con2 = ConnectionPatch((2, 6), (6, 2), "data", "data",
                       arrowstyle="->", shrinkA=20, shrinkB=20,
                       mutation_scale=15, fc="k")
ax.add_artist(con2)
ax.text(4, 4, "contains (1:N)", ha='center', fontsize=8)

plt.title('ER Diagram of Sentiment Dataset', fontsize=14, pad=20)
plt.savefig('er_diagram.png', dpi=300, bbox_inches='tight')
plt.close()

print("ER diagram saved as er_diagram.png")
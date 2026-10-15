import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ConnectionPatch

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Entities
entities = {
    "REVIEW": (2, 8, ["review_id (PK)", "review_text", "overall_sentiment_id (FK)"]),
    "SENTIMENT": (6, 8, ["sentiment_id (PK)", "label (UK)", "description"]),
    "ASPECT": (10, 8, ["aspect_id (PK)", "name (UK)", "description"]),
    "REVIEW_ASPECT_SENTIMENT": (6, 4, ["review_id (PK,FK)", "aspect_id (PK,FK)", "sentiment_id (FK)"])
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
# REVIEW to SENTIMENT (overall)
con1 = ConnectionPatch((2, 8), (6, 8), "data", "data",
                       arrowstyle="->", shrinkA=20, shrinkB=20,
                       mutation_scale=15, fc="k")
ax.add_artist(con1)
ax.text(4, 8.3, "overall_sentiment (N:1)", ha='center', fontsize=8)

# REVIEW to REVIEW_ASPECT_SENTIMENT
con2 = ConnectionPatch((2, 8), (6, 4), "data", "data",
                       arrowstyle="->", shrinkA=20, shrinkB=20,
                       mutation_scale=15, fc="k")
ax.add_artist(con2)
ax.text(4, 6, "has (1:N)", ha='center', fontsize=8)

# ASPECT to REVIEW_ASPECT_SENTIMENT
con3 = ConnectionPatch((10, 8), (6, 4), "data", "data",
                       arrowstyle="->", shrinkA=20, shrinkB=20,
                       mutation_scale=15, fc="k")
ax.add_artist(con3)
ax.text(8, 6, "associated_with (1:N)", ha='center', fontsize=8)

# SENTIMENT to REVIEW_ASPECT_SENTIMENT
con4 = ConnectionPatch((6, 8), (6, 4), "data", "data",
                       arrowstyle="->", shrinkA=20, shrinkB=20,
                       mutation_scale=15, fc="k")
ax.add_artist(con4)
ax.text(6, 6, "classified_as (N:1)", ha='center', fontsize=8)

plt.title('Normalized ER Diagram for Sentiment Analysis Database', fontsize=14, pad=20)
plt.savefig('normalized_er_diagram.png', dpi=300, bbox_inches='tight')
plt.close()

print("Normalized ER diagram saved as normalized_er_diagram.png")
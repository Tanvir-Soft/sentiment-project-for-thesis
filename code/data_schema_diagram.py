import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import pandas as pd

# Load data to get exact schema
df = pd.read_csv('data/final_sentiment_dataset.csv')

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Table representation
table_x, table_y = 2, 4
table_width, table_height = 6, 1.5

# Table header
header_rect = FancyBboxPatch((table_x, table_y), table_width, 0.3,
                             boxstyle="round,pad=0.1", facecolor='lightblue', edgecolor='black')
ax.add_patch(header_rect)
ax.text(table_x + table_width/2, table_y + 0.15, 'final_sentiment_dataset.csv', ha='center', va='center', fontsize=12, fontweight='bold')

# Columns
columns = df.columns.tolist()
col_width = table_width / len(columns)

for i, col in enumerate(columns):
    col_x = table_x + i * col_width
    col_rect = FancyBboxPatch((col_x, table_y - 0.3), col_width, 0.3,
                              boxstyle="round,pad=0.05", facecolor='lightgray', edgecolor='black')
    ax.add_patch(col_rect)
    ax.text(col_x + col_width/2, table_y - 0.15, col, ha='center', va='center', fontsize=10, fontweight='bold')

    # Sample values
    sample_vals = df[col].dropna().unique()[:3].tolist()
    for j, val in enumerate(sample_vals):
        val_y = table_y - 0.6 - j * 0.2
        val_rect = FancyBboxPatch((col_x, val_y - 0.1), col_width, 0.2,
                                  boxstyle="round,pad=0.02", facecolor='white', edgecolor='gray')
        ax.add_patch(val_rect)
        ax.text(col_x + col_width/2, val_y, str(val)[:15], ha='center', va='center', fontsize=8)

# Data types
types = ['string'] * len(columns)  # Assuming all strings
for i, typ in enumerate(types):
    col_x = table_x + i * col_width
    ax.text(col_x + col_width/2, table_y - 1.2, f'Type: {typ}', ha='center', va='center', fontsize=8, style='italic')

plt.title('Data Schema Diagram for Sentiment Dataset', fontsize=14, pad=20)
plt.savefig('data_schema_diagram.png', dpi=300, bbox_inches='tight')
plt.close()

print("Data schema diagram saved as data_schema_diagram.png")
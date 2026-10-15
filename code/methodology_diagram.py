import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import textwrap

fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.axis('off')

# Define positions
positions = {
    'Data Collection': (6, 11),
    'Preprocessing': (6, 9.5),
    'Cleaning': (6, 8),
    'Label Mapping': (6, 6.5),
    'Split': (6, 5),
    'Tokenization': (6, 3.5),
    'Model Selection': (6, 2),
    'Training': (3, 0.5),
    'Saving': (9, 0.5),
    'Evaluation': (3, -0.5),
    'Testing': (9, -0.5),
    'Visualization': (6, -1.5),
    'Complete': (6, -3)
}

# Draw boxes
def draw_box(text, pos, width=3, height=0.8, color='lightblue'):
    x, y = pos
    rect = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.1", facecolor=color, edgecolor='black')
    ax.add_patch(rect)
    ax.text(x, y, '\n'.join(textwrap.wrap(text, 15)), ha='center', va='center', fontsize=8)

for label, pos in positions.items():
    if label in ['Data Collection', 'Preprocessing', 'Model Selection', 'Visualization', 'Complete']:
        draw_box(label, pos, color='lightgreen')
    elif label in ['Training', 'Evaluation', 'Testing']:
        draw_box(label, pos, color='lightyellow')
    else:
        draw_box(label, pos)

# Draw arrows
def draw_arrow(start, end):
    con = ConnectionPatch(start, end, "data", "data",
                          arrowstyle="->", shrinkA=20, shrinkB=20,
                          mutation_scale=15, fc="k")
    ax.add_artist(con)

# Connections
draw_arrow(positions['Data Collection'], positions['Preprocessing'])
draw_arrow(positions['Preprocessing'], positions['Cleaning'])
draw_arrow(positions['Cleaning'], positions['Label Mapping'])
draw_arrow(positions['Label Mapping'], positions['Split'])
draw_arrow(positions['Split'], positions['Tokenization'])
draw_arrow(positions['Tokenization'], positions['Model Selection'])
draw_arrow(positions['Model Selection'], positions['Training'])
draw_arrow(positions['Model Selection'], positions['Saving'])
draw_arrow(positions['Training'], positions['Evaluation'])
draw_arrow(positions['Saving'], positions['Testing'])
draw_arrow(positions['Evaluation'], positions['Visualization'])
draw_arrow(positions['Testing'], positions['Visualization'])
draw_arrow(positions['Visualization'], positions['Complete'])

plt.title('Methodology Flowchart of Sentiment Analysis System', fontsize=14, pad=20)
plt.savefig('methodology_flowchart.png', dpi=300, bbox_inches='tight')
plt.close()

print("Methodology flowchart saved as methodology_flowchart.png")
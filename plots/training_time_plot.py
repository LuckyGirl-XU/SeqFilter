import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

# Use a non-interactive backend
plt.switch_backend('agg')

# Read the data
df = pd.read_csv('training_times_analysis.csv')

# Convert to pivot table and convert seconds to hours
pivot_df = df.pivot(index='Method', columns='Dataset', values='Avg Time (s)')
pivot_df = pivot_df / 3600  # Convert seconds to hours

# Define a color palette using seaborn
colors_palette = sns.color_palette("coolwarm", 8)
colors = {
    'JODIE': colors_palette[0],
    'DyRep': colors_palette[1],
    'TGAT': colors_palette[2],
    'TGN': colors_palette[3],
    'CAWN': colors_palette[4],
    'TCL': colors_palette[5],
    'GraphMixer': colors_palette[6],
    'DyGFormer': colors_palette[7]
}

# Update font sizes
parameters = {"xtick.labelsize": 25, 'ytick.labelsize': 25}
plt.rcParams.update(parameters)

# Set up the figure with three subplots
fig, axs = plt.subplots(1, 3, figsize=(20, 7), gridspec_kw={'width_ratios': [1, 1, 0.9]})

# Create bar plots for each dataset
def create_subplot(ax, data, title):
    methods = ['JODIE', 'DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']
    values = [data[m] for m in methods]
    valid_indices = [i for i, v in enumerate(values) if not np.isnan(v)]
    valid_methods = [methods[i] for i in valid_indices]
    valid_values = [values[i] for i in valid_indices]
    
    # Create positions for bars with consistent width
    x_pos = np.arange(len(valid_methods))
    # width = 0.8  # Fixed width for all bars
    
    bars = ax.bar(x_pos, valid_values,  color=[colors[m] for m in valid_methods], edgecolor='black')
    
    # Set x-ticks and labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(valid_methods)
    
    # Set x-axis limits to keep bars close together
    ax.set_xlim(-0.5, len(valid_methods) - 0.5)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f'{height:.1f}h', ha='center', va='bottom', fontsize=20)
    
    # Set y-axis to hours and move title below the plot
    ax.set_ylim(0, max(valid_values) * 1.2)
    ax.text(0.5, -0.08, title, ha='center', va='center', fontsize=25, transform=ax.transAxes)
    ax.set_ylabel('Training Time / Epoch', fontsize=25)
    ax.tick_params(axis='x', labelbottom=False)

# Create subplots
datasets = ['GoogleLocal', 'Patent', 'Yelp']
for ax, dataset in zip(axs, datasets):
    create_subplot(ax, pivot_df[dataset], dataset)

# Create legend
legend_handles = [Patch(color=colors[method], label=method) for method in colors.keys()]
fig.legend(handles=legend_handles, loc='upper center', ncol=len(colors), 
          fontsize=22, bbox_to_anchor=(0.5, 0.85))

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.7, bottom=0.05)

# Save the figure
plt.savefig('training_times.pdf', bbox_inches='tight', dpi=300)
plt.close()
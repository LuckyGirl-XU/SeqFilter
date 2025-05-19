import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Use a non-interactive backend
plt.switch_backend('agg')
# Preparing the plot with datasets as keys in the adjusted MRR values dictionary
fig, ax = plt.subplots(figsize=(22, 10))

# Updated MRR values dictionary with datasets as keys
adjusted_mrr_values = {
    'Wikipedia': {'EdgeBank': 78.10, 'GraphMixer': 72.14, 'DyGFormer': 84.64, 'SGNN-HN': 83.83},
    'Reddit': {'EdgeBank': 78.08, 'GraphMixer': 71.73, 'DyGFormer': 83.57, 'SGNN-HN': 89.01},
    # 'Yelp': {'EdgeBank': 9.77, 'GraphMixer': 33.96, 'DyGFormer': 21.68, 'SGNN-HN': 69.34},
    'GoogleLocal': {'EdgeBank': 1.96, 'GraphMixer': 21.31, 'DyGFormer': 18.39, 'SGNN-HN': 62.88},
    'ML-20M': {'EdgeBank': 1.82, 'GraphMixer': 21.97, 'DyGFormer': None, 'SGNN-HN': 33.12},
    # 'Taobao': {'EdgeBank': None, 'GraphMixer': 31.54, 'DyGFormer': None, 'SGNN-HN': 63.37}  # None represents missing data
}

# List of datasets in the desired order
datasets = ['Wikipedia', 'Reddit', 'GoogleLocal', 'ML-20M']

# Configuring bar positions
bar_width = 0.1
gap_width = 0.05  # Width of the gap between dataset groups
x_base_positions = np.arange(len(datasets)) * (4 * bar_width + gap_width)  # Calculate group positions with gaps
colors = ['#2a2a2a', 'grey', '#ec9f8c', '#db7b71']
methods = ['SGNN-HN', 'EdgeBank', 'GraphMixer', 'DyGFormer', ]
# colors = sns.color_palette("GnBu", len(methods))
# Plotting each method's bars
for i, method in enumerate(methods):
    for j, dataset in enumerate(datasets):
        if adjusted_mrr_values[dataset][method] is not None:  # Only plot if data is not None
            x_position = x_base_positions[j] + (i * bar_width)
            y_value = adjusted_mrr_values[dataset][method]
            ax.bar(x_position, y_value, bar_width, label=method if j == 0 else "", color=colors[i],edgecolor = 'black')  # Label only once for the legend

            # Add value label on top of the bar
            ax.text(x_position, y_value + 1, f'{round(y_value)}', ha='center', va='bottom', fontsize=40)


# Adding labels and title
# ax.set_xlabel('Datasets', fontsize=60)
ax.set_ylabel('MRR (%)', fontsize=60)
# ax.set_title('MRR across different datasets', fontsize=25)

# Setting x-axis ticks and labels
ax.set_xticks(x_base_positions + 1.5 * bar_width)
ax.set_xticklabels(datasets, fontsize=60)
ax.tick_params(axis='y', labelsize=40)

# Adding grid and legend
# ax.grid(True, axis='y', linestyle='--', alpha=0.6)
legend=ax.legend(fontsize=30, loc='upper right', bbox_to_anchor=(1, 1),handlelength=1.2)
frame = legend.get_frame()
frame.set_linewidth(0.15) 


ax.set_ylim(0,105)
# Adjusting layout for visual clarity
plt.tight_layout()

# Save the plot
plt.savefig('mrr_bar_chart.pdf', bbox_inches='tight', dpi=300)
plt.show()

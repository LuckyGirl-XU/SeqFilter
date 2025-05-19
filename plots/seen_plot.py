import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties

# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
# config = {
#     "font.family": 'serif',
#     # "font.size": 20,
#     "mathtext.fontset": 'stix',
#     "font.serif": ['Times New Roman'],  # simsun字体中文版就是宋体
# }
# matplotlib.rcParams.update(config)

def parse_data_to_datadict(input_text):
	data_dict = {}
	current_dataset = None

	# 分割输入文本并进行迭代
	for line in input_text.strip().splitlines():
		line = line.strip()

		# 跳过空行
		if not line:
			continue

		# 如果行以 "dataset" 开头，提取数据集名称
		if line.startswith('dataset'):
			current_dataset = line.split(maxsplit=1)[1]
			data_dict[current_dataset] = {}

		# 否则是方法行
		else:
			parts = line.split('\t', maxsplit=1)
			method = parts[0]
			values = parts[1]
			hist, nonhist = map(float, values.split('\t'))

			data_dict[current_dataset][method] = {'Hist': hist, 'NonHist': nonhist}

	data_dict = {dataset: pd.DataFrame.from_dict(methods, orient='index') for dataset, methods in data_dict.items()}

	return data_dict

def plot_stacked_bar_charts(data_dict, dataset_names, method_names, filename):
	font_size = 33
	# For legend
	legend_size = 28
	# For y ticks
	tick_size = 30
	# For MRR values
	value_size = 21

	times_new_roman_path = 'Times New Roman.ttf'
	font_prop = FontProperties(size = 38)
	legend_prop = FontProperties(size = 28)
	value_prop = FontProperties(size = value_size)

	sns.set_style('ticks')
	colors = sns.color_palette("coolwarm", len(method_names) + 1)
	colors = [colors[i] for i in range(len(colors)) if i not in [len(method_names) // 2]]  # 去掉中间的一些颜色
	

	bar_width = 0.5
	num_methods = len(method_names)
	x = np.arange(0, num_methods, bar_width)
	fig, axs = plt.subplots(1, len(dataset_names), figsize = (7 * len(dataset_names), 6), sharey = False)
	y_limits = [(0, 110), (0, 110), (0, 55), (0, 55)]
	for i, dataset_name in enumerate(dataset_names):
		ax = axs[i]
		data = data_dict[dataset_name]

		for j, method in enumerate(method_names):
			hist_value = data.loc[method, 'Hist']
			ax.bar(x[j], hist_value, bar_width, label = f'{method}' if i == 0 else '', color = colors[j], edgecolor = 'black', alpha = .7)
			ax.text(x[j], hist_value, f'${int(hist_value)}$', ha = 'center', va = 'bottom', fontsize = value_size, fontproperties = value_prop)

			nonhist_value = data.loc[method, 'NonHist']
			ax.bar(x[j], nonhist_value, bar_width, color = 'none', edgecolor = 'black', alpha = 1.0, hatch ='xxx')
			if dataset_name != 'Social Evo.' or method != 'TCL':
				ax.text(x[j], nonhist_value, f'${int(nonhist_value)}$', ha = 'center', va = 'bottom', fontsize = value_size, fontproperties = value_prop)
			else:
				print
		ax.set_title(dataset_name, y = -.15, fontproperties = font_prop)
		ax.set_xticks([])
		ax.tick_params(axis = 'y', labelsize = tick_size)
		ax.set_ylim(y_limits[i])


	axs[0].set_ylabel('MRR(%)', fontproperties = font_prop)
	# axs[2].set_ylabel('MRR(%)', fontproperties = font_prop)
	# axs[2].set_ylim(0, 60)
	# axs[3].set_ylim(0, 60)


	# 图例
	method_handles = [mpatches.Patch(color = colors[i], label = method_names[i]) for i in range(num_methods)]
	nonhist_patch = mpatches.Patch(facecolor = 'none', edgecolor = 'black', hatch = 'xxx', label = 'Unseen')

	fig.legend(handles = method_handles + [nonhist_patch], loc = 'upper center', ncol = num_methods + 1, bbox_to_anchor = (0.5, 1.05), frameon = False, prop = legend_prop)
	
	plt.tight_layout(rect = [0, 0, 1, 0.9])

	# 保存为 PDF/PGF 文件
	plt.savefig(filename, bbox_inches = 'tight', dpi = 300)
	plt.close()

input_text = """
dataset Wikipedia
JODIE	82.63	24.43
DyRep	73.96	16.99
TGAT	78.53	17.79
TGN	83.55	12.87
CAWN	96.80	16.50
TCL	52.76	15.60
GraphMixer	79.00	23.07
DyGFormer	93.76	8.77

dataset Reddit
JODIE	77.08	21.74
DyRep	80.18	24.95
TGAT	81.30	23.29
TGN	84.47	26.50
CAWN	92.42	17.89
TCL	40.36	12.45
GraphMixer	74.97	23.25
DyGFormer	90.73	6.55

dataset Social Evo.
JODIE	27.67	11.75
DyRep	16.18	8.36
TGAT	46.24	10.15
TGN	39.83	9.96
CAWN	46.35	7.13
TCL	12.94	12.03
GraphMixer	47.07	9.98
DyGFormer	41.01	5.11

dataset Enron
JODIE	24.34	8.80
DyRep	25.97	8.30
TGAT	17.16	6.51
TGN	15.06	7.42
CAWN	48.29	9.19
TCL	9.17	4.62
GraphMixer	25.54	8.96
DyGFormer	45.71	8.02
"""

data_dict = parse_data_to_datadict(input_text)

plot_stacked_bar_charts(data_dict, ['Wikipedia', 'Reddit', 'Social Evo.', 'Enron'], ['JODIE', 'DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer'], filename = 'fig.pdf')
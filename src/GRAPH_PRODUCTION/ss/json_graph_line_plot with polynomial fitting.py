import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import copy
from format_json import *
import numpy as np
from pprint import pprint
from math import log10, floor

def round_sig(x, sig=2):
    if x == 0:
        return x
    return round(x, sig-int(floor(log10(abs(x))))-1)

# file_name = 'tune_lr_lr_decay_together_grid_.json'
title = 'Validation Accuracy Sensitivity to Frame Differencing and Preprocessing'
y_data = 'val_accuracy'
y_axis_title = 'Validation accuracy'
# my_label_data1 = 'Learning Rate'
# my_label_data2 = 'lr_decay'
x_axis_title = 'Epoch Number'
x_data = 'epoch'
graph_name = 'Preprocessing and Frame Difference Analysis.eps'

cwd = os.getcwd()

# file_path = os.path.join(cwd, 'IMPORTANT_RESULTS', 'JSON_LOGS', 'FD_IM_SIZE_RES_V1')
file_path = os.path.join(cwd, 'IMPORTANT_RESULTS', 'IMAGE_SIZE_AND_FD_VS_RAW_RESULTS', 'old_json', 'FD and Preprocessing')

fig, ax = plt.subplots()
colors = []

for file in os.listdir(file_path):
    if file.endswith('.json'):

        my_data = reformat_json(os.path.join(file_path, file))
        my_data = json.dumps(my_data)
        my_data = json.loads(my_data)
        x_data_set = [i[x_data] for i in my_data['data']]
        y_data_set = [i[y_data] for i in my_data['data']]

        poly_deg = 14
        coefs = np.polyfit(x_data_set, y_data_set, poly_deg)

        x = np.linspace(x_data_set[0], x_data_set[-1] , len(x_data_set))

        y_data_set = np.polyval(coefs, x)

        df = pd.DataFrame({y_axis_title: y_data_set, x_axis_title: x_data_set})

        file_variables = extract_filename_data(file)
        my_label_data1 = file_variables[9]#.split('-')[1]
        my_label_data2 = file_variables[-1]

        line, = ax.plot(x_data_set, y_data_set)
        my_color = plt.getp(line, 'color')
        name1 = my_label_data1
        name2 = my_label_data2
        # name1 = round_sig(name1, 2)
        # name2 = label_data2[i]
        # name2 = round_sig(name2, 2)
        my_label = name1 + " " + name2[:-5]

        ax.text(x_data_set[-1], y_data_set[-1], my_label, horizontalalignment='left', size='small', color=my_color, fontsize=10)



ax.set_title(title)
ax.set_xlabel(x_axis_title)
ax.set_ylabel(y_axis_title)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0.6, 1.0)

fig.savefig(graph_name, bbox_inches='tight', format='eps', dpi=1200)
plt.show()
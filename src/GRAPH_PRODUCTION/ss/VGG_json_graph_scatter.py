import matplotlib.pyplot as plt
import json
import os
from format_json import *
import cv2
import random
import numpy as np
import pandas as pd
from pprint import pprint


def scatter_generator(my_path):
    file_name = '24_vgg_netoptimiseLR-3_C1-5_C2-6_DS-8_D-6_M-0.9_IM-250_FD-True_Time_Stamp-1524317471.7931182_Preprop-True.json'
    title = 'Validation Accuracy for VGG'
    y_data1 = 'val_accuracy'
    y_data2 = 'accuracy'
    y_axis_title = 'Accuracy'
    x_data = 'epoch'
    x_axis_title = 'Epoch'
    graph_name = 'Validation accuracy for VGG.eps'

    file_path = os.path.join(my_path, file_name)

    my_data = reformat_json(file_path)
    my_data = json.dumps(my_data)

    my_data = json.loads(my_data)
    y_data1 = [i[y_data1] for i in my_data["data"]]
    y_data2 = [i[y_data2] for i in my_data["data"]]
    x_data = [i[x_data] for i in my_data["data"]]


    fig, ax = plt.subplots()

    ax.scatter(x_data, y_data1, marker="+", s=15, linewidths=1, label="Val Acc")
    ax.scatter(x_data, y_data2, marker="*", s=15, linewidths=1, label="Acc")

    dx = (max(x_data) - min(x_data))*0.1
    # dy = (max(y_data1) - min(y_data1))*0.1

    # ax.set_xlim(min(x_data)-dx, max(x_data)+dx)
    ax.legend(loc=4)
    ax.set_xlim(0, max(x_data)+dx)
    # ax.set_ylim(min(y_data1)-dy, max(y_data1)+dy)

    ax.set_title(title)
    ax.set_xlabel(x_axis_title)
    ax.set_ylabel(y_axis_title)
    ax.set_ylim(0.8,)
    fig.savefig(graph_name, bbox_inches='tight', format= 'eps', dpi= 1200)
    plt.show()
    cv2.waitKey(0)



cwd = os.getcwd()
print(cwd)
root_path = os.path.join(cwd, 'IMPORTANT_RESULTS', 'LR_vs_LR_DECAY_RESULTS', 'old_json')

scatter_generator(root_path)

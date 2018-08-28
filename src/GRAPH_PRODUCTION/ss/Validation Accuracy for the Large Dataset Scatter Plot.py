import matplotlib.pyplot as plt
import json
import os
from format_json import *

import pandas as pd
from pprint import pprint


def scatter_generator(my_path):
    my_files = os.listdir(my_path)
    for file in my_files:
        if file.endswith('.json'):
            file_name = file
            title = 'Validation Accuracy for 400 Classes'
            y_data1 = 'val_accuracy'
            y_data2 = 'accuracy'
            y_axis_title = 'Accuracy'
            x_data = 'epoch'
            x_axis_title = 'Epoch'
            graph_name = 'graph_' + file_name[-14:-5] + ".eps"

            file_path = os.path.join(my_path, file_name)
            print(file_path)

            # my_data = json.load(open(file_path))
            my_data = reformat_json(file_path)
            my_data = json.dumps(my_data)

            my_data = json.loads(my_data)
            y_data1 = [i[y_data1] for i in my_data["data"]]
            y_data2 = [i[y_data2] for i in my_data["data"]]
            x_data = [i[x_data] for i in my_data["data"]]
            print(my_data)


            # df = pd.DataFrame({y_axis_title: y_data, x_axis_title: x_data})
            # print(df.sort_values(by=x_axis_title))

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

            fig.savefig(graph_name, bbox_inches='tight', format='eps', dpi=1200)


cwd = os.getcwd()
print(cwd)
root_path = os.path.join(cwd, 'IMPORTANT_RESULTS', 'LARGER_DATASET')

scatter_generator(root_path)

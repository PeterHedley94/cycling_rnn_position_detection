import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
from src.GRAPH_PRODUCTION.load_data import *
from src.GRAPH_PRODUCTION.heat_map import *
from src.common import *
import numpy as np


###########################################
#Setup


############################################
#Pose Data - Bayesian
filename = "bayes_opt_pose_all_results.txt"
data = get_bayes_json(filename)
x_data = [val['lstm_1_size'] for val in data]
y_data = [val['lstm_2_size'] for val in data]

filename = "bayes_opt_pose_all_values.npy"
z_data = -get_numpy_data(filename)

title = 'Validation loss sensitivity to LSTM layer size'
x_axis_title = 'LSTM Layer 1 Size'
y_axis_title = 'LSTM Layer 2 Size'
z_axis_title = 'Validation Loss'

dz = (max(z_data) - min(z_data))*0.1
bounds = np.arange(min(z_data) - dz, max(z_data) + dz, 0.01)

graph_name = os.path.join(GRAPH_OUTPUT_FOLDER,'pose_lstm1_lstm2_graph_.png')

heat_map_params = {'title':title,'x_data':x_data
                    ,'y_data':y_data,'z_data':z_data,'x_axis_title':x_axis_title,
                    'y_axis_title':y_axis_title,'z_axis_title':z_axis_title,
                    'graph_name':graph_name,'bounds':bounds}

generate_heat_map(**heat_map_params)

############################################
#Pose Data - Learning Rate and Decay
filename = "lr_pose.txt"
data = get_lr_decay_data(filename)
x_data = data[:,0]
y_data = data[:,1]
z_data = -data[:,2]

title = 'Optimise LR and LR Decay Pose Network'
x_axis_title = 'Learning Rate (10**x)'
y_axis_title = 'Learning Rate Decay'
z_axis_title = 'Validation Loss'

bounds = np.linspace(0, 10, 10).tolist()
dz = (max(z_data) - min(z_data))*0.1
bounds.extend([20,30,40,50])
bounds.append(max(z_data)+max(z_data)*0.01)
print(bounds)

graph_name = os.path.join(GRAPH_OUTPUT_FOLDER,'pose_lr_lr_decay_graph_.png')

heat_map_params = {'title':title,'x_data':x_data
                    ,'y_data':y_data,'z_data':z_data,'x_axis_title':x_axis_title,
                    'y_axis_title':y_axis_title,'z_axis_title':z_axis_title,
                    'graph_name':graph_name,'bounds':bounds}

generate_heat_map(**heat_map_params)

###############################################
#Images Data - Learning Rate and Decay
filename = "lr_images.txt"
data = get_lr_decay_data(filename)
x_data = data[:,0]
y_data = data[:,1]
z_data = -data[:,2]

title = 'Optimise LR and LR Decay Images Network'
x_axis_title = 'Learning Rate (10**x)'
y_axis_title = 'Learning Rate Decay'
z_axis_title = 'Validation Loss'


bounds = np.linspace(0, 30, 10).tolist()
dz = (max(z_data) - min(z_data))*0.1
bounds.extend([40,50,60,70,80])
bounds.append(max(z_data)+max(z_data)*0.01)

graph_name = os.path.join(GRAPH_OUTPUT_FOLDER,'images_lr_lr_decay_graph_.png')

heat_map_params = {'title':title,'x_data':x_data
                    ,'y_data':y_data,'z_data':z_data,'x_axis_title':x_axis_title,
                    'y_axis_title':y_axis_title,'z_axis_title':z_axis_title,
                    'graph_name':graph_name,'bounds':bounds}

generate_heat_map(**heat_map_params)

###############################################
'''
#Pose Model vs linear model , time gap
filename = "time_gap_pose_results.txt"
data = get_lr_decay_data(filename)
ax.text(x_data_set[-1], y_data_set[-1], my_label, horizontalalignment='left', size='small', color=my_color, fontsize=10)
ax.bar(x, height, *, align='center', **kwargs)
ax.set_title(title)
ax.set_xlabel(x_axis_title)
ax.set_ylabel(y_axis_title)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0.6, 1.0)
fig.savefig(graph_name, bbox_inches='tight', format='eps', dpi=1200)
plt.show()
'''

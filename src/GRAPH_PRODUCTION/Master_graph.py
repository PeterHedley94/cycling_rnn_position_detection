import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
from src.GRAPH_PRODUCTION.load_data import *
from src.GRAPH_PRODUCTION.heat_map import *
from src.common import *


###########################################
#Setup


############################################
#Pose Data - Bayesian
filename = "bayes_opt_pose_all_results.txt"
data = get_bayes_json(filename)
x_data = [val['lstm_1_size'] for val in data]
y_data = [val['lstm_2_size'] for val in data]

filename = "bayes_opt_all_values.npy"
z_data = -get_numpy_data(filename)

title = 'Validation loss sensitivity to LSTM layer size'
x_axis_title = 'LSTM Layer 1 Size'
y_axis_title = 'LSTM Layer 2 Size'
z_axis_title = 'Validation Loss'

graph_name = os.path.join(GRAPH_OUTPUT_FOLDER,'pose_lstm1_lstm2_graph_.png')

heat_map_params = {'title':title,'x_data':x_data
                    ,'y_data':y_data,'z_data':z_data,'x_axis_title':x_axis_title,
                    'y_axis_title':y_axis_title,'z_axis_title':z_axis_title,
                    'graph_name':graph_name}

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

graph_name = os.path.join(GRAPH_OUTPUT_FOLDER,'pose_lr_lr_decay_graph_.png')

heat_map_params = {'title':title,'x_data':x_data
                    ,'y_data':y_data,'z_data':z_data,'x_axis_title':x_axis_title,
                    'y_axis_title':y_axis_title,'z_axis_title':z_axis_title,
                    'graph_name':graph_name}

generate_heat_map(**heat_map_params)

###############################################
#Images Data - Learning Rate and Decay
filename = "lr_rcnn.txt"
data = get_lr_decay_data(filename)
x_data = data[:,0]
y_data = data[:,1]
z_data = -data[:,2]

title = 'Optimise LR and LR Decay Images Network'
x_axis_title = 'Learning Rate (10**x)'
y_axis_title = 'Learning Rate Decay'
z_axis_title = 'Validation Loss'

graph_name = os.path.join(GRAPH_OUTPUT_FOLDER,'images_lr_lr_decay_graph_.png')

heat_map_params = {'title':title,'x_data':x_data
                    ,'y_data':y_data,'z_data':z_data,'x_axis_title':x_axis_title,
                    'y_axis_title':y_axis_title,'z_axis_title':z_axis_title,
                    'graph_name':graph_name}

generate_heat_map(**heat_map_params)

###############################################

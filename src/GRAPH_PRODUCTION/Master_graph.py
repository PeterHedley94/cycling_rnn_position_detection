import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
from src.GRAPH_PRODUCTION.load_data import *
from src.GRAPH_PRODUCTION.heat_map import *
from src.GRAPH_PRODUCTION.bar_plot import *
from src.GRAPH_PRODUCTION.training_graph import *
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
x_axis_title = 'LSTM Layer 1 Size 2**x'
y_axis_title = 'LSTM Layer 2 Size 2**x'
z_axis_title = 'Validation Loss'

x = []
y = []
z = []

for x_,y_,z_ in zip(x_data,y_data,z_data):
    if z_ < 200:
        x.append(x_)
        y.append(y_)
        z.append(z_)


bounds = [0,1,2,3,4,5,6,7,8,9,10,25,50,75,100,200]#np.arange(min(z_data) - dz, max(z_data) + dz, 0.1)

graph_name = os.path.join(GRAPH_OUTPUT_FOLDER,'pose_lstm1_lstm2_graph_.eps')

heat_map_params = {'title':title,'x_data':x
                    ,'y_data':y,'z_data':z,'x_axis_title':x_axis_title,
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

graph_name = os.path.join(GRAPH_OUTPUT_FOLDER,'pose_lr_lr_decay_graph_.eps')

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

graph_name = os.path.join(GRAPH_OUTPUT_FOLDER,'images_lr_lr_decay_graph_.eps')

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

x_data = data[:,0]
y1_data = data[:,1]
y2_data = data[:,2]


title = 'Time gap vs Validation Loss: Pose NN Model'
x_axis_title = 'Time Gap'
y_axis_title = 'Validation Loss'
graph_name = os.path.join(GRAPH_OUTPUT_FOLDER,'pose_nn_vs_time_gap.eps')
make_barplot(title,x_data,y1_data,x_axis_title,y_axis_title,graph_name)
title = 'Time gap vs Validation Loss: Linear Model'
graph_name = os.path.join(GRAPH_OUTPUT_FOLDER,'pose_linear_vs_time_gap.eps')
make_barplot(title,x_data,y2_data,x_axis_title,y_axis_title,graph_name)
'''


###############################################
title = "Optimisation of Images Network"
filename = "bayes_opt_images_all_results.txt"
data = get_bayes_json(filename)
x1_data = [val['conv1_size'] for val in data]
x2_data = [val['conv2_size'] for val in data]
x3_data = [val['no_layers'] for val in data]

filename = "bayes_rcnn_opt_all_values.npy"
z_data = -get_numpy_data(filename)

data = np.array([x1_data,x2_data,x3_data]).reshape((-1,3))
dim_names = ["Conv 1 Size", "Conv 2 Size","No Layers"]
z_axis_title = "Validation Loss"
graph_name = os.path.join(GRAPH_OUTPUT_FOLDER,"opt_bayes_images_nn.eps")
generate_3D_heat_map(title,data,z_data,dim_names,z_axis_title,graph_name)

################################################
#Training Curves

#Pose Model
filename = os.path.join(JSON_OLD_LOGS_FOLDER,"1_pose_loss.json")
data = get_bayes_json(filename)
x_data = [val['epoch'] for val in data]
val_loss = [val['val_loss'] for val in data]
train_loss = [val['loss'] for val in data]
x_axis_title = "Epoch"
y_axis_title = "MSE"
title = "Pose Model Training Curve"
graph_name = os.path.join(GRAPH_OUTPUT_FOLDER,"Pose_training_curve.eps")
plot_training_curve(x_data,train_loss,val_loss,x_axis_title,y_axis_title,title,graph_name)

#Images models
filename = os.path.join(JSON_OLD_LOGS_FOLDER,"0_Images_model.json")
data = get_bayes_json(filename)
x_data = [val['epoch'] for val in data]
val_loss = [val['val_loss'] for val in data]
train_loss = [val['loss'] for val in data]
x_axis_title = "Epoch"
y_axis_title = "MSE"
title = "Images Model Training Curve"
graph_name = os.path.join(GRAPH_OUTPUT_FOLDER,"Images_training_curve.eps")
plot_training_curve(x_data,train_loss,val_loss,x_axis_title,y_axis_title,title,graph_name)


#Images models
filename = os.path.join(JSON_OLD_LOGS_FOLDER,"0_total_model.json")
data = get_bayes_json(filename)
x_data = [val['epoch'] for val in data]
val_loss = [val['val_loss'] for val in data]
train_loss = [val['loss'] for val in data]
x_axis_title = "Epoch"
y_axis_title = "MSE"
title = "Images Model Training Curve"
graph_name = os.path.join(GRAPH_OUTPUT_FOLDER,"total_training_curve.eps")
plot_training_curve(x_data,train_loss,val_loss,x_axis_title,y_axis_title,title,graph_name)

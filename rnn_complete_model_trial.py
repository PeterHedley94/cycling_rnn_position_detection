#!/usr/bin/env python
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
#parentdir = os.path.dirname(parentdir)
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization
import numpy as np,math,json
from src.NN_MODELS.rnn_complete_model import *
from src.DATA_PREPARATION.pretend_data import *
from src.DATA_PREPARATION.data_generator import *
from src.common import *
from src.callbacks import *



try:
    send_slack_message("Started training complete model")


    conv1_size = 5
    conv2_size = 5
    no_layers=4
    opt = 3
    lr=-6
    decay = 0
    images = True
    def get_params(time_gap,batch_size,directory):
        #batch_size = int(math.floor(990/time_gap))
        params_train = {'dir': directory,
                  'batch_size': batch_size,'time_gap':time_gap,
                  'shuffle': True,'debug_mode':False,
                    'sequence_length': sequence_length ,'time_distributed' : True,
                    'images':images,'image_height':IM_HEIGHT,'image_width':IM_WIDTH,
                    'no_image_channels':NUMBER_CHANNELS,'camera_model_location':camera_model_location}
        return params_train


    params_train = get_params(time_gap = 15,batch_size=batch_size,directory=train_directory)
    params_val = get_params(time_gap = 15,batch_size=batch_size,directory=validation_directory)

    #train model
    epochs = 500
    model = rcnn_total_model(number_outputs,sequence_length,conv1_size,conv2_size,no_layers,opt,lr,decay)
    model.train(epochs,params_train,params_val)
except:
    send_slack_message("Complete model failed")

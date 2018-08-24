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
from src.NN_MODELS.rnn_pose_model import *
from src.DATA_PREPARATION.pretend_data import *
from src.DATA_PREPARATION.data_generator import *

sequence_length_ = 5
sequence_length = 5
data_dims = (4,3)
number_outputs = 3
batch_size = 4
epochs = 100
images = False
image_height = 480#180
image_width= 640#240
no_image_channels=3
camera_model_location = os.path.join("utils",'test_camera_model.json')
directory = "/home/peter/Documents/okvis_drl/build/blackfriars1_dataset"#"/home/peter/Tests"


def get_params(time_gap):
    params_train = {'dir': directory,
                  'batch_size': int(math.floor(990/time_gap)),'time_gap':time_gap,
                  'shuffle': True,'debug_mode':False,
                    'sequence_length': sequence_length ,'time_distributed' : True,
                    'images':images,'image_height':image_height,'image_width':image_width,
                    'no_image_channels':no_image_channels,'camera_model_location':camera_model_location}
    return params_train


def get_vals(lstm_1_size = 5,lstm_2_size = 6,opt = 0.9,lr=10**(-6)):
    params_train = get_params(10)
    train_generator = DataGenerator(**params_train)
    train_gen = train_generator.generate()
    x_,y_ = train_gen.__next__()
    x = x_.reshape((-1,sequence_length,data_dims[0]*data_dims[1]))
    y = y_
    model = rnn_model(number_outputs,sequence_length,data_dims,lstm_1_size,lstm_2_size,opt,lr)
    model.train(x,y,epochs)
    val = model.test(x,y)
    if math.isnan(val):
        return -10.0**100
    else:
        return -val.astype(np.int64)


bo = BayesianOptimization(lambda lstm_1_size,lstm_2_size,opt,lr: get_vals(lstm_1_size,lstm_2_size,opt,lr),
                          {"lstm_1_size":(4,6),"lstm_2_size":(4,6),"opt":(0,3),"lr":(-8,-3)})
bo.explore({"lstm_1_size":(4,8),"lstm_2_size":(4,8),"opt":(0,3),"lr":(-8,-3)})
bo.maximize(init_points=2, n_iter=300, kappa=10,acq="ucb") #, acq="ucb"


json.dump(bo.res['max'], open("bayes_opt_results.txt",'w'))
print(bo.res['all'])

np.save("bayes_opt_all_values",bo.res['all']['values'])

for i in bo.res['all']['params']:
    json.dump(i, open("bayes_opt_all_results.txt",'a'))

epochs = 2000

with open("time_gap_results.txt","w") as file:
    file.write("Time_gap,nn,linear_model\n")
    for time_gap in range(1,100,5):
        print("On time gap " + str(time_gap))
        params_train = get_params(time_gap)
        train_generator = DataGenerator(**params_train)
        train_gen = train_generator.generate()
        x_,y_ = train_gen.__next__()
        x = x_.reshape((-1,sequence_length,data_dims[0]*data_dims[1]))
        y = y_
        model = rnn_model(number_outputs,sequence_length,data_dims)
        model.train(x,y,epochs)
        results = [time_gap]
        results.append(model.test(x,y))
        x = x_.reshape((-1,sequence_length*data_dims[0]*data_dims[1]))
        y = y_.reshape((-1,3))
        reg = linear_model.LinearRegression()
        reg.fit(x,y)
        results.append(mean_squared_error(y, reg.predict(x)))
        file.write(str(results[0]) + "," + str(results[1]) + "," + str(results[2]) + "\n")

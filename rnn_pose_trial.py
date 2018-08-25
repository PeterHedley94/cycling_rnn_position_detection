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
from src.common import *
from src.callbacks import *

send_slack_message("Started optimising pose model")

try:
    sequence_length_ = 5
    sequence_length = 5
    pose_data_dims = (4,3)
    number_outputs = 3
    batch_size = 4
    epochs = 100
    images = False
    image_height = 480#180
    image_width= 640#240
    no_image_channels=3
    camera_model_location = os.path.join("utils",'test_camera_model.json')


    def get_params(time_gap,directory):
        params_train = {'dir': directory,
                      'batch_size': int(math.floor(990/time_gap)),'time_gap':time_gap,
                      'shuffle': True,'debug_mode':False,
                        'sequence_length': sequence_length ,'time_distributed' : True,
                        'images':images,'image_height':image_height,'image_width':image_width,
                        'no_image_channels':no_image_channels,'camera_model_location':camera_model_location}
        return params_train


    def get_vals(lstm_1_size = 5,lstm_2_size = 6,opt = 3,lr=-6,decay=0):

        #Train
        params_train = get_params(15,train_directory)
        train_generator = DataGenerator(**params_train)
        train_gen = train_generator.generate()
        x_,y_ = train_gen.__next__()
        x = x_.reshape((-1,sequence_length,pose_data_dims[0]*pose_data_dims[1]))
        y = y_
        model = rnn_model(number_outputs,sequence_length,pose_data_dims,lstm_1_size,lstm_2_size,opt,lr,decay)
        model.train(x,y,epochs)

        #Test
        params_train = get_params(15,validation_directory)
        train_generator = DataGenerator(**params_train)
        train_gen = train_generator.generate()
        x_,y_ = train_gen.__next__()
        x = x_.reshape((-1,sequence_length,pose_data_dims[0]*pose_data_dims[1]))
        y = y_

        val = model.test(x,y)
        if math.isnan(val):
            return -10.0**100
        else:
            return -val.astype(np.int64)



    lr = -7
    decay = 0
    epochs = 500
    opt = 3


    with open("lr_pose.txt","w") as file:
        file.write("lr,decay,val\n")
        for lr_test in [-3,-4,-5,-6,-7,-8]:
            for decay_test in [0,1e-6,1e-5,1e-4]:
                max = -1e-6
                val = get_vals(lr = lr_test,decay = decay_test)
                file.write(str(lr) + "," + str(decay_test)+ "," + str(val) + "\n")
                if val > max:
                    lr = lr_test
                    decay = decay_test


    send_slack_message("Finished Lr and decay optimisation")
    bo = BayesianOptimization(lambda lstm_1_size,lstm_2_size: get_vals(lstm_1_size,lstm_2_size,opt,lr,decay),
                              {"lstm_1_size":(4,8),"lstm_2_size":(4,8)})
    bo.explore({"lstm_1_size":(4,8),"lstm_2_size":(4,8)})
    bo.maximize(init_points=2, n_iter=500, kappa=10,acq="ucb") #, acq="ucb"


    json.dump(bo.res['max'], open("bayes_opt_pose_results.txt",'w'))
    print(bo.res['all'])

    np.save("bayes_opt_all_values",bo.res['all']['values'])

    for i in bo.res['all']['params']:
        json.dump(i, open("bayes_opt_pose_all_results.txt",'a'))

    lstm_1_size = bo.res["max"]['max_params']['lstm_1_size']
    lstm_2_size = bo.res["max"]['max_params']['lstm_2_size']
    send_slack_message("Finished Bayesian Optimisation")


    epochs = 2000

    with open("time_gap_pose_results.txt","w") as file:
        file.write("Time_gap,nn,linear_model\n")
        for time_gap in range(10,100,5):
            print("On time gap " + str(time_gap))
            send_slack_message("Checking pose model time gap : " + str(time_gap))
            #Train
            params_train = get_params(15,train_directory)
            train_generator = DataGenerator(**params_train)
            train_gen = train_generator.generate()
            x_,y_ = train_gen.__next__()
            x = x_.reshape((-1,sequence_length,pose_data_dims[0]*pose_data_dims[1]))
            y = y_
            model = rnn_model(number_outputs,sequence_length,pose_data_dims,lstm_1_size,lstm_2_size,opt,lr,decay)
            model.train(x,y,epochs)
            results = [time_gap]
            x = x_.reshape((-1,sequence_length*pose_data_dims[0]*pose_data_dims[1]))
            y = y_.reshape((-1,3))
            reg = linear_model.LinearRegression()
            reg.fit(x,y)

            #Results
            params_train = get_params(15,validation_directory)
            train_generator = DataGenerator(**params_train)
            train_gen = train_generator.generate()
            x_,y_ = train_gen.__next__()
            x = x_.reshape((-1,sequence_length,pose_data_dims[0]*pose_data_dims[1]))
            y = y_
            results.append(model.test(x,y))
            x = x_.reshape((-1,sequence_length*pose_data_dims[0]*pose_data_dims[1]))
            y = y_.reshape((-1,3))
            results.append(mean_squared_error(y, reg.predict(x)))
            file.write(str(results[0]) + "," + str(results[1]) + "," + str(results[2]) + "\n")

    send_slack_message("finished optimising pose model")
except:
    send_slack_message("Pose model failed")

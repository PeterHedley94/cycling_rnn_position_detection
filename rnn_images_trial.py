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
from src.NN_MODELS.rnn_images_model import *
from src.DATA_PREPARATION.pretend_data import *
from src.DATA_PREPARATION.data_generator import *
from src.common import *
from src.callbacks import *

send_slack_message("Started optimising images model")


try:
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


    def get_vals(conv1_size = 5,conv2_size = 5,no_layers=4,opt = 3,lr=-6,decay = 0):
        batch_size = 16
        #Get parameters
        params_train = get_params(time_gap = 15,batch_size=batch_size,directory=train_directory)
        params_val = get_params(time_gap = 15,batch_size=batch_size,directory=validation_directory)

        #train model
        model = rcnn_model(number_outputs,sequence_length,conv1_size,conv2_size,no_layers,opt,lr,decay)
        model.train(epochs,params_train)

        #test model
        train_generator = DataGenerator(**params_train)
        train_gen = train_generator.generate()

        val = 0
        for i in range(train_generator.batches_per_epoch):
            [x_images_,x_],y_ = train_gen.__next__()
            x = x_images_.reshape((batch_size,sequence_length,IM_HEIGHT,IM_WIDTH,NUMBER_CHANNELS))
            y = y_
            val += model.test([x,x_],y)/train_generator.batches_per_epoch
        if math.isnan(val):
            return -10.0**100
        else:
            return -val #.astype(np.int64)

    lr = -4
    decay = 0.001
    epochs = 250

    with open("lr_images.txt","w") as file:
        file.write("lr,decay,val\n")
        for lr_test in range(-80,-30,5):#[-3,-4,-5,-6,-7,-8]:
            lr_test = lr_test/10
            max = -1e-6
            val = get_vals(lr = lr_test,decay = 0)
            file.write(str(lr_test) + "," + str(0)+ "," + str(val) + "\n")
            if val > max:
                lr = lr_test
                decay = 0
            for decay_test in range(-60,-30,10):#[0,10**-6,10**-5,10**-4]:
                max = -1e-6
                decay_test = 10**(decay_test/10)
                val = get_vals(lr = lr_test,decay = decay_test)
                file.write(str(lr_test) + "," + str(decay_test)+ "," + str(val) + "\n")
                if val > max:
                    lr = lr_test
                    decay = decay_test

    send_slack_message("Finished Lr and decay optimisation images")
    bo1 = BayesianOptimization(lambda conv1_size,conv2_size,no_layers: get_vals(conv1_size,conv2_size,no_layers,opt=3,lr=lr,decay=decay),
                              {"conv1_size":(4,5),"conv2_size":(4,5),"no_layers":(2,4)})
    bo1.explore({"conv1_size":(4,5),"conv2_size":(4,5),"no_layers":(2,4)})
    bo1.maximize(init_points=2, n_iter=60, kappa=10,acq="ucb") #, acq="ucb"


    json.dump(bo.res['max'], open("bayes_orcnn_pt_results.txt",'w'))
    print(bo.res['all'])

    np.save("bayes_rcnn_opt_all_values",bo.res['all']['values'])

    with open("bayes_opt_images_all_results.txt",'w') as file:
        for i in bo.res['all']['params']:
                json.dump(i,file)
                file.write("\n")

    conv1_size = bo.res["max"]['max_params']['conv1_size']
    conv2_size = bo.res["max"]['max_params']['conv2_size']
    no_layers = bo.res["max"]['max_params']['no_layers']


    send_slack_message("Finished Bayesian Optimisation of images")

    epochs = 500

    with open("time_gap_images_results.txt","w") as file:
        file.write("Time_gap,nnl\n")
        for time_gap in range(5,100,5):
            print("On time gap " + str(time_gap))
            send_slack_message("Checking pose model time gap : " + str(time_gap))
            params_train = get_params(time_gap,2)
            model = rcnn_model(number_outputs,sequence_length)
            model.train(epochs,params_train)
            results = [time_gap]


            ##########################################
            params_val = get_params(time_gap = 15,batch_size=batch_size,directory=validation_directory)


            #test model
            train_generator = DataGenerator(**params_train)
            train_gen = train_generator.generate()

            val = 0
            for i in range(train_generator.batches_per_epoch):
                [x_images_,x_],y_ = train_gen.__next__()
                x = x_images_.reshape((batch_size,sequence_length,IM_HEIGHT,IM_WIDTH,NUMBER_CHANNELS))
                y = y_
                val += model.test([x,x_],y)/train_generator.batches_per_epoch

            results.append(val)
            file.write(str(results[0]) + "," + str(results[1]) + "\n")
            '''
            train_generator = DataGenerator(**params_train)
            train_gen = train_generator.generate()
            [x_images_,x_],y_ = train_gen.__next__()
            x = x_images_.reshape((-1,sequence_length,IM_HEIGHT*IM_WIDTH*NUMBER_CHANNELS))
            y = y_.reshape((-1,3))
            reg = linear_model.LinearRegression()
            reg.fit(x,y)
            results.append(mean_squared_error(y, reg.predict(x)))
            file.write(str(results[0]) + "," + str(results[1]) + "," + str(results[2]) + "\n")'''
    send_slack_message("Finished optimising images model")

except:
    send_slack_message("Images model failed")

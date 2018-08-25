#!/usr/bin/env python
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
#parentdir = os.path.dirname(parentdir)
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

import numpy as np,math

from keras.layers import Input, LSTM
from keras.models import Model
import keras
from keras import backend as K
 #set learning phase
K.set_learning_phase(1)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,Adam
from keras.models import load_model
from keras.layers import TimeDistributed
from src.common import *
from src.DATA_PREPARATION.data_generator import *



class rcnn_model:
    def __init__(self,number_outputs,sequence_length,conv1_size = 8,conv2_size = 8,no_layers=4,opt = 3,lr=-3,decay=0):
        conv1_size = int(2**conv1_size)
        conv2_size = int(2**conv2_size)
        lr = 10.0**(lr)
        print("Learning rate is " + str(lr))
        pose_input = Input(shape=(sequence_length,pose_data_dims[0]*pose_data_dims[1]))
        self.model = self.get_cnn_model(conv1_size,conv2_size,no_layers)
        video_input = Input(shape=(sequence_length, IM_HEIGHT, IM_WIDTH, NUMBER_CHANNELS))
        encoded_frame_sequence = TimeDistributed(self.model)(video_input) # the output will be a sequence of vectors
        encoded_video_1 = LSTM(64)(encoded_frame_sequence)
        self.encoded_video = Dense(32)
        encoded_video = self.encoded_video(encoded_video_1)
        output = Dense(number_outputs, activation='linear')(encoded_video)#([encoded_video,self.imumodel])
        self.model = Model(inputs=[video_input,pose_input], outputs=output)

          # the output will be one vector
        if opt <= 1:
            rms = RMSprop(lr=lr, rho=0.9, epsilon=None, decay=decay)
            self.model.compile(loss='mse', optimizer=rms)
        elif opt <= 2:
            sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
            self.model.compile(loss='mse', optimizer=sgd)
        else:
            adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay, amsgrad=False)
            self.model.compile(loss='mse', optimizer=adam)


    def get_cnn_model(self,conv1_size,conv2_size,no_layers):
        cnnmodel = Sequential()
        cnnmodel.add(Conv2D(conv1_size, (3, 3), activation='relu', input_shape=(IM_HEIGHT, IM_WIDTH, NUMBER_CHANNELS)))
        if no_layers > 2:
            cnnmodel.add(Conv2D(conv1_size, (3, 3), activation='relu'))
        cnnmodel.add(MaxPooling2D(pool_size=(2, 2)))
        cnnmodel.add(Dropout(0.25))
        cnnmodel.add(Conv2D(conv2_size, (3, 3), activation='relu'))
        if no_layers > 3:
            cnnmodel.add(Conv2D(conv2_size, (3, 3), activation='relu'))
        cnnmodel.add(MaxPooling2D(pool_size=(2, 2)))
        cnnmodel.add(Dropout(0.25))
        cnnmodel.add(Flatten())
        return cnnmodel

    def train(self,epochs,params_train):
        train_generator = DataGenerator(**params_train)
        train_gen = train_generator.generate()
        #CHECKS!################
        #test_in = train_gen.__next__()
        steps_per_epoch_ =  train_generator.batches_per_epoch
        ##########################
        self.model.fit_generator(train_gen,steps_per_epoch = steps_per_epoch_, epochs=epochs,verbose=0)

    def test(self,x,y):
        return self.model.evaluate(x,y)

    def test_eval_function(self,x,y):
        mse = 0
        for i in range(x.shape[0]):
            y_pred = self.model.predict(x[None,i,:])
            for pred in range(number_outputs):
                print(y[i,pred])
                print(y_pred)
                print(y_pred[0][pred])
                mse += (y[i,pred]-y_pred[0][pred])**2
        mse = mse/x.shape[0]/number_outputs
        print("Model evaluation is " + str(self.model.evaluate(x,y)))
        return mse

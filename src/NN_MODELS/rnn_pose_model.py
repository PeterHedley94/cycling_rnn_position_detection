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



class rnn_model:
    def __init__(self,number_outputs,sequence_length,data_dims,lstm_1_size = 5,lstm2_size = 5,opt = 0.9,lr=6):
        lstm_1_size = int(2**lstm_1_size)
        lstm2_size = int(2**lstm2_size)
        lr = 10.0**(lr)
        print("Learning rate is " + str(lr))
        self.model = Sequential()
        self.model.add(LSTM(lstm_1_size,return_sequences=True,activation = "linear",input_shape=(sequence_length,data_dims[0]*data_dims[1])))  # returns a sequence of vectors of dimension 32
        self.model.add(LSTM(lstm2_size,activation = "linear"))
        self.model.add(Dense(number_outputs,activation = "linear"))
        if opt <= 1:
            rms = RMSprop(lr=lr, rho=0.9, epsilon=None, decay=0.0)
            self.model.compile(loss='mse', optimizer=rms)
        elif opt <= 2:
            sgd = SGD(lr=lr, decay=5e-5, momentum=0.9, nesterov=True)
            self.model.compile(loss='mse', optimizer=sgd)
        else:
            adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            self.model.compile(loss='mse', optimizer=adam)

    def train(self,x,y,epochs):
        self.model.fit(x,y,batch_size=4, epochs=epochs, shuffle=False,verbose = 0)

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

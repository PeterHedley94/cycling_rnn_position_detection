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
from src.NN_MODELS.common_network_operations import *
from src.callbacks import *



class rcnn_total_model:
    def __init__(self,number_outputs,sequence_length,conv1_size = 8,conv2_size = 8,no_layers=4,opt = 3,lr=-3,decay=0,
                pose_cached_model=None,image_cached_model = None,complete_cached_model = None):
        conv1_size = int(2**conv1_size)
        conv2_size = int(2**conv2_size)
        lr = 10.0**(lr)
        self.lr = lr
        self.decay = decay
        print("Learning rate is " + str(lr))
        self.pose_cached = False
        self.image_cached = False
        self.complete_cached = False


        #Inputs
        pose_input = Input(shape=(sequence_length,pose_data_dims[0]*pose_data_dims[1]))
        video_input = Input(shape=(sequence_length, IM_HEIGHT, IM_WIDTH, NUMBER_CHANNELS))


        #IMAGES RCNN MODE
        self.cnnmodel = self.get_cnn_model(conv1_size,conv2_size,no_layers)
        encoded_frame_sequence = TimeDistributed(self.cnnmodel)(video_input) # the output will be a sequence of vectors
        encoded_video_1 = LSTM(64)(encoded_frame_sequence)
        self.encoded_video = Dense(32)
        encoded_video = self.encoded_video(encoded_video_1)
        output = Dense(number_outputs, activation='linear')(encoded_video)#([encoded_video,self.imumodel])
        self.initial_cnn_model = Model(inputs=[video_input,pose_input], outputs=output)


        #POSE SEQUENCE MODEL
        self.posemodel = self.get_pose_model()
        pose_model_branch = self.posemodel(pose_input)
        pose_model_output = Dense(number_outputs, activation='linear')(pose_model_branch)
        #self.initial_pose_model = Model(inputs=[video_input,pose_input], outputs=pose_model_output)
        self.initial_pose_model = Model(inputs=[pose_input], outputs=pose_model_output)

        #Total MODEL
        output_branches = keras.layers.Concatenate(axis=-1)([encoded_video,pose_model_branch])
        final_dense = Dense(64, activation='linear')(output_branches)
        total_output= Dense(number_outputs, activation='linear')(final_dense)
        self.total_model = Model(inputs=[video_input,pose_input], outputs=total_output)


        if pose_cached_model is not None:
            self.initial_pose_model = load_model(pose_cached_model)
            self.pose_cached = True
        if image_cached_model is not None:
            self.initial_cnn_model = load_model(image_cached_model)
            self.image_cached = True
        if complete_cached_model is not None:
            self.total_model = load_model(complete_cached_model)
            self.complete_cached = True

        sgd = SGD(lr=10**(-3.5), decay=10**(-6), momentum=0.9, nesterov=True)
        self.initial_cnn_model.compile(loss='mse', optimizer=sgd)
        sgd = SGD(lr=10**(-3), decay=10**(-5), momentum=0.9, nesterov=True)
        self.initial_pose_model.compile(loss='mse', optimizer=sgd)



    def get_cnn_model(self,conv1_size,conv2_size,no_layers):
        cnnmodel = Sequential()
        cnnmodel.add(Conv2D(20, (3, 3), activation='relu', input_shape=(IM_HEIGHT, IM_WIDTH, NUMBER_CHANNELS)))
        cnnmodel.add(MaxPooling2D(pool_size=(2, 2)))
        cnnmodel.add(Dropout(0.25))
        cnnmodel.add(Conv2D(32, (3, 3), activation='relu'))
        cnnmodel.add(MaxPooling2D(pool_size=(2, 2)))
        cnnmodel.add(Dropout(0.25))
        cnnmodel.add(Flatten())
        return cnnmodel

    def get_pose_model(self):
        # expected input data shape: (batch_size, timesteps, data_dim)
        model = Sequential()
        model.add(LSTM(63, return_sequences=True,input_shape=(sequence_length,pose_data_dims[0]*pose_data_dims[1])))  # returns a sequence of vectors of dimension 32
        model.add(LSTM(53, return_sequences=True))  # returns a sequence of vectors of dimension 32
        model.add(LSTM(32))  # return a single vector of dimension 32
        return model

    #Note images model takes a long time to load photos - so for training pose it is faster not to
    def train(self,epochs,complete_params_train,complete_params_val,pose_params_train,pose_params_val):
        train_generator = DataGenerator(**complete_params_train)
        train_gen = train_generator.generate()
        validate_generator = DataGenerator(**complete_params_val)
        validate_gen = validate_generator.generate()
        #CHECKS!################
        steps_per_epoch_ =  train_generator.batches_per_epoch
        validation_steps_per_epoch_ =  validate_generator.batches_per_epoch
        calls_ = logs()
        ################################
        #IMAGES Model

        if not self.image_cached:
            self.initial_cnn_model.fit_generator(train_gen,validation_data=validate_gen,
                                                callbacks=[calls_.json_logging_callback,calls_.slack_callback,
                                                get_model_checkpoint(),get_Tensorboard()],validation_steps=validation_steps_per_epoch_,
                                                steps_per_epoch = steps_per_epoch_, epochs=epochs,verbose=1)

            self.model_name = "Images_model"
            current_directory = os.path.dirname(os.path.abspath(__file__))
            print("Model saved to " + os.path.join(current_directory, os.path.pardir,MODEL_SAVE_FOLDER,self.model_name + '.hdf5'))
            if not os.path.exists(MODEL_SAVE_FOLDER):
                os.makedirs(MODEL_SAVE_FOLDER)
            self.initial_cnn_model.save(os.path.join(MODEL_SAVE_FOLDER,str(self.model_name + '.hdf5')))
            clean_up(self.model_name)


        ########################################
        #POSE MODEL

        #######################################
        train_generator = DataGenerator(**pose_params_train)
        train_gen = train_generator.generate()
        validate_generator = DataGenerator(**pose_params_val)
        validate_gen = validate_generator.generate()
        #CHECKS!################
        steps_per_epoch_ =  train_generator.batches_per_epoch
        validation_steps_per_epoch_ =  validate_generator.batches_per_epoch
        ###################################################################

        if not self.pose_cached:
            self.initial_pose_model.fit_generator(train_gen,validation_data=validate_gen,
                                                callbacks=[calls_.json_logging_callback,calls_.slack_callback,
                                                get_model_checkpoint(),get_Tensorboard()],validation_steps=validation_steps_per_epoch_,
                                                steps_per_epoch = steps_per_epoch_, epochs=5000,verbose=1)

            self.model_name = "Pose_model"
            current_directory = os.path.dirname(os.path.abspath(__file__))
            print("Model saved to " + os.path.join(current_directory, os.path.pardir,MODEL_SAVE_FOLDER,self.model_name + '.hdf5'))
            if not os.path.exists(MODEL_SAVE_FOLDER):
                os.makedirs(MODEL_SAVE_FOLDER)
            self.initial_pose_model.save(os.path.join(MODEL_SAVE_FOLDER,str(self.model_name + '.hdf5')))
            clean_up(self.model_name)

        ##########################
        #COMPLETE MODEL FINAL LAYERS

        ###############################################
        train_generator = DataGenerator(**complete_params_train)
        train_gen = train_generator.generate()
        validate_generator = DataGenerator(**complete_params_val)
        validate_gen = validate_generator.generate()
        #CHECKS!################
        steps_per_epoch_ =  train_generator.batches_per_epoch
        validation_steps_per_epoch_ =  validate_generator.batches_per_epoch
        ######################################################################

        if not self.complete_cached:
            # train only the top layers (which were randomly initialized)
            for layer in self.posemodel.layers:
                layer.trainable = False
            for layer in self.initial_cnn_model.layers:
                layer.trainable = False

            #adam = Adam(lr=10**(-6), beta_1=0.9, beta_2=0.999, epsilon=None, decay=10**(-7), amsgrad=False)
            adam = Adam(lr=10**(-4), beta_1=0.9, beta_2=0.999, epsilon=None, decay=10**(-5), amsgrad=False)
            #sgd = SGD(lr=10*(-100000), decay=1e-05, momentum=0.9, nesterov=True)
            self.total_model.compile(loss='mse', optimizer=adam)
            self.total_model.fit_generator(train_gen,validation_data=validate_gen,
                                                callbacks=[calls_.json_logging_callback,calls_.slack_callback,
                                                get_model_checkpoint(),get_Tensorboard()],validation_steps=validation_steps_per_epoch_,
                                                steps_per_epoch = steps_per_epoch_, epochs=epochs,verbose=1)


            self.model_name = "Full_model_pretuning"
            current_directory = os.path.dirname(os.path.abspath(__file__))
            print("Model saved to " + os.path.join(current_directory, os.path.pardir,MODEL_SAVE_FOLDER,self.model_name + '.hdf5'))
            if not os.path.exists(MODEL_SAVE_FOLDER):
                os.makedirs(MODEL_SAVE_FOLDER)
            self.total_model.save(os.path.join(MODEL_SAVE_FOLDER,str(self.model_name + '.hdf5')))
            clean_up(self.model_name)

        ###############################
        #COMPLETE MODEL FINE TUNE
        # unfreeze layers and fine-tune
        for layer in self.posemodel.layers:
            layer.trainable = True
        for layer in self.initial_cnn_model.layers:
            layer.trainable = True

        adam = Adam(lr=10**(-6), beta_1=0.9, beta_2=0.999, epsilon=None, decay=10**(-9), amsgrad=False)
        self.total_model.compile(loss='mse', optimizer=adam)
        self.total_model.fit_generator(train_gen,validation_data=validate_gen,
                                            callbacks=[calls_.json_logging_callback,calls_.slack_callback,
                                            get_model_checkpoint(),get_Tensorboard()],validation_steps=validation_steps_per_epoch_,
                                            steps_per_epoch = steps_per_epoch_, epochs=epochs,verbose=1)


        self.model_name = "Final_model"
        current_directory = os.path.dirname(os.path.abspath(__file__))
        print("Model saved to " + os.path.join(current_directory, os.path.pardir,MODEL_SAVE_FOLDER,self.model_name + '.hdf5'))
        if not os.path.exists(MODEL_SAVE_FOLDER):
            os.makedirs(MODEL_SAVE_FOLDER)
        self.total_model.save(os.path.join(MODEL_SAVE_FOLDER,str(self.model_name + '.hdf5')))
        clean_up(self.model_name)



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

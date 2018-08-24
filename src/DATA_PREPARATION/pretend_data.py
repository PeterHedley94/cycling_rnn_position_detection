#!/usr/bin/env python
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
#parentdir = os.path.dirname(parentdir)
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
import numpy as np,math


def create_circle_roll_test_data(sequence_length,no_points):
    global COUNT
    radius = 52
    angle = 0
    count = 0
    x_data = []
    y_data = []
    for i in range(no_points):
        x_data_intermediate = []
        for no_in_seq in range(sequence_length+1):
            if i+no_in_seq > 0:
                angle_0 = i*(360+90)/1000
                angle_1 = i+no_in_seq*(360+90)/1000
            else:
                angle_1,angle_0 = 0,0
            rad = math.radians(angle_1)
            x = radius*(math.sin(math.radians(angle_1))-math.sin(math.radians(angle_0)))
            z = radius*(math.cos(math.radians(angle_1))-math.cos(math.radians(angle_0)))
            d = np.array([[1,0,0,x],[0,math.cos(rad),-math.sin(rad),0],[0,math.sin(rad),math.cos(rad),z]])
            if no_in_seq == sequence_length:
                x_data.append(np.array(x_data_intermediate))
                y_data.append([x,0,z])
            else:
                x_data_intermediate.append(d)
    return np.array(x_data),np.array(y_data)

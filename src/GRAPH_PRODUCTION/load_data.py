import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
#parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
import numpy as np
import json

def get_bayes_json(filename):
    data = []
    with open(filename,'r') as file:
        lines = file.read().splitlines()
        for line in lines:
            data.append(json.loads(line))
    return data

def get_numpy_data(filename):
    return np.load(filename)


def get_lr_decay_data(filename):
    data = np.loadtxt(filename,delimiter = ",",skiprows=1)
    return data

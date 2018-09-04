import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')

import matplotlib.pyplot as plt
import json

def plot_training_curve(x_data,train_loss,val_loss,x_axis_title,y_axis_title,title,graph_name,o1=0,o2=0):
    fig, ax=plt.subplots(ncols=1)
    ax.set_xlabel(x_axis_title)
    ax.set_ylabel(y_axis_title)
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot(x_data,val_loss,color = 'orange')
    ax.text(x_data[-1],val_loss[-1]+o2,"Validation Loss",color = "orange",verticalalignment='bottom', horizontalalignment='right', fontsize=10)
    ax.plot(x_data,train_loss, color = "blue")
    ax.text(x_data[-1],train_loss[-1]+o2,"Training Loss",color = "blue",verticalalignment='bottom', horizontalalignment='right', fontsize=10)
    fig.savefig(graph_name, bbox_inches='tight', format='eps', dpi=500)

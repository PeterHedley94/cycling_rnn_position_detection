import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
#from src.GRAPH_PRODUCTION.load_data import *
#from src.GRAPH_PRODUCTION.heat_map import *
#from src.common import *
import matplotlib.pyplot as plt


###############################################
#Pose Model vs linear model , time gap
filename = "time_gap_pose_results.txt"
def make_barplot(title,x_data,y_data,x_axis_title,y_axis_title,graph_name):
    fig, ax=plt.subplots(ncols=1)
    bar = ax.bar(x_data, y_data,align='center')
    for rect in bar.patches:
        x0,x1 = rect.get_bbox().get_points()[:,0]
        y0,y1 = rect.get_bbox().get_points()[:,1]
        ax.text((x0 + (x1-x0)/2),y1-0.3,str(y1), style='oblique',fontsize = 8, color = 'white',verticalalignment='bottom', horizontalalignment='center')
    ax.set_xlabel(x_axis_title)
    ax.set_ylabel(y_axis_title)
    ax.set_title(title)
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    fig.savefig(graph_name, bbox_inches='tight', format='eps', dpi=500)

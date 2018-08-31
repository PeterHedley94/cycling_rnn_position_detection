import os, sys, inspect
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')

import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import os
import numpy as np
import pandas as pd


def generate_heat_map(title,x_data,y_data,z_data,x_axis_title,y_axis_title,z_axis_title,graph_name,bounds):
    df = pd.DataFrame({y_axis_title: y_data, x_axis_title: x_data, z_axis_title: z_data})
    fig, ax=plt.subplots(ncols=1)
    #dz = (max(z_data) - min(z_data))*0.1
    dz = (max(bounds) - min(bounds))*0.1
    #levels = np.arange(min(z_data) - dz, max(z_data) + dz, 0.01)
    levels = bounds #np.arange(min(bounds) - dz, max(bounds) + dz, 0.01)
    norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    contour = ax.tricontourf(df[x_axis_title], df[y_axis_title], df[z_axis_title],levels=levels,norm=norm)
    cbar = plt.colorbar(contour)
    ax.set_xlabel(x_axis_title)
    ax.set_ylabel(y_axis_title)
    ax.set_title(title)
    cbar.set_label(z_axis_title)
    fig.savefig(graph_name, bbox_inches='tight')

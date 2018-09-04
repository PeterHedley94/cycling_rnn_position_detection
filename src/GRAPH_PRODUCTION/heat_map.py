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
    fig.savefig(graph_name, bbox_inches='tight', format='eps', dpi=500)


def generate_3D_heat_map(title,data,z_data,dims_names,z_axis_title,graph_name):
    fig, ax=plt.subplots(data.shape[1],data.shape[1])
    fig.suptitle(title, fontsize=16)
    start = True
    for dim1 in range(data.shape[1]):
        for dim2 in range(data.shape[1]):

            df = pd.DataFrame({dims_names[dim1]: data[:,dim1], dims_names[dim2]: data[:,dim2], z_axis_title: z_data})
            dz = (max(z_data) - min(z_data))*0.1
            levels = np.arange(min(z_data) - dz, max(z_data) + dz, 0.01)
            if dim1 != dim2:
                contour = ax[dim1,dim2].tricontourf(df[dims_names[dim1]], df[dims_names[dim2]], df[z_axis_title],levels=levels)#,norm=norm)
                #cbar = plt.colorbar(contour)
                #cbar.set_label(z_axis_title)
            if dim1 == data.shape[1]-1:
                ax[dim1,dim2].set_xlabel(dims_names[dim2])
                ax[dim1,dim2].set_xlim(min(df[dims_names[dim1]]),max(df[dims_names[dim1]]))
            else:
                ax[dim1,dim2].get_xaxis().set_visible(False)
            if dim2 == 0:
                ax[dim1,dim2].set_ylabel(dims_names[dim1])
                ax[dim1,dim2].set_ylim(min(df[dims_names[dim1]]),max(df[dims_names[dim1]]))
            elif dim2 == data.shape[1]-1:
                ax[dim1,dim2].get_yaxis().set_visible(False)
            else:
                ax[dim1,dim2].get_yaxis().set_visible(False)
            #title = dims_names[dim1] + " vs " +  dims_names[dim2] + " vs " + z_axis_title
            #ax[dim1,dim2].set_title(title)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(contour, cax=cbar_ax,shrink=0.6)
    cbar.set_label(z_axis_title)
    cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(), fontsize=10)
    fig.savefig(graph_name, bbox_inches='tight', format='eps', dpi=500)

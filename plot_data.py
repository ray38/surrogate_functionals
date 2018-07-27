import matplotlib
matplotlib.use('Agg') 
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt

import numpy as np
import csv
import sys
import os
import time
import math
import json
from glob import glob

import scipy

import itertools
import multiprocessing

try: import cPickle as pickle
except: import pickle
import matplotlib.pyplot as plt
from subsampling import subsampling_system,random_subsampling,subsampling_system_with_PCA


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def sae(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true))

def map_to_n1_1(arr, maxx, minn):
    return np.divide(np.subtract(arr,minn),(maxx-minn)/2.)-1.
    
def log(log_filename, text):
    with open(log_filename, "a") as myfile:
        myfile.write(text)
    return

def write(log_filename, text):
    with open(log_filename, "w") as myfile:
        myfile.write(text)
    return



def map_to_0_1(arr, maxx, minn):
    return np.divide(np.subtract(arr,minn),(maxx-minn))
    
def map_back(arr, maxx, minn):
    return np.add(np.multiply(arr,(maxx-minn)),minn)

def get_x0():
    x = [ -0.45816529328314287, 0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294]
    return x

def optimization_constants(x):
    #C0I = x[0]
    #C1  = x[1]
    #CC1 = x[2]
    #CC2 = x[3]
    #IF2 = x[4]

    C1  = x[0]
    gamma = x[1]
    alpha1 = x[2]
    beta1 = x[3]
    beta2 = x[4]
    beta3 = x[5]
    beta4 = x[6]

    #return C0I, C1, CC1, CC2, IF2, gamma, alpha1, beta1, beta2, beta3, beta4
    return C1, gamma, alpha1, beta1, beta2, beta3, beta4

def G(rtrs, gamma, alpha1, beta1, beta2, beta3, beta4):
    Q0 = -2.0 * gamma * (1.0 + alpha1 * rtrs * rtrs)
    Q1 = 2.0 * gamma * rtrs * (beta1 +
                           rtrs * (beta2 +
                                   rtrs * (beta3 +
                                           rtrs * beta4)))
    G1 = Q0 * np.log(1.0 + 1.0 / Q1)
    return G1

def lda_x( n, x):
#    C0I, C1, CC1, CC2, IF2 = lda_constants()
    C1, gamma, alpha1, beta1, beta2, beta3, beta4 = optimization_constants(x)

    C0I = 0.238732414637843
    #C1 = -0.45816529328314287
    rs = (C0I / n) ** (1 / 3.)
    ex = C1 / rs
    return n*ex
    #e[:] += n * ex

def lda_c( n, x):
    #C0I, C1, CC1, CC2, IF2 = lda_constants()
    C1, gamma, alpha1, beta1, beta2, beta3, beta4 = optimization_constants(x)

    C0I = 0.238732414637843
    #C1 = -0.45816529328314287
    rs = (C0I / n) ** (1 / 3.)
    ec = G(rs ** 0.5, gamma, alpha1, beta1, beta2, beta3, beta4)
    return n*ec
    #e[:] += n * ec

def predict_LDA(n,LDA_x):

    n = np.asarray(n)

    return lda_x(n,LDA_x) + lda_c(n,LDA_x)

def predict_LDA_residual(n,LDA_x,X,NN_model):

    n = np.asarray(n)

    return lda_x(n,LDA_x) + lda_c(n,LDA_x) + ((NN_model.predict(X*1e6))/1e6)

def predict(n,LDA_x,X,NN_model,y):

    dens = n
    predict_y = predict_LDA_residual(n,LDA_x,X,NN_model)

    #LDA_predict_y = predict_LDA(n,LDA_x)

    error = y - predict_y


    return predict_y, error


  
def plot_result(x,y,x_name,y_name, filename,figure_size,x_scale = "linear",y_scale="linear"):
    result = {}
    print x.shape
    print y.shape
    result[x_name] = x
    result[y_name] = y


    plot_data = pd.DataFrame(data=result)

    sns.set_context("poster")
    grid = sns.lmplot( x=x_name, y=y_name, data=plot_data, fit_reg=False, legend=False,size=figure_size,aspect=2.0)
    plt.xlabel(x_name,fontsize=30)
    plt.ylabel(y_name,fontsize=30)
    grid.set(xscale=x_scale, yscale=y_scale)
    plt.tick_params(labelsize=20)
    
     
    # Move the legend to an empty part of the plot
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


    return
    

if __name__ == "__main__":

    data_filename = sys.argv[1]
    LDA_filename = sys.argv[2]

    print "start"
    
    with open(data_filename, 'rb') as handle:
        x,y,dens = pickle.load(handle)

    LDA_model = pickle.load(open(LDA_filename, 'rb'))


    predict_y = predict_LDA(dens,LDA_model.x)
    residual = y - predict_y

    x_name = "Density"
    y_name = "Energy Density (eV/$A^3$)"

    plot_result(dens,y,x_name,y_name, "data_plot_epxc_dens_{}_{}_{}.png".format(10,"real","real"),10,x_scale = "linear",y_scale="linear")
    plot_result(dens,y,x_name,y_name, "data_plot_epxc_dens_{}_{}_{}.png".format(20,"real","real"),20,x_scale = "linear",y_scale="linear")
    plot_result(dens,y,x_name,y_name, "data_plot_epxc_dens_{}_{}_{}.png".format(10,"log","real"),10,x_scale = "log",y_scale="linear")
    plot_result(dens,y,x_name,y_name, "data_plot_epxc_dens_{}_{}_{}.png".format(20,"log","real"),20,x_scale = "log",y_scale="linear")
    plot_result(dens,y,x_name,y_name, "data_plot_epxc_dens_{}_{}_{}.png".format(10,"log","symlog"),10,x_scale = "log",y_scale="symlog")
    plot_result(dens,y,x_name,y_name, "data_plot_epxc_dens_{}_{}_{}.png".format(20,"log","symlog"),20,x_scale = "log",y_scale="symlog")
    y_name = "E. Dens. Residual (eV/$A^3$)"
    plot_result(dens,residual,x_name,y_name, "data_plot_residual_dens_{}_{}_{}.png".format(10,"log","real"),10,x_scale = "log",y_scale="linear")
    plot_result(dens,residual,x_name,y_name, "data_plot_residual_dens_{}_{}_{}.png".format(10,"log","real"),20,x_scale = "log",y_scale="linear")
    plot_result(dens,residual,x_name,y_name, "data_plot_residual_dens_{}_{}_{}.png".format(10,"log","symlog"),10,x_scale = "log",y_scale="symlog")
    plot_result(dens,residual,x_name,y_name, "data_plot_residual_dens_{}_{}_{}.png".format(10,"log","symlog"),20,x_scale = "log",y_scale="symlog")


    result = {}
    print data.shape
    result["Density"] = dens
    result["log(Density)"] = np.log10(dens)


    plot_data = pd.DataFrame(data=result)
    plt.figure(figsize=(20,10))
    sns.set(style="white", palette="pastel", color_codes=True)
    ax = sns.distplot(data["log(Density)"],bins=100,kde=True,hist_kws={ "linewidth": 0,"alpha": 1},kde_kws={"color": "k", "lw": 0})
    ax.set_xlim(-9.,3.)
    plt.tight_layout()
    plt.savefig("dens_dist_log.png")
    plt.close()
    


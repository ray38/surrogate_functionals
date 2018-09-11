# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:43:05 2017

@author: ray
"""
import matplotlib
matplotlib.use('Agg') 
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt

import numpy as np
import sys
import os
import time
import math
import json
from glob import glob
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3

import itertools
import multiprocessing

try: import cPickle as pickle
except: import pickle
import matplotlib.pyplot as plt
from subsampling import subsampling_system,random_subsampling,subsampling_system_with_PCA

def map_to_n1_1(arr, maxx, minn):
    return np.divide(np.subtract(arr,minn),(maxx-minn)/2.)-1.
    
def log(log_filename, text):
    with open(log_filename, "a") as myfile:
        myfile.write(text)
    return

def map_to_0_1(arr, maxx, minn):
    return np.divide(np.subtract(arr,minn),(maxx-minn))
    
def map_back(arr, maxx, minn):
    return np.add(np.multiply(arr,(maxx-minn)),minn)




def save_resulting_figure(n,poly_model,y):

    dens = n
#    predict_y = predict_LDA_residual(n,LDA_x,X,NN_model)

    predict_y = poly_model.predict(n)


    error = y - predict_y

    log_dens = np.log10(n)

    log_predict_y = np.log10(np.multiply(-1.,predict_y))
    log_y = np.log10(np.multiply(-1.,y))

    log_a_error = np.log10(np.abs(error))



    fig, axes = plt.subplots(2, 2, figsize=(100,100))
    ((ax1, ax2),(ax3,ax4)) = axes
    ax1.scatter(dens, y,            c= 'red',  lw = 0,label='original',alpha=1.0)
    ax1.scatter(dens, predict_y,    c= 'blue',  lw = 0,label='predict',alpha=1.0)
    ax1.scatter(dens, error,            c= 'yellow',  lw = 0,label='error',alpha=1.0)
    legend = ax1.legend(loc="best", shadow=False, scatterpoints=1, fontsize=80, markerscale=10)
    ax1.tick_params(labelsize=80)
    ax1.set_xlabel('density', fontsize=100)
    ax1.set_ylabel('energy density', fontsize=100)

    ax2.scatter(log_dens, y,            c= 'red',  lw = 0,label='original',alpha=1.0)
    ax2.scatter(log_dens, predict_y,    c= 'blue',  lw = 0,label='predict',alpha=1.0)
    ax2.scatter(log_dens, error,            c= 'yellow',  lw = 0,label='error',alpha=1.0)
    legend = ax2.legend(loc="best", shadow=False, scatterpoints=1, fontsize=80, markerscale=10)
    ax2.tick_params(labelsize=80)
    ax2.set_xlabel('log10 density', fontsize=100)
    ax2.set_ylabel('energy density', fontsize=100)

    ax3.scatter(dens, log_y,            c= 'red',  lw = 0,label='original',alpha=1.0)
    try:
        ax3.scatter(dens, log_predict_y,    c= 'blue',  lw = 0,label='predict',alpha=1.0)
    except:
        pass
    legend = ax3.legend(loc="best", shadow=False, scatterpoints=1, fontsize=80, markerscale=10)
    ax3.tick_params(labelsize=80)
    ax3.set_xlabel('density', fontsize=100)
    ax3.set_ylabel('log10 negative energy density', fontsize=100)

    ax4.scatter(log_dens, log_y,            c= 'red',  lw = 0,label='original',alpha=1.0)
    try:
        ax4.scatter(log_dens, log_predict_y,    c= 'blue',  lw = 0,label='predict',alpha=1.0)
    except:
        pass
    legend = ax4.legend(loc="best", shadow=False, scatterpoints=1, fontsize=80, markerscale=10)
    ax4.tick_params(labelsize=80)
    ax4.set_xlabel('log10 density', fontsize=100)
    ax4.set_ylabel('log10 negative energy density', fontsize=100)

    plt.savefig('result_plot.png')


    fig, axes = plt.subplots(2, 2, figsize=(100,100))
    ((ax1, ax2),(ax3,ax4)) = axes
    ax1.scatter(dens, error,            c= 'red',  lw = 0,label='error',alpha=1.0)
    legend = ax1.legend(loc="best", shadow=False, scatterpoints=1, fontsize=80, markerscale=10)
    ax1.tick_params(labelsize=80)
    ax1.set_xlabel('density', fontsize=100)
    ax1.set_ylabel('error', fontsize=100)

    ax2.scatter(log_dens, error,            c= 'red',  lw = 0,label='error',alpha=1.0)
    legend = ax2.legend(loc="best", shadow=False, scatterpoints=1, fontsize=80, markerscale=10)
    ax2.tick_params(labelsize=80)
    ax2.set_xlabel('log10 density', fontsize=100)
    ax2.set_ylabel('error', fontsize=100)

    ax3.scatter(dens, log_a_error,            c= 'red',  lw = 0,label='absolute error',alpha=1.0)
    legend = ax3.legend(loc="best", shadow=False, scatterpoints=1, fontsize=80, markerscale=10)
    ax3.tick_params(labelsize=80)
    ax3.set_xlabel('density', fontsize=100)
    ax3.set_ylabel('log10 absolute error', fontsize=100)

    ax4.scatter(log_dens, log_a_error,            c= 'red',  lw = 0,label='absolute error',alpha=1.0)
    legend = ax4.legend(loc="best", shadow=False, scatterpoints=1, fontsize=80, markerscale=10)
    ax4.tick_params(labelsize=80)
    ax4.set_xlabel('log10 density', fontsize=100)
    ax4.set_ylabel('log10 absolute error', fontsize=100)

    plt.savefig('error_plot.png')



    return

def fit_with_Poly(X,y,degree):

    filename = "Poly_{}_model.sav".format(degree)

    try:
        poly_model = pickle.load(open(filename, 'rb'))
        print 'model loaded'
    except:
        poly_model = make_pipeline(PolynomialFeatures(degree), Ridge())
        poly_model.fit(X, y)

        pickle.dump(poly_model, open(filename, 'wb'))
    
    # The coefficients
    #print 'Coefficients: \n', li_model.coef_
    # The mean squared error
    print "Mean squared error: %.20f" % np.mean((poly_model.predict(X) - y) ** 2)
    # Explained variance score: 1 is perfect prediction
    print 'Variance score: %.20f' % poly_model.score(X, y)
    
    residual = y-poly_model.predict(X)
#    plt.scatter(X, residual,  color='black')
#    plt.show()
    return residual, poly_model




def read_data_from_one_dir(directory):
    temp_cwd = os.getcwd()
    os.chdir(directory)

    print directory

    subsampled_filename = "overall_subsampled_data.p"
    random_filename = "overall_random_data.p"

    try:
        molecule_subsampled_data = pickle.load(open(subsampled_filename,'rb'))
        print "read subsampled data"
    except:
        molecule_subsampled_data = []

    try:
        molecule_random_data = pickle.load(open(random_filename,'rb'))
        print "read random data"
    except:
        molecule_random_data = []

    os.chdir(temp_cwd)

    return molecule_subsampled_data, molecule_random_data



def get_training_data(dataset_name,setup):

    data_dir_name = setup["working_dir"] + "/data/*/" 
    data_paths = glob(data_dir_name)
    print data_paths


    overall_subsampled_data = []
    overall_random_data = []
    num_samples = len(data_paths)
    num_random_per_molecule = int(math.ceil(float(setup["random_pick"])/float(num_samples)))
    for directory in data_paths:
        temp_molecule_subsampled_data, temp_molecule_random_data = read_data_from_one_dir(directory)
        overall_subsampled_data += temp_molecule_subsampled_data
#        overall_random_data += temp_molecule_random_data
#        overall_random_data += random_subsampling(temp_molecule_random_data, num_random_per_molecule)


#    overall_random_data = random_subsampling(overall_random_data, setup["random_pick"])

    list_subsample = setup["subsample_feature_list"]
    temp_list_subsample = setup["subsample_feature_list"]
    if temp_list_subsample == []:
        for m in range(len(overall_subsampled_data[0])):
            temp_list_subsample.append(m)

    if len(temp_list_subsample) <= 10:
        overall_subsampled_data = subsampling_system(overall_subsampled_data, list_desc = list_subsample, cutoff_sig = float(setup["subsample_cutoff_sig"]), rate = float(setup["subsample_rate"]))
    else:
        overall_subsampled_data = subsampling_system_with_PCA(overall_subsampled_data, list_desc = list_subsample, cutoff_sig = float(setup["subsample_cutoff_sig"]), rate = float(setup["subsample_rate"]),start_trial_component = 9)


#    overall = overall_random_data + overall_subsampled_data
    overall = overall_subsampled_data


    X_train = []
    y_train = []
    dens = []

    for entry in overall:
#        if entry[0] >= lower and entry[0] <= upper:
        X_train.append(list(entry[1:]))
        dens.append(entry[1])
        y_train.append(entry[0])
    
    
    X_train = (np.asarray(X_train))
    y_train = np.asarray(y_train).reshape((len(y_train),1))
    dens = np.asarray(dens).reshape((len(dens),1))
    
    return X_train, y_train, dens



if __name__ == "__main__":


    setup_filename = sys.argv[1]
    dataset_name = sys.argv[2]
    degree = int(sys.argv[3])

    with open(setup_filename) as f:
        setup = json.load(f)


    h = float(setup['grid_spacing'])
    L = float(setup['box_dimension'])
    N = int(setup['number_segment_per_side'])
    dir_name = "{}_{}_{}".format(str(L).replace('.','-'),str(h).replace('.','-'),N)

    working_dir = os.getcwd() + '/' + dir_name + '/' + dataset_name

    setup["working_dir"] = working_dir

    model_save_dir = working_dir + "/" + "NN_poly_{}".format(degree)
   
    setup["model_save_dir"] = model_save_dir

    
    
    X_train,y, dens = get_training_data(dataset_name,setup)
   
    if os.path.isdir(model_save_dir) == False:
        os.makedirs(model_save_dir)

    os.chdir(model_save_dir)
    
    residual,poly_model = fit_with_Poly(dens,y,degree)
    save_resulting_figure(dens,poly_model,y)
    #model = fit_with_KerasNN(X_train,residual, tol, slowdown_factor, early_stop_trials)


    
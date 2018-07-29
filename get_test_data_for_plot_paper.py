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
import csv
import sys
import os
import time
import math
import json
from glob import glob
from sklearn import linear_model

import scipy

import itertools
import multiprocessing

try: import cPickle as pickle
except: import pickle
import matplotlib.pyplot as plt
from subsampling import subsampling_system,random_subsampling,subsampling_system_with_PCA

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
    #molecule_random_data = []

    os.chdir(temp_cwd)

    return molecule_subsampled_data, molecule_random_data



def get_training_data(dataset_name,setup):

    data_dir_name = setup["working_dir"] + "/data/*/" 
    data_paths = glob(data_dir_name)
    print data_paths


    overall_subsampled_data = []
    overall_random_data = []
    molecule_list = []
    num_samples = len(data_paths)
    num_random_per_molecule = int(math.ceil(float(setup["random_pick"])/float(num_samples)))
    for directory in data_paths:
        temp_molecule_subsampled_data, temp_molecule_random_data = read_data_from_one_dir(directory)
        overall_subsampled_data += temp_molecule_subsampled_data
        overall_random_data += random_subsampling(temp_molecule_random_data, num_random_per_molecule)
        temp_molecule_name = directory.split('/')[-2]
        print directory.split('/')
        print "molecule name: {}".format(temp_molecule_name)
        molecule_list += [temp_molecule_name] * num_random_per_molecule



    list_subsample = setup["subsample_feature_list"]
    temp_list_subsample = setup["subsample_feature_list"]
    if temp_list_subsample == []:
        for m in range(len(overall_subsampled_data[0])):
            temp_list_subsample.append(m)

    #if len(temp_list_subsample) <= 10:
    #    overall_subsampled_data = subsampling_system(overall_subsampled_data, list_desc = list_subsample, cutoff_sig = float(setup["subsample_cutoff_sig"]), rate = float(setup["subsample_rate"]))
    #else:
    #    overall_subsampled_data = subsampling_system_with_PCA(overall_subsampled_data, list_desc = list_subsample, cutoff_sig = float(setup["subsample_cutoff_sig"]), rate = float(setup["subsample_rate"]),start_trial_component = 9)


    overall = overall_random_data + overall_subsampled_data
    #overall = overall_subsampled_data



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

    molecule = []
    
    return X_train, y_train, dens, molecule_list


def fit_model(LDA_result, dens, X_train, residual, loss, tol, slowdown_factor, early_stop_trials):

    NN_model,loss_result = fit_with_KerasNN(X_train * 1e6, residual * 1e6, loss, tol, slowdown_factor, early_stop_trials)
    save_resulting_figure(dens,result.x,X_train,NN_model,y,loss,loss_result)

    return NN_model


if __name__ == "__main__":


    setup_filename = sys.argv[1]
    dataset_name = sys.argv[2]


    with open(setup_filename) as f:
        setup = json.load(f)



    h = float(setup['grid_spacing'])
    L = float(setup['box_dimension'])
    N = int(setup['number_segment_per_side'])
    dir_name = "{}_{}_{}".format(str(L).replace('.','-'),str(h).replace('.','-'),N)

    working_dir = os.getcwd() + '/' + dir_name + '/' + dataset_name

    setup["working_dir"] = working_dir

    
    
    plot_save_result = get_training_data(dataset_name,setup)
   
    #NN_model = fit_with_KerasNN(X_train * 1e6, residual * 1e6, loss, tol, slowdown_factor, early_stop_trials)
    #save_resulting_figure(dens,result.x,X_train,NN_model,y)

    os.chdir(setup["working_dir"])
    with open('test_data_to_plot_extra_large_molecule.pickle', 'wb') as handle:
        pickle.dump(plot_save_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    

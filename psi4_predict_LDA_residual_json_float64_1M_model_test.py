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
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation
from keras import backend as K
import keras
import scipy

import itertools
import multiprocessing

try: import cPickle as pickle
except: import pickle
import matplotlib.pyplot as plt
from subsampling import subsampling_system,random_subsampling,subsampling_system_with_PCA
import keras.backend as K

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

def get_start_loss(log_filename,loss):
    
    with open(log_filename, 'r') as f:
        for line in f:
            pass
        temp = line
    
    if temp.strip().startswith('updated') and temp.split()[9] == loss:
        return float(temp.split()[2])
    else:
        raise ValueError

def test_KerasNN(X, y):

    loss_list = ["mse","sae","mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error", "mean_squared_logarithmic_error", "squared_hinge", "hinge", "categorical_hinge", "logcosh", "categorical_crossentropy", "sparse_categorical_crossentropy", "binary_crossentropy", "kullback_leibler_divergence", "poisson", "cosine_proximity"]
    if loss not in loss_list:
        raise NotImplemented


    filename = "NN.h5"
    log_filename = "NN_test_result.log"
    num_samples = len(y)

    n_layers = setup["NN_setup"]["number_layers"]
    n_per_layer = setup["NN_setup"]["number_neuron_per_layer"]
    activation = setup["NN_setup"]["activation"]


    model = load_model(filename, custom_objects={'sae': sae})

    y_original = y / 1e6
    y_predict = model.predict(X) / 1e6

    y_original_result = np.sum(y_original) *0.02 * 0.02 * 0.02 *27.2114
    y_predict_result = np.sum(y_predict) *0.02 * 0.02 * 0.02 *27.2114
    error =  np.sum(y_predict - y_original) *0.02 * 0.02 * 0.02 *27.2114

    abs_error = np.sum(np.abs(y_predict - y_original)) *0.02 * 0.02 * 0.02 *27.2114

    log(log_filename,"y: {}\t predicted_y: {}\t absolute_error: {}\t error:{}".format(y_original_result, y_predict_result, error, abs_error))

    return


def fit_with_LDA(density,energy):

    filename = "LDA_model.sav"
    text_filename = "LDA_model_result.txt"

    temp_res = pickle.load(open(filename, 'rb'))
    res = temp_res


    predict_y = predict_LDA(density,res.x)
    residual = energy - predict_y

    return residual, res


def LDA_least_suqare_fit(x,density,energy):


    result = np.mean(np.square(lda_x(density,x) + lda_c(density,x) - energy))
    return result


def get_x0():
    x = [ -0.45816529328314287, 0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294]
    return x

def optimization_constants(x):

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



def get_training_data_entry(entry, index_list):
    result = []
    for index in index_list:
        result.append(entry[index])
    return result




def get_training_data(dataset_name,setup, dataset_setup):

    os.chdir(setup["data_dir"])
    overall = pickle.load(open(setup["training_data_filename"],'rb'))

    descriptor_index_list = dataset_setup[dataset_name]

    X_train = []
    y_train = []
    dens = []

    for entry in overall:
        X_train.append(get_training_data_entry(entry, descriptor_index_list))
        dens.append(entry[1])
        y_train.append(entry[0])
    
    
    X_train = (np.asarray(X_train))
    y_train = np.asarray(y_train).reshape((len(y_train),1))
    dens = np.asarray(dens).reshape((len(dens),1))
    
    return X_train, y_train, dens


def test_model(LDA_result, dens, X_train, residual, y):

    test_KerasNN(X_train * 1e6, residual * 1e6)

    return 


if __name__ == "__main__":


    setup_filename = sys.argv[1]
    dataset_setup_filename = sys.argv[2]
    dataset_name = sys.argv[3]

    with open(setup_filename) as f:
        setup = json.load(f)

    with open(dataset_setup_filename) as f:
        dataset_setup = json.load(f)



    K.set_floatx('float64')
    K.floatx()

    h = float(setup['grid_spacing'])
    L = float(setup['box_dimension'])
    N = int(setup['number_segment_per_side'])
    dir_name = "{}_{}_{}".format(str(L).replace('.','-'),str(h).replace('.','-'),N)

    data_dir = os.getcwd() + '/' + dir_name + '/epxc_MCSH_2_0-20_real_real/model_test/'

    setup["data_dir"] = data_dir

    working_dir = os.getcwd() + '/' + dir_name + '/epxc_MCSH_2_0-20_real_real/model_test/' + dataset_name

    setup["working_dir"] = working_dir

    model_save_dir = working_dir + "/" + "NN_LDA_residual_1M_{}_{}_{}".format(setup["NN_setup"]["number_neuron_per_layer"], setup["NN_setup"]["number_layers"], setup["NN_setup"]["activation"])
   
    setup["model_save_dir"] = model_save_dir

    
    
    X_train,y, dens = get_training_data(dataset_name,setup, dataset_setup)
   
    if os.path.isdir(model_save_dir) == False:
        os.makedirs(model_save_dir)

    os.chdir(data_dir)
    #residual,li_model = fit_with_Linear(dens,y)

    residual, LDA_result = fit_with_LDA(dens,y)
    setup['LDA_model'] = LDA_result

    os.chdir(model_save_dir)

    test_model(LDA_result, dens, X_train, residual, y)



    

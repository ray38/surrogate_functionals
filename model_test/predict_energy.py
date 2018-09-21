# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 16:12:24 2017

@author: ray
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 13:56:12 2017

@author: ray
"""

import os
import itertools
import h5py
import json
import sys
import csv
import numpy as np
from numpy import mean, sqrt, square, arange
try: import cPickle as pickle
except: import pickle
import time
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation
from keras import backend as K
import matplotlib.pyplot as plt
import pprint
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline



def sae(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true))

def map_to_0_1(arr, maxx, minn):
    return np.divide(np.subtract(arr,minn),(maxx-minn))
    
def map_back_0_1(arr, maxx, minn):
    return np.add(np.multiply(arr,(maxx-minn)),minn)
    
def map_back_n1_1(arr, maxx, minn):
    temp = np.multiply(np.add(arr,1.),0.5)
    return np.add(np.multiply(temp,(maxx-minn)),minn)

def log(log_filename, text):
    with open(log_filename, "a") as myfile:
        myfile.write(text)
    return

def get_start_loss(log_filename):
    
    with open(log_filename, 'r') as f:
        for line in f:
            pass
        temp = line
    
    if temp.strip().startswith('updated'):
        return float(temp.split()[2]), temp.split()[9]
    else:
        raise ValueError

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



def predict_each_block(setup,dens,X,y):

    NN_model = setup["NN_model"]
    LDA_model = setup["LDA_model"]
    y_transform  = setup["dataset_setup"]["target_transform"]

    original_y = detransform_data(y, y_transform)

    raw_predict_y = predict_LDA(dens,LDA_model.x) + (NN_model.predict(X*1e6)/1e6)
    predict_y = detransform_data(raw_predict_y, y_transform)

    return original_y, predict_y


def detransform_data(temp_data, transform):
    if transform == "real":
        return temp_data
    elif transform == "log10":
        return np.power(10.,temp_data)
    elif transform == "neglog10":
        return np.multiply(-1.,np.power(10.,temp_data))

def transform_data(temp_data, transform):
    if transform == "real":
        return temp_data.flatten().tolist()
    elif transform == "log10":
        return np.log10(temp_data.flatten()).tolist()
    elif transform == "neglog10":
        return np.log10(np.multiply(-1., temp_data.flatten())).tolist()
    
def load_data_each_block(molecule,functional,i,j,k, dataset_setup, molecule_data_directory):
    data_filename = "{}_{}_{}_{}_{}_all_descriptors.hdf5".format(molecule,functional,i,j,k)

    os.chdir(molecule_data_directory)


    data =  h5py.File(data_filename,'r')
    

    result_list = []
    y = []
    dens = []
    
    target = dataset_setup['target']
 
    if target == 'epxc':
        target_set_name = 'epsilon_xc'
    if target == 'tau':
        target_set_name = 'tau'

    
    temp_data = np.asarray(data[target_set_name])
    y.append(transform_data(temp_data, dataset_setup['target_transform']))
    
    if int(dataset_setup['density']) == 1:
        temp_data = np.asarray(data['rho'])
        result_list.append(transform_data(temp_data, dataset_setup['density_transform']))
        dens.append(transform_data(temp_data, dataset_setup['density_transform']))

    if int(dataset_setup['gamma']) == 1:
        temp_data = np.asarray(data['gamma'])
        result_list.append(transform_data(temp_data, dataset_setup['gamma_transform']))

    if int(dataset_setup['tau']) == 1:
        temp_data = np.asarray(data['tau'])
        result_list.append(transform_data(temp_data, dataset_setup['tau_transform']))


    group_name = 'derivative'
    temp_list = dataset_setup["derivative_list"]
    if len(temp_list) > 0: 
        for derivative_count in temp_list:
            dataset_name = 'derivative_{}'.format(derivative_count)
            temp_data = np.asarray(data[group_name][dataset_name])
            result_list.append(transform_data(temp_data, dataset_setup['derivative_transform']))

    temp_list = dataset_setup["derivative_square_list"]
    if len(temp_list) > 0: 
        for derivative_count in temp_list:
            dataset_name = 'derivative_{}'.format(derivative_count)
            temp_data = np.power(np.asarray(data[group_name][dataset_name]), 2.)
            result_list.append(transform_data(temp_data, dataset_setup['derivative_square_transform']))
    
   
    group_name = 'average_density'
    temp_list = dataset_setup["average_density_r_list"]
    if len(temp_list) > 0:
        for r_list_count in temp_list:
            dataset_name = 'average_density_{}'.format(str(r_list_count).replace('.','-'))
            temp_data = np.asarray(data[group_name][dataset_name])
            result_list.append(transform_data(temp_data, dataset_setup['average_density_transform']))


    group_name = 'asym_integral'
    temp_list = dataset_setup["asym_desc_r_list"]
    if len(temp_list) > 0:
        for r_list_count in temp_list:
            dataset_name = 'asym_integral_x_{}'.format(str(r_list_count).replace('.','-'))
            temp_data = np.asarray(data[group_name][dataset_name])
            result_list.append(transform_data(temp_data, dataset_setup['asym_desc_transform']))

            dataset_name = 'asym_integral_y_{}'.format(str(r_list_count).replace('.','-'))
            temp_data = np.asarray(data[group_name][dataset_name])
            result_list.append(transform_data(temp_data, dataset_setup['asym_desc_transform']))

            dataset_name = 'asym_integral_z_{}'.format(str(r_list_count).replace('.','-'))
            temp_data = np.asarray(data[group_name][dataset_name])
            result_list.append(transform_data(temp_data, dataset_setup['asym_desc_transform']))

    
    group_name = 'MC_surface_spherical_harmonic'

    try:
        temp_list = dataset_setup["MC_surface_spherical_harmonic_0_r_list"]
        if len(temp_list) > 0:
            for r_list_count in temp_list:
                dataset_name = 'MC_surface_shperical_harmonic_0_{}'.format(str(r_list_count).replace('.','-'))
                temp_data = np.asarray(data[group_name][dataset_name])
                result_list.append(transform_data(temp_data, dataset_setup['MC_surface_spherical_harmonic_0_transform']))
    except:
        pass

    
    try:
        temp_list = dataset_setup["MC_surface_spherical_harmonic_1_r_list"]
        if len(temp_list) > 0:
            for r_list_count in temp_list:
                dataset_name = 'MC_surface_shperical_harmonic_1_{}'.format(str(r_list_count).replace('.','-'))
                temp_data = np.asarray(data[group_name][dataset_name])
                result_list.append(transform_data(temp_data, dataset_setup['MC_surface_spherical_harmonic_1_transform']))
    except:
        pass

    try:
        temp_list = dataset_setup["MC_surface_spherical_harmonic_2_r_list"]
        if len(temp_list) > 0:
            for r_list_count in temp_list:
                dataset_name = 'MC_surface_shperical_harmonic_2_{}'.format(str(r_list_count).replace('.','-'))
                temp_data = np.asarray(data[group_name][dataset_name])
                result_list.append(transform_data(temp_data, dataset_setup['MC_surface_spherical_harmonic_2_transform']))
    except:
        pass

    try:
        temp_list = dataset_setup["MC_surface_spherical_harmonic_3_r_list"]
        if len(temp_list) > 0:
            for r_list_count in temp_list:
                dataset_name = 'MC_surface_shperical_harmonic_3_{}'.format(str(r_list_count).replace('.','-'))
                temp_data = np.asarray(data[group_name][dataset_name])
                result_list.append(transform_data(temp_data, dataset_setup['MC_surface_spherical_harmonic_3_transform']))
    except:
        pass

    try:
        temp_list = dataset_setup["MC_surface_spherical_harmonic_4_r_list"]
        if len(temp_list) > 0:
            for r_list_count in temp_list:
                dataset_name = 'MC_surface_shperical_harmonic_4_{}'.format(str(r_list_count).replace('.','-'))
                temp_data = np.asarray(data[group_name][dataset_name])
                result_list.append(transform_data(temp_data, dataset_setup['MC_surface_spherical_harmonic_4_transform']))
    except:
        pass



    result = zip(*result_list)
    y = zip(*y)
    dens = zip(*dens)

    os.chdir(setup["working_dir"])
    return np.asarray(dens), np.asarray(result), np.asarray(y)

def process_each_block(molecule, i,j,k, setup, molecule_data_directory):

    h = setup['grid_spacing']
    functional = setup['functional']

    start = time.time()
    dens, X,y = load_data_each_block(molecule,functional,i,j,k, setup["dataset_setup"], molecule_data_directory)
    original_y, predict_y = predict_each_block(setup, dens, X, y)
    dv = h*h*h
    y = original_y * dv*27.2114
    y_predict = predict_y*dv*27.2114
    y_sum = np.sum(y)
    y_predict_sum = np.sum(y_predict)
    error = y-y_predict
    absolute_error = np.abs(error)

    sum_error = y_sum - y_predict_sum
    sum_absolute_error = np.sum(absolute_error)
    
    os.chdir(setup["working_dir"])
    log(setup["predict_full_log_name"],"\ndone predicting: " + molecule + "\t took time: " + str(time.time()-start)+ "\t" + str(i) + "\t" + str(j) + "\t" + str(k))
    log(setup["predict_full_log_name"],"\nenergy: " + str(y_sum) + "\tpredicted energy: " + str(y_predict_sum) + "\tpredict error: " + str(sum_error) + "\tabsolute error: " + str(sum_absolute_error))


    return sum_error, sum_absolute_error,y_predict_sum, y_sum 



def process_one_molecule(molecule, molecule_data_directory, setup):
 

    N = setup["N"]
    Nx = Ny = Nz = N
    i_li = range(Nx)
    j_li = range(Ny)
    k_li = range(Nz)
    
    system_sum_error = 0 
    system_sum_absolute_error = 0
    system_y_predict_sum = 0
    system_y_sum = 0 
    
    paramlist = list(itertools.product(i_li,j_li,k_li))
    for i,j,k in paramlist:
        
        sum_error, sum_absolute_error,y_predict_sum, y_sum = process_each_block(molecule, i,j,k, setup, molecule_data_directory)
        system_sum_error += sum_error
        system_sum_absolute_error += sum_absolute_error
        system_y_predict_sum += y_predict_sum
        system_y_sum += y_sum


    os.chdir(setup["model_save_dir"])
    log(setup["predict_log_name"],"\n\ndone predicting: " + molecule )
    log(setup["predict_log_name"],"\nenergy: " + str(system_y_sum) + "\tpredicted energy: " + str(system_y_predict_sum) + "\tpredicted error: " + str(system_sum_error)+ "\n")

    log(setup["predict_full_log_name"],"\n\ndone predicting: " + molecule )
    log(setup["predict_full_log_name"],"\nenergy: " + str(system_y_sum) + "\tpredicted energy: " + str(system_y_predict_sum) + "\tpredicted error: " + str(system_sum_error)+ "\n") 
    
    log(setup["predict_error_log_name"], "\n{}\t{}\t{}\t{}\t{}".format(molecule, system_y_sum, system_y_predict_sum, system_sum_error,system_sum_absolute_error))
    return system_sum_error,system_sum_absolute_error, system_y_predict_sum,system_y_sum

    


def initialize(setup,NN_model_filename,LDA_model_filename):


    try:
        NN_model = load_model(NN_model_filename, custom_objects={'sae': sae})
    except:
        NN_model = load_model(NN_model_filename)

    LDA_model = pickle.load(open(LDA_model_filename, 'rb'))

    predict_log_name = "predict_log.log"
    predict_full_log_name = "predict_full_log.log"
    predict_error_log_name = "predict_error_log.log"


    setup["NN_model"] = NN_model
    setup["LDA_model"] = LDA_model
    setup["predict_log_name"] = predict_log_name
    setup["predict_full_log_name"] = predict_full_log_name
    setup["predict_error_log_name"] = predict_error_log_name

    return


if __name__ == "__main__":

    dataset_setup_database_filename = sys.argv[1]
    dataset_name = sys.argv[2]
    NN_model_filename = sys.argv[3]
    LDA_model_filename = sys.argv[4]
    molecule = sys.argv[5]
    molecule_data_directory = sys.argv[6]
    num_box_per_side = int(sys.argv[7])

    K.set_floatx('float64')
    K.floatx()

    setup = {}
    setup["NN_model_filename"] = NN_model_filename
    setup["LDA_model_filename"] = LDA_model_filename
    setup["N"] = num_box_per_side
    setup["functional"] = "B3LYP"
    setup['grid_spacing'] = 0.02


    with open(dataset_setup_database_filename) as f:
        setup_database = json.load(f)
    dataset_setup = setup_database[dataset_name]

    setup["dataset_setup"] = dataset_setup
    setup["working_dir"] = os.getcwd()



    initialize(setup,NN_model_filename)

    temp_error,temp_absolute_error, temp_y_predict,temp_y = process_one_molecule(molecule,molecule_data_directory, setup)




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
import numpy as np
from numpy import mean, sqrt, square, arange
try: import cPickle as pickle
except: import pickle
import time
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt
import pprint
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

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
        return float(temp.split()[2])
    else:
        raise ValueError



def predict_each_block(setup,dens,X,y):

    NN_model = setup["NN_model"]
    linear_model = setup["linear_model"]
    y_transform  = setup["dataset_setup"]["target_transform"]

    original_y = detransform_data(y, y_transform)

    #raw_predict_y = linear_model.predict(dens) + NN_model.predict(X)
    #predict_y = detransform_data(raw_predict_y, y_transform)

    return original_y, np.zeors_like(original_y)#, predict_y


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
    
def load_data_each_block(molecule,functional,i,j,k, dataset_setup, data_dir_full):
    data_filename = "{}_{}_{}_{}_{}_all_descriptors.hdf5".format(molecule,functional,i,j,k)

    os.chdir(data_dir_full)

    print data_dir_full
    print os.getcwd()

    data =  h5py.File(data_filename,'r')
    

    result_list = []
    y = []
    dens = []
    
    target = dataset_setup['target']
    
    if target == 'Vxc':
        target_set_name = 'V_xc'   
    if target == 'epxc':
        target_set_name = 'epsilon_xc'
    if target == 'tau':
        target_set_name = 'tau'
    if target == 'gamma':
        target_set_name = 'gamma'
    
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

    

    result = zip(*result_list)
    y = zip(*y)
    dens = zip(*dens)
    print "done loading: {} {} {}".format(i,j,k)

    os.chdir(setup["model_save_dir"])
    return np.asarray(dens), np.asarray(result), np.asarray(y)


def process_each_block(molecule, i,j,k, setup, data_dir_full):
    print 'started'

    h = float(setup['grid_spacing'])
    L = float(setup['box_dimension'])
    N = int(setup['number_segment_per_side'])
    functional = setup['functional']
    #log_filename = setup["predict_full_log_name"]

    start = time.time()
    dens, X,y = load_data_each_block(molecule,functional,i,j,k, setup["dataset_setup"], data_dir_full)
    original_y, predict_y = predict_each_block(setup, dens, X, y)
    dv = h*h*h
    y = original_y * dv*27.2114
    y_predict = predict_y*dv*27.2114
    y_sum = np.sum(y)
    y_predict_sum = np.sum(y_predict)
    error = y-y_predict
    
#    fraction_error = np.divide(error, y)
    sum_error = y_sum - y_predict_sum
    
    os.chdir(setup["model_save_dir"])
    log(setup["predict_full_log_name"],"\ndone predicting: " + molecule + "\t took time: " + str(time.time()-start)+ "\t" + str(i) + "\t" + str(j) + "\t" + str(k))
    log(setup["predict_full_log_name"],"\nenergy: " + str(y_sum) + "\tpredicted energy: " + str(y_predict_sum) + "\tpredicted error: " + str(sum_error)) 

    os.chdir(setup["model_save_dir"])

    return sum_error, y_predict_sum, y_sum   



def process_one_molecule(molecule, setup):
 

    h = float(setup['grid_spacing'])
    L = float(setup['box_dimension'])
    N = int(setup['number_segment_per_side'])

    functional = setup['functional']  

#    log_filename = setup["predict_log_name"]
    
    data_dir_name = "{}_{}_{}_{}_{}".format(molecule,functional,str(L).replace('.','-'),str(h).replace('.','-'),N)


    data_dir_full = setup["sub_database"]  + "/" + data_dir_name

#    os.chdir(data_dir_full)

    Nx = Ny = Nz = N
    i_li = range(Nx)
    j_li = range(Ny)
    k_li = range(Nz)
    
    system_sum_error = 0 
    system_y_predict_sum = 0
    system_y_sum = 0 
    
    paramlist = list(itertools.product(i_li,j_li,k_li))
    for i,j,k in paramlist:
        
        sum_error, y_predict_sum, y_sum = process_each_block(molecule, i,j,k, setup, data_dir_full)
        system_sum_error += sum_error
        system_y_predict_sum += y_predict_sum
        system_y_sum += y_sum


    os.chdir(setup["model_save_dir"])
    log(setup["predict_log_name"],"\n\ndone predicting: " + molecule )
    log(setup["predict_log_name"],"\nenergy: " + str(system_y_sum) + "\tpredicted energy: " + str(system_y_predict_sum) + "\tpredicted error: " + str(system_sum_error)+ "\n")

    log(setup["predict_full_log_name"],"\n\ndone predicting: " + molecule )
    log(setup["predict_full_log_name"],"\nenergy: " + str(system_y_sum) + "\tpredicted energy: " + str(system_y_predict_sum) + "\tpredicted error: " + str(system_sum_error)+ "\n") 
    
    log(setup["predict_error_log_name"], "\n{}\t{}\t{}\t{}".format(molecule, system_y_sum, system_y_predict_sum, system_sum_error))
    return system_sum_error, system_y_predict_sum,system_y_sum

    


def initialize(setup):
    os.chdir(setup["model_save_dir"])
    fit_log_name = "NN_fit_log.log"
    
    linear_model_name = "Linear_model.sav"
    NN_model_name = "NN.h5"

    start_loss = get_start_loss(fit_log_name)
    predict_log_name = "predict_{}_log.log".format(start_loss)
    predict_full_log_name = "predict_{}_full_log.log".format(start_loss)
    predict_error_log_name = "predict_{}_error_log.log".format(start_loss)
    NN_model = load_model(NN_model_name)
    linear_model = pickle.load(open(linear_model_name , 'rb'))

    setup["NN_model"] = NN_model
    setup["linear_model"] = linear_model
    setup["predict_log_name"] = predict_log_name
    setup["predict_full_log_name"] = predict_full_log_name
    setup["predict_error_log_name"] = predict_error_log_name

    os.chdir(setup["working_dir"])
    return


if __name__ == "__main__":

    print "start"
    predict_setup_filename = sys.argv[1]
    dataset_setup_database_filename = sys.argv[2]
    dataset_name = sys.argv[3]
    functional = sys.argv[4]



    with open(predict_setup_filename) as f:
        setup = json.load(f)


    h = float(setup['grid_spacing'])
    L = float(setup['box_dimension'])
    N = int(setup['number_segment_per_side'])

    setup['functional'] = functional
#    functional = setup['functional']

    with open(dataset_setup_database_filename) as f:
        setup_database = json.load(f)

    dataset_setup = setup_database[dataset_name]

    setup["dataset_setup"] = dataset_setup

    dir_name = "{}_{}_{}".format(str(L).replace('.','-'),str(h).replace('.','-'),N)

    working_dir = os.getcwd() + '/' + dir_name + '/' + dataset_name
    setup["working_dir"] = working_dir

    model_save_dir = working_dir + "/" + "NN_{}_{}_{}_{}".format(setup["fit_type"],setup["NN_setup"]["number_neuron_per_layer"], setup["NN_setup"]["number_layers"], setup["NN_setup"]["activation"])
    setup["model_save_dir"] = model_save_dir


    database_name = setup["database_directory"]
    sub_database_name = "{}_{}_{}".format(str(L).replace('.','-'),str(h).replace('.','-'),N)

    setup["sub_database"] = database_name + '/' + sub_database_name

    initialize(setup)


    error_list = []
    for molecule in setup["molecule_list"]:
        try:
            temp_error,temp_y_predict,temp_y = process_one_molecule(molecule, setup)
            error_list.append(temp_error)
        except:
            log(setup["predict_log_name"],"\n\n Failed")
            log(setup["predict_full_log_name"],"\n\n Failed") 
    

    log(setup["predict_log_name"],"\n\naverage error: " + str(np.mean(error_list)) + "\tstddev error: " + str(np.std(error_list))) 
    log(setup["predict_log_name"],"\n\naverage abs error: " + str(np.mean(np.abs(error_list))) + "\tstddev abs error: " + str(np.std(np.abs(error_list))))
    log(setup["predict_log_name"],"\n\naverage abs error: " + str(sqrt(mean(square(error_list)))))

    log(setup["predict_full_log_name"],"\n\naverage error: " + str(np.mean(error_list)) + "\tstddev error: " + str(np.std(error_list))) 
    log(setup["predict_full_log_name"],"\n\naverage abs error: " + str(np.mean(np.abs(error_list))) + "\tstddev abs error: " + str(np.std(np.abs(error_list))))
    log(setup["predict_full_log_name"],"\n\naverage abs error: " + str(sqrt(mean(square(error_list)))))




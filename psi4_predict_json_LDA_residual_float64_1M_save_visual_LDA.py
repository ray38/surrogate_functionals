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
from keras import backend as K
import matplotlib.pyplot as plt
import pprint
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import csv

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
        return float(temp.split()[2])
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

    raw_predict_y = predict_LDA(dens,LDA_model.x)
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
    
def load_data_each_block(molecule,functional,i,j,k, dataset_setup, data_dir_full):
    data_filename = "{}_{}_{}_{}_{}_all_descriptors.hdf5".format(molecule,functional,i,j,k)

    os.chdir(data_dir_full)

    print data_dir_full
    print os.getcwd()

    data =  h5py.File(data_filename,'r')
    

    result_list = []
    coordinate_list = []
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



    print "start loading coordinates"


    raw_data_filename = "{}_{}_{}_{}_{}.hdf5".format(molecule,functional,i,j,k)
    raw_data = h5py.File(raw_data_filename,'r')

    temp_coord_x = np.asarray(raw_data['x'])
    temp_coord_y = np.asarray(raw_data['y'])
    temp_coord_z = np.asarray(raw_data['z'])
    temp_coord_n = np.asarray(raw_data['rho'])




    temp_x_list = down_sample(np.around(temp_coord_x,2).flatten().tolist(), 5)
    temp_y_list = down_sample(np.around(temp_coord_y,2).flatten().tolist(), 5)
    temp_z_list = down_sample(np.around(temp_coord_z,2).flatten().tolist(), 5)
    temp_n_list = down_sample(np.around(temp_coord_n,9).flatten().tolist(), 5)



    print "done loading coordinates"

    result = zip(*result_list)
    y = zip(*y)
    dens = zip(*dens)
    print "done loading: {} {} {}".format(i,j,k)

    os.chdir(setup["model_save_dir"])
    return np.asarray(dens), np.asarray(result), np.asarray(y), temp_x_list, temp_y_list, temp_z_list, temp_n_list


def down_sample(list,space):
    result_list = []

    #for counter, value in enumerate(list):
    #    if counter%space == 0:
    #        result_list.append(value)


    a = np.arange(1000000).reshape(100,100,100)
    index_list = a[::5,::5,::5].flatten().tolist()
    for index in index_list:
        result_list.append(list[index])

    return result_list

def process_each_block(molecule, i,j,k, setup, data_dir_full):
    print 'started'

    h = float(setup['grid_spacing'])
    L = float(setup['box_dimension'])
    N = int(setup['number_segment_per_side'])
    functional = setup['functional']
    #log_filename = setup["predict_full_log_name"]

    start = time.time()
    dens, X,y, temp_x_list, temp_y_list, temp_z_list, temp_n_list = load_data_each_block(molecule,functional,i,j,k, setup["dataset_setup"], data_dir_full)
    original_y, predict_y = predict_each_block(setup, dens, X, y)
    dv = h*h*h
    y = original_y * dv*27.2114
    y_predict = predict_y*dv*27.2114
    y_sum = np.sum(y)
    y_predict_sum = np.sum(y_predict)
    error = y-y_predict

    temp_original_y_list = down_sample(np.around(y,9).flatten().tolist(), 5)
    temp_predict_y_list  = down_sample(np.around(y_predict,9).flatten().tolist(), 5)

    
#    fraction_error = np.divide(error, y)
    sum_error = y_sum - y_predict_sum
    
    os.chdir(setup["model_save_dir"])
    log(setup["predict_full_log_name"],"\ndone predicting: " + molecule + "\t took time: " + str(time.time()-start)+ "\t" + str(i) + "\t" + str(j) + "\t" + str(k))
    log(setup["predict_full_log_name"],"\nenergy: " + str(y_sum) + "\tpredicted energy: " + str(y_predict_sum) + "\tpredicted error: " + str(sum_error)) 

    os.chdir(setup["model_save_dir"])

    return sum_error, y_predict_sum, y_sum, temp_x_list, temp_y_list, temp_z_list, temp_n_list, temp_original_y_list, temp_predict_y_list



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

    overall_list = []

    x_list = []
    y_list = []
    z_list = []
    n_list = []
    original_y_list = []
    predict_y_list = []


    
    paramlist = list(itertools.product(i_li,j_li,k_li))
    for i,j,k in paramlist:
        
        sum_error, y_predict_sum, y_sum, temp_x_list, temp_y_list, temp_z_list, temp_n_list, temp_original_y_list, temp_predict_y_list = process_each_block(molecule, i,j,k, setup, data_dir_full)

        system_sum_error += sum_error
        system_y_predict_sum += y_predict_sum
        system_y_sum += y_sum
        x_list += temp_x_list
        y_list += temp_y_list
        z_list += temp_z_list
        n_list += temp_n_list
        original_y_list += temp_original_y_list
        predict_y_list += temp_predict_y_list



    #sum_error, y_predict_sum, y_sum, temp_x_list, temp_y_list, temp_z_list, temp_n_list, temp_original_y_list, temp_predict_y_list = process_each_block(molecule, 2,2,2, setup, data_dir_full)
    #system_sum_error += sum_error
    #system_y_predict_sum += y_predict_sum
    #system_y_sum += y_sum
    #x_list += temp_x_list
    #y_list += temp_y_list
    #z_list += temp_z_list
    #n_list += temp_n_list
    #original_y_list += temp_original_y_list
    #predict_y_list += temp_predict_y_list

    #print len(temp_x_list)
    #print len(temp_y_list)
    #print len(temp_z_list)
    #print len(temp_n_list)
    #print len(temp_original_y_list)
    #print len(temp_predict_y_list)

    overall_list.append(x_list)
    overall_list.append(y_list)
    overall_list.append(z_list)
    overall_list.append(n_list)




    overall_list.append(original_y_list)
    overall_list.append(predict_y_list)
    overall_list = np.stack(overall_list,axis=1).tolist()


    os.chdir(setup["model_save_dir"])
    log(setup["predict_log_name"],"\n\ndone predicting: " + molecule )
    log(setup["predict_log_name"],"\nenergy: " + str(system_y_sum) + "\tpredicted energy: " + str(system_y_predict_sum) + "\tpredicted error: " + str(system_sum_error)+ "\n")

    log(setup["predict_full_log_name"],"\n\ndone predicting: " + molecule )
    log(setup["predict_full_log_name"],"\nenergy: " + str(system_y_sum) + "\tpredicted energy: " + str(system_y_predict_sum) + "\tpredicted error: " + str(system_sum_error)+ "\n") 
    
    log(setup["predict_error_log_name"], "\n{}\t{}\t{}\t{}".format(molecule, system_y_sum, system_y_predict_sum, system_sum_error))


    with open("{}_downsampled_prediction_LDA_data.csv".format(molecule), "wb") as f:
        writer = csv.writer(f)
        writer.writerow(['x','y','z','rho', 'epxc','epxc_LDA_predict'])
        writer.writerows(overall_list)

    return system_sum_error, system_y_predict_sum,system_y_sum

    


def initialize(setup):
    os.chdir(setup["model_save_dir"])
    fit_log_name = "NN_fit_log.log"
    
    LDA_model_name = "LDA_model.sav"
    NN_model_name = "NN.h5"

    start_loss = get_start_loss(fit_log_name)
    predict_log_name = "predict_{}_viz_LDA_log.log".format(start_loss)
    predict_full_log_name = "predict_{}_viz_LDA_full_log.log".format(start_loss)
    predict_error_log_name = "predict_{}_viz_LDA_error_log.log".format(start_loss)
    try:
        NN_model = load_model(NN_model_name, custom_objects={'sae': sae})
    except:
        NN_model = load_model(NN_model_name)
    LDA_model = pickle.load(open(LDA_model_name, 'rb'))

    setup["NN_model"] = NN_model
    setup["LDA_model"] = LDA_model
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

    K.set_floatx('float64')
    K.floatx()


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
        #try:
        temp_error,temp_y_predict,temp_y = process_one_molecule(molecule, setup)
        error_list.append(temp_error)
        #except:
        #    log(setup["predict_log_name"],"\n\n Failed")
        #    log(setup["predict_full_log_name"],"\n\n Failed") 
    

    log(setup["predict_log_name"],"\n\naverage error: " + str(np.mean(error_list)) + "\tstddev error: " + str(np.std(error_list))) 
    log(setup["predict_log_name"],"\n\naverage abs error: " + str(np.mean(np.abs(error_list))) + "\tstddev abs error: " + str(np.std(np.abs(error_list))))
    log(setup["predict_log_name"],"\n\naverage abs error: " + str(sqrt(mean(square(error_list))))) 

    log(setup["predict_full_log_name"],"\n\naverage error: " + str(np.mean(error_list)) + "\tstddev error: " + str(np.std(error_list))) 
    log(setup["predict_full_log_name"],"\n\naverage abs error: " + str(np.mean(np.abs(error_list))) + "\tstddev abs error: " + str(np.std(np.abs(error_list))))
    log(setup["predict_full_log_name"],"\n\naverage abs error: " + str(sqrt(mean(square(error_list))))) 




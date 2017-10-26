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
try: import cPickle as pickle
except: import pickle
import time
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt
import pprint

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
    

def load_data_each_block(molecule,functional,i,j,k, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform='real',target_transform = 'real'):
    data_filename = "{}_{}_{}_{}_{}_all_descriptors.hdf5".format(molecule,functional,i,j,k)
    print data_filename
    data =  h5py.File(data_filename,'r')
    r_list = [0.04,0.06,0.08,0.1,0.2,0.3,0.4,0.5]
    
    result_list = []
    y = []
  
    
    if target == 'Vxc':
        target_set_name = 'V_xc'   
    if target == 'epxc':
        target_set_name = 'epsilon_xc'
    if target == 'tau':
        target_set_name = 'tau'
    
#    y = np.asarray(data[target_set_name])
#    temp_data = np.asarray([])
    temp_data = np.asarray(data[target_set_name])
    
    
    if target_transform == 'real':
        y.append(temp_data.flatten().tolist())
    elif target_transform == 'log':
        y.append(np.log10(np.multiply(-1., temp_data.flatten())).tolist())

    
    temp_data = np.asarray(data['rho'])

    if desc_transform == 'real':
        result_list.append(temp_data.flatten().tolist())
    elif desc_transform == 'log':
        result_list.append(np.log10(temp_data.flatten()).tolist())
    
    if gamma == 1:
        temp_data = np.asarray(data['gamma'])
        if desc_transform == 'real':
                result_list.append(temp_data.flatten().tolist())
        elif desc_transform == 'log':
            result_list.append(np.log10(temp_data.flatten()).tolist())
    
    if num_desc_deri > 0:
        group_name = 'derivative'
        for desc_deri_count in range(num_desc_deri):
            dataset_name = 'derivative_{}'.format(desc_deri_count+1)
            temp_data = np.asarray(data[group_name][dataset_name])
            temp = temp_data.flatten()
            result_list.append(temp.tolist())
    
    if num_desc_deri_squa > 0:
        group_name = 'derivative'
        for desc_deri_count in range(num_desc_deri_squa):
            print str(i) + ' start' 
            dataset_name = 'derivative_{}'.format(desc_deri_count+1)
            temp_data = np.asarray(data[group_name][dataset_name])
            temp = np.power(temp_data,2.)
            result_list.append(temp.flatten().tolist())
   
    if num_desc_ave_dens > 0:
        group_name = 'average_density'
        for desc_ave_dens_count in range(num_desc_ave_dens):
            dataset_name = 'average_density_{}'.format(str(r_list[desc_ave_dens_count]).replace('.','-'))
            temp_data = np.asarray(data[group_name][dataset_name])
            if desc_transform == 'real':
                result_list.append(temp_data.flatten().tolist())
            elif desc_transform == 'log':
                result_list.append(np.log10(temp_data.flatten()).tolist())
        
    
    
    result = zip(*result_list)
    y = zip(*y)


    return np.asarray(result), np.asarray(y)

def predict_each_block(model_dict, X, y):
    temp_X_li = []
    temp_y_li = []
    for i in range(len(model_dict)):
        temp_X_li.append([])
        temp_y_li.append([])
    print X[0]
    print y[0]
#    print model_dict[key]['X'][:,0]
    X_temp = X.tolist()
    y_temp = y.reshape(len(y)).tolist()
    
    y_log_temp = np.log10(np.multiply(y.reshape(len(y)),-1.)).tolist()
    for index, x in enumerate(y_log_temp):
        temp_count = 0
        for key in model_dict:
            if x >= model_dict[key]['interval_start'] and x < model_dict[key]['interval_end']:
                temp_X_li[temp_count].append(X_temp[index])
                temp_y_li[temp_count].append(y_temp[index])

                break
            temp_count +=1
    temp_count = 0

    for key in model_dict:
        model_dict[key]['X'] = np.asarray(temp_X_li[temp_count])
        model_dict[key]['y'] = np.asarray(temp_y_li[temp_count])
        temp_count += 1
    
    
    original_y = np.asarray([])
    predict_y  = np.asarray([])
    
    
    for key in model_dict:
        original_y = np.append(original_y, model_dict[key]['y'])
#        print model_dict[key]['X']
        temp_dens = model_dict[key]['X'][:,0]
        temp = model_dict[key]['li_model'].predict(temp_dens.reshape(len(temp_dens),1)) + model_dict[key]['model'].predict(model_dict[key]['X'])
#        predict_y = np.multiply(-1.,(np.power(10.,temp)))
#        model_dict[key]['predicted_y'] = np.multiply(-1.,(np.power(10.,temp)))
#        model_dict[key]['predicted_y'] = np.multiply(-1.,(np.power(10.,model_dict[key]['model'].predict(model_dict[key]['X'] + model_dict[key]['li_model'].predict(model_dict[key]['X'])))))
#        if model_dict[key]['activation'] == 'relu':       
#            model_dict[key]['predicted_y'] = np.multiply(-1.,(np.power(10.,map_back_0_1(model_dict[key]['model'].predict(model_dict[key]['X']),model_dict[key]['model_end'], model_dict[key]['model_start']))))
#        elif model_dict[key]['activation'] == 'tanh':      
#            model_dict[key]['predicted_y'] = np.multiply(-1.,(np.power(10.,map_back_n1_1(model_dict[key]['model'].predict(model_dict[key]['X']),model_dict[key]['model_end'], model_dict[key]['model_start']))))
#        original_y = np.append(original_y, model_dict[key]['y'])
        predict_y  = np.append(predict_y , np.multiply(-1.,(np.power(10.,temp))))
    print original_y.shape
    print predict_y.shape
    return original_y, predict_y


    



def process_one_molecule(molecule, model_dict, functional,h,N,log_filename, target,gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens):
    cwd = os.getcwd()
    dir_name = "{}_{}_{}_{}_{}".format(molecule,functional,str(L).replace('.','-'),str(h).replace('.','-'),N)
    print cwd
    print dir_name
    
#    if os.path.isdir(cwd + '/'+dir_name) == False:
#        print '\n****Error: Cant find the directory! ****\n'
#        raise NotImplementedError
    
    os.chdir(cwd + '/' + dir_name)
    Nx = Ny = Nz = N
    i_li = range(Nx)
    j_li = range(Ny)
    k_li = range(Nz)
    
    system_sum_error = 0 
    system_y_predict_sum = 0
    system_y_sum = 0 
    
    paramlist = list(itertools.product(i_li,j_li,k_li))
    for i,j,k in paramlist:
        
        sum_error, y_predict_sum, y_sum = process_each_block(molecule, model_dict, functional,i,j,k,h,N,log_filename, target,gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens)
        system_sum_error += sum_error
        system_y_predict_sum += y_predict_sum
        system_y_sum += y_sum

    
#    num_cores = multiprocessing.cpu_count()
#    print "number of cores: {}".format(num_cores)

    
    
    
#    pool = multiprocessing.Pool()
#    for i,j,k in paramlist:
#        pool.apply_async(process, args=(molecule, functional,i,j,k,h,N))
#    pool.close()
#    pool.join()

    os.chdir(cwd)
    return system_sum_error, system_y_predict_sum,system_y_sum
    
def process_each_block(molecule, model_dict, functional,i,j,k,h,N,log_filename, target,gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens):
    print 'started'    
    log(log_filename,"\n\nstart loading: " + molecule + "\t" + str(i) + "\t" + str(j) + "\t" + str(k))
    start = time.time()
    X,y = load_data_each_block(molecule,functional,i,j,k, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens)
    original_y, predict_y = predict_each_block(model_dict, X, y)
    dv = h*h*h
    y = original_y * dv*27.2114
    y_predict = predict_y*dv*27.2114
    y_sum = np.sum(y)
    y_predict_sum = np.sum(y_predict)
    error = y-y_predict
    
#    fraction_error = np.divide(error, y)
    sum_error = y_sum - y_predict_sum

    
    log(log_filename,"\ndone predicting: " + molecule + "\t took time: " + str(time.time()-start)+ "\t" + str(i) + "\t" + str(j) + "\t" + str(k))
    log(log_filename,"\nenergy: " + str(y_sum) + "\tpredicted energy: " + str(y_predict_sum) + "\tpredicted error: " + str(sum_error))    

    return sum_error, y_predict_sum, y_sum    


def read_model_inputfile(input_filename,h,L,N,target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens):
    model_dict = {}
    interval_divide = []
    with open(input_filename,'r') as f:
        for line in f:
            temp = line.split()
            model_name = temp[3]
            li_model_name = temp[4]
            temp_info = model_name.replace('.','_').split('_')
            
            cwd = os.getcwd()
            result_dir = "{}_{}_{}_{}_{}_gamma{}_dev{}_devsq{}_inte{}_log_log_models".format(functional,str(L).replace('.','-'),str(h).replace('.','-'),N,target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,temp_info[10],temp_info[11])
            os.chdir(cwd + '/' + result_dir)            
            temp_model = load_model(model_name)
            li_model = pickle.load(open(li_model_name, 'rb'))
            os.chdir(cwd)
            
            model_dict[temp[0]] = {}
            model_dict[temp[0]]['interval_start'] = float(temp[1])
            model_dict[temp[0]]['interval_end']   = float(temp[2])           
            model_dict[temp[0]]['model_name'] = model_name
            model_dict[temp[0]]['model'] = temp_model
            model_dict[temp[0]]['li_model_name'] = li_model_name
            model_dict[temp[0]]['li_model'] = li_model
            model_dict[temp[0]]['desc_transform'] = temp_info[10]
            model_dict[temp[0]]['ener_transform'] = temp_info[11]
#            model_dict[temp[0]]['num_desc'] = int(temp_info[7][3:])
            model_dict[temp[0]]['activation'] = temp_info[6]
            model_dict[temp[0]]['X'] = np.asarray([])
            model_dict[temp[0]]['y'] = np.asarray([])
            model_dict[temp[0]]['model_start'] = float(temp[8])
            model_dict[temp[0]]['model_end']   = float(temp[9])
            #            model_dict[temp[0]]['predicted_y'] = []
            interval_divide.append([float(temp[1]), float(temp[2])])

    
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(model_dict)
    print len(model_dict)
    print interval_divide
    return model_dict, interval_divide


model_setup_filename = sys.argv[1]
list_of_molecule_filename = sys.argv[2]
functional = sys.argv[3]
h = float(sys.argv[4])
L = float(sys.argv[5])
N = int(sys.argv[6])
log_filename = sys.argv[7]
target = sys.argv[8]
gamma = int(sys.argv[9])
num_desc_deri = int(sys.argv[10])
num_desc_deri_squa = int(sys.argv[11])
num_desc_ave_dens = int(sys.argv[12])


model_dict, interval_divide = read_model_inputfile(model_setup_filename,h,L,N,target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens)


log(log_filename,'\n\n***************\n' + pprint.pformat(model_dict, indent=4) + '\n---')


with open(list_of_molecule_filename) as f:
    molecule_names = f.readlines()
molecule_names = [x.strip() for x in molecule_names]

error_list = []

for molecule in molecule_names:
#    try:
    temp_error,temp_y_predict,temp_y = process_one_molecule(molecule, model_dict, functional,h,N,log_filename, target,gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens)
    error_list.append(temp_error)
#    except:
#        log(log_filename,"\n\n Failed") 
        
log(log_filename,"\n\naverage error: " + str(np.mean(error_list)) + "\tstddev error: " + str(np.std(error_list))) 
log(log_filename,"\n\naverage abs error: " + str(np.mean(np.abs(error_list))) + "\tstddev abs error: " + str(np.std(np.abs(error_list)))) 



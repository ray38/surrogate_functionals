# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 17:34:20 2017

@author: ray
"""

import h5py
import os
import sys
import numpy as np
try: import cPickle as pickle
except: import pickle
import math
import time
import os
import json

import itertools
import multiprocessing
import pandas as pd


def log(log_filename, text):
    with open(log_filename, "a") as myfile:
        myfile.write(text)
    return



def process_each_block(molecule,functional,i,j,k, setup, data_dir, order_list, r_list):
    data_filename = "{}_{}_{}_{}_{}_all_descriptors.hdf5".format(molecule,functional,i,j,k)

    

    print data_filename
    data =  h5py.File(data_filename,'r')
    


    original_result_array = np.zeros((len(order_list), len(r_list)))

    normalized_result_array = np.zeros((len(order_list), len(r_list)))


    group_name = 'MCSH'

    for j in range(len(r_list)):

        r = r_list[j]
        print "start {}".format(r)
        dataset_name = 'MCSH_1_1_{}'.format(str(r_list[j]).replace('.','-'))
        temp_data = np.asarray(data[group_name][dataset_name])
        temp_sum = np.sum(temp_data)
        temp_normalized = temp_sum / ((4./3.)*math.pi*r*r*r)
        original_result_array[0][j] = temp_sum
        normalized_result_array[0][j] = temp_normalized

        dataset_name = 'MCSH_2_1_{}'.format(str(r_list[j]).replace('.','-'))
        temp_data = np.asarray(data[group_name][dataset_name])
        temp_sum = np.sum(temp_data)
        temp_normalized = temp_sum / ((4./3.)*math.pi*r*r*r)
        original_result_array[1][j] = temp_sum
        normalized_result_array[1][j] = temp_normalized

        dataset_name = 'MCSH_3_1_{}'.format(str(r_list[j]).replace('.','-'))
        temp_data = np.asarray(data[group_name][dataset_name])
        temp_sum = np.sum(temp_data)
        temp_normalized = temp_sum / ((4./3.)*math.pi*r*r*r)
        original_result_array[2][j] = temp_sum
        normalized_result_array[2][j] = temp_normalized

        dataset_name = 'MCSH_3_2_{}'.format(str(r_list[j]).replace('.','-'))
        temp_data = np.asarray(data[group_name][dataset_name])
        temp_sum = np.sum(temp_data)
        temp_normalized = temp_sum / ((4./3.)*math.pi*r*r*r)
        original_result_array[3][j] = temp_sum
        normalized_result_array[3][j] = temp_normalized


    return original_result_array, normalized_result_array




def process_one_molecule(molecule, functional,h,L,N, setup, order_list, r_list):

    data_dir = "{}_{}_{}_{}_{}".format(molecule,functional,str(L).replace('.','-'),str(h).replace('.','-'),N)

    
    if os.path.isdir(data_dir) == False:
        print '\n****Error: Cant find the data directory! ****\n'
        raise NotImplementedError
    os.chdir(data_dir)
    

    original_result_array = np.zeros((len(order_list), len(r_list)))

    normalized_result_array = np.zeros((len(order_list), len(r_list)))



    #Nx = Ny = Nz = N
    Nx = Ny = Nz = 1
    
    i_li = range(Nx)
    j_li = range(Ny)
    k_li = range(Nz)
    
    paramlist = list(itertools.product(i_li,j_li,k_li))
    

    for i,j,k in paramlist:
        print "{} {} {} {}".format(molecule, i, j, k)
        temp_original_result_array, temp_normalized_result_array = process_each_block(molecule,functional,i,j,k, setup, data_dir, order_list, r_list)
        original_result_array = np.add(original_result_array, temp_original_result_array)
        normalized_result_array = np.add(normalized_result_array, temp_normalized_result_array)


    os.chdir(setup["cwd"])

    print original_result_array, normalized_result_array
    return original_result_array, normalized_result_array


def prepare_df(result_dict, order_list, r_list):

    molecule_list = []
    r_list = []
    original_0_1_list = []
    original_1_1_list = []
    original_2_1_list = []
    original_2_2_list = []

    normalized_0_1_list = []
    normalized_1_1_list = []
    normalized_2_1_list = []
    normalized_2_2_list = []

    print result_dict.keys()
    print result_dict

    for key in result_dict.keys():
        print key
        print "start"
        temp_original = result_dict[key]["original"]
        temp_normalized = result_dict[key]["normalized"]
        #for i in range(len(order_list)):
        for j in range(len(r_list)):
            print j, temp_original[0][j], temp_original[1][j], temp_original[2][j], temp_original[3][j]
            molecule_list.append(key)

            r_list.append(r_list[j])
            original_0_1_list.append(temp_original[0][j])
            original_1_1_list.append(temp_original[1][j])
            original_2_1_list.append(temp_original[2][j])
            original_2_2_list.append(temp_original[3][j])

            normalized_0_1_list.append(temp_normalized[0][j])
            normalized_1_1_list.append(temp_normalized[1][j])
            normalized_2_1_list.append(temp_normalized[2][j])
            normalized_2_2_list.append(temp_normalized[3][j])


    print molecule_list
    d = {"molecule": molecule_list, "r":r_list, \
         "original_0_1": original_0_1_list, "original_1_1": original_1_1_list, "original_2_1": original_2_1_list, "original_2_2": original_2_2_list,\
         "normalized_0_1": normalized_0_1_list, "normalized_1_1": normalized_1_1_list, "normalized_2_1": normalized_2_1_list, "normalized_2_2": normalized_2_2_list}

    return pd.DataFrame(data=d)


if __name__ == "__main__":

    setup_filename = sys.argv[1]


    with open(setup_filename) as f:
        setup = json.load(f)

    h = float(setup['grid_spacing'])
    L = float(setup['box_dimension'])
    N = int(setup['number_segment_per_side'])
    functional = setup['functional']

    setup["cwd"] = os.getcwd() 

    molecules = ["glycine"]

    order_list = ["0 1","1 1","2 1","2 2"]

    r_list = [0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2,0.22,0.24,0.26,0.28,0.3,0.32,0.34,0.36,0.38,0.4]

    result = {}
    for molecule in molecules:

        temp_original_result_array, temp_normalized_result_array = process_one_molecule(molecule, functional,h,L,N, setup, order_list, r_list)
        print temp_original_result_array, temp_normalized_result_array
        result[molecule] = {}
        result[molecule]["original"] = temp_original_result_array
        result[molecule]["normalized"] =  temp_normalized_result_array

    
    data = prepare_df(result, order_list, r_list)

    with open('mcsh_variation_glycine.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
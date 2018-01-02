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
from subsampling import subsampling_system_with_PCA, random_subsampling, subsampling_system
import math
import time
import os
import json

import itertools
import multiprocessing


def log(log_filename, text):
    with open(log_filename, "a") as myfile:
        myfile.write(text)
    return



def write_data_to_file2(i,j,k,data):
    temp = "{}_{}_{}".format(i,j,k)
    with open(temp + "_subsampled_data.p", 'wb') as handle:
        pickle.dump(data, handle, protocol=2)

    return

def write_data_to_file_random(i,j,k,random_data):
    temp = "{}_{}_{}".format(i,j,k)
    
    with open(temp + "_random_data.p", 'wb') as handle:
        pickle.dump(random_data, handle, protocol=2)

    return



def process_data(i,j,k,log_filename,setup):
    random_pick_number = int(math.ceil((float(len(processed_data)) * float(setup["random_pick_rate"]))))
    temp_random_data = random_subsampling(processed_data, random_pick_number)
    write_data_to_file_random(i,j,k,temp_random_data)
    
    log(log_filename,"\nstart sub-sampling") 
    sample_start = time.time() 
    log(log_filename,"\nmolecule length before: " + str(len(processed_data)))
    if len(list_subsample) <= 10:
        processed_data = subsampling_system(processed_data, list_desc = list_subsample, cutoff_sig = float(setup["subsample_cutoff_sig"]), rate = float(setup["subsample_rate"]))
    else:
        processed_data = subsampling_system_with_PCA(processed_data, list_desc = list_subsample, cutoff_sig = float(setup["subsample_cutoff_sig"]), rate = float(setup["subsample_rate"]),start_trial_component = 9)
    log(log_filename,"\nmolecule length after: " + str(len(processed_data)))  
    log(log_filename,"\nfinished sampling, took: " + str(time.time()-sample_start))
    write_data_to_file2(i,j,k,processed_data)
    
    return
    
    
def subsample_one_molecule(data,i,j,k, setup):
    log_filename = "{}_{}_{}_subsample_log.log".format(i,j,k)
    with open(log_filename, "w") as myfile:
        myfile.write('')
    log(log_filename,"\nstart reading: " + molecule) 
    print "start reading: " + molecule
    process_data(i,j,k,log_filename,setup)
    return




def transform_data(temp_data, transform):
    if transform == "real":
        return temp_data.flatten().tolist()
    elif transform == "log10":
        return np.log10(temp_data.flatten()).tolist()
    elif transform == "neglog10":
        return np.log10(np.multiply(-1., temp_data.flatten())).tolist()

def process_each_block(molecule,functional,i,j,k, setup, data_dir_full):
    data_filename = "{}_{}_{}_{}_{}_all_descriptors.hdf5".format(molecule,functional,i,j,k)
    print data_filename
    data =  h5py.File(data_filename,'r')
    

    result_list = []
  
    
    if target == 'Vxc':
        target_set_name = 'V_xc'   
    if target == 'epxc':
        target_set_name = 'epsilon_xc'
    if target == 'tau':
        target_set_name = 'tau'
    
    temp_data = np.asarray(data[target_set_name])
#    print np.sum(np.sort(temp_data))*0.1*0.1*0.1
    result_list.append(transform_data(temp_data, setup['target_transform']))
    
    if int(setup['density']) == 1:
        temp_data = np.asarray(data['rho'])
        result_list.append(transform_data(temp_data, setup['density_transform']))

    if int(setup['gamma']) == 1:
        temp_data = np.asarray(data['gamma'])
        result_list.append(transform_data(temp_data, setup['gamma_transform']))

    if int(setup['tau']) == 1:
        temp_data = np.asarray(data['tau'])
        result_list.append(transform_data(temp_data, setup['tau_transform']))


    group_name = 'derivative'
    for derivative_count in setup["derivative_list"]:
        dataset_name = 'derivative_{}'.format(derivative_count)
        temp_data = np.asarray(data[group_name][dataset_name])
        result_list.append(transform_data(temp_data, setup['derivative_transform']))

    for derivative_count in setup["derivative_square_list"]:
        dataset_name = 'derivative_{}'.format(derivative_count)
        temp_data = np.power(np.asarray(data[group_name][dataset_name]), 2.)
        result_list.append(transform_data(temp_data, setup['derivative_square_transform']))
    
   
    group_name = 'average_density'
    for r_list_count in data["average_density_r_list"]:
        dataset_name = 'average_density_{}'.format(r_list_count.replace('.','-'))
        temp_data = np.asarray(data[group_name][dataset_name])
        result_list.append(transform_data(temp_data, setup['average_density_transform']))


    group_name = 'asym_integral'
    for r_list_count in data["asym_desc_r_list"]:
        dataset_name = 'asym_integral_x_{}'.format(r_list_count.replace('.','-'))
        temp_data = np.asarray(data[group_name][dataset_name])
        result_list.append(transform_data(temp_data, setup['asym_desc_transform']))

        dataset_name = 'asym_integral_y_{}'.format(r_list_count.replace('.','-'))
        temp_data = np.asarray(data[group_name][dataset_name])
        result_list.append(transform_data(temp_data, setup['asym_desc_transform']))

        dataset_name = 'asym_integral_z_{}'.format(r_list_count.replace('.','-'))
        temp_data = np.asarray(data[group_name][dataset_name])
        result_list.append(transform_data(temp_data, setup['asym_desc_transform']))

    
    result = zip(*result_list)
    print "done picking: {} {} {}".format(i,j,k)
    

    molecule_dir = setup["working_dir"] + '/' + molecule
    if os.path.isdir(molecule_dir) == False:
        os.makedirs(molecule_dir)
    os.chdir(molecule_dir)
    
    
    subsample_one_molecule(result,i,j,k,setup)

    
    os.chdir(data_dir_full)
    return


        
        
        
def process_one_molecule(molecule, functional,h,L,N, setup):
    database_name = setup["database_directory"]
    sub_database_name = "{}_{}_{}".format(str(L).replace('.','-'),str(h).replace('.','-'),N)
    data_dir_name = "{}_{}_{}_{}_{}".format(molecule,functional,str(L).replace('.','-'),str(h).replace('.','-'),N)

    data_dir_full = database_name + '/' + sub_database_name + '/' + dir_name
    print data_dir_full
    
    if os.path.isdir(data_dir_full) == False:
        print '\n****Error: Cant find the database directory! ****\n'
        raise NotImplementedError
    
    os.chdir(data_dir_full)
    
    Nx = Ny = Nz = N
#    for i in range(Nx):
#        for j in range(Ny):
#            pool = multiprocessing.Pool() #use all available cores, otherwise specify the number you want as an argument
#            for k in range(Nz):
#                pool.apply_async(process_each_block, args=(molecule,functional,k,k,k, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform))
#            pool.close()
#            pool.join()
    
    i_li = range(Nx)
    j_li = range(Ny)
    k_li = range(Nz)
    
    paramlist = list(itertools.product(i_li,j_li,k_li))
    
    pool = multiprocessing.Pool()
    for i,j,k in paramlist:
        pool.apply_async(process_each_block, args=(molecule,functional,i,j,k, setup, data_dir_full))
    pool.close()
    pool.join()

#    process_each_block(molecule,functional,0,0,0, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform)

    return


if __name__ == "__main__":

    setup_database_filename = sys.argv[1]
    setup_name = sys.argv[2]
    choice = sys.argv[3]

    if choice not in ['single','set']:
        raise NotImplementedError

    with open(setup_database_filename) as f:
        setup_database = json.load(f)

    setup = setup_database[setup_name]
    h = float(setup['grid_spacing'])
    L = float(setup['box_dimension'])
    N = int(setup['number_segment_per_side'])
    dir_name = "{}_{}_{}".format(str(L).replace('.','-'),str(h).replace('.','-'),N)


    working_dir = os.getcwd() + '/' + dir_name + '/' + setup_name

    if os.path.isdir(working_dir) == False:
        os.makedirs(working_dir)    

    setup["working_dir"] = working_dir
    
    if choice == 'single':
        molecule = sys.argv[4]
        
        functional = setup['functional']
        

        process_one_molecule(molecule, functional,h,L,N, setup)
        
        
        
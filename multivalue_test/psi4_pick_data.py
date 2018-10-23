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



def write_data_to_file(i,j,k,data):
    temp = "{}_{}_{}".format(i,j,k)
    
    with open(temp + "_all_data.p", 'wb') as handle:
        pickle.dump(data, handle, protocol=2)

    return

    
    
def subsample_one_molecule(data,i,j,k, setup):
    write_data_to_file(i,j,k,data)
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
    molecule_dir = setup["working_dir"] + '/data/' + molecule

    try:
        os.chdir(molecule_dir)
        temp_log_filename = "{}_{}_{}_subsample_log.log".format(i,j,k)
        temp_random_data_filename = "{}_{}_{}_random_data.p".format(i,j,k)
        temp_subsampled_data_filename = "{}_{}_{}_subsampled_data.p".format(i,j,k)
        if os.path.isfile(temp_log_filename) and os.path.isfile(temp_random_data_filename) and os.path.isfile(temp_subsampled_data_filename):
            os.chdir(data_dir_full)
            return
        else:
            raise SystemError

    except:
        os.chdir(data_dir_full)

        print data_filename
        data =  h5py.File(data_filename,'r')
        

        result_list = []
        
        target = setup['target']
        
        if target == 'Vxc':
            target_set_name = 'V_xc'   
        if target == 'epxc':
            target_set_name = 'epsilon_xc'
        if target == 'tau':
            target_set_name = 'tau'
        if target == 'gamma':
            target_set_name = 'gamma'
        
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
        temp_list = setup["derivative_list"]
        if len(temp_list) > 0: 
            for derivative_count in temp_list:
                dataset_name = 'derivative_{}'.format(derivative_count)
                temp_data = np.asarray(data[group_name][dataset_name])
                result_list.append(transform_data(temp_data, setup['derivative_transform']))

        temp_list = setup["derivative_square_list"]
        if len(temp_list) > 0: 
            for derivative_count in temp_list:
                dataset_name = 'derivative_{}'.format(derivative_count)
                temp_data = np.power(np.asarray(data[group_name][dataset_name]), 2.)
                result_list.append(transform_data(temp_data, setup['derivative_square_transform']))
        
       
        group_name = 'average_density'
        temp_list = setup["average_density_r_list"]
        if len(temp_list) > 0:
            for r_list_count in temp_list:
                dataset_name = 'average_density_{}'.format(str(r_list_count).replace('.','-'))
                temp_data = np.asarray(data[group_name][dataset_name])
                result_list.append(transform_data(temp_data, setup['average_density_transform']))


        group_name = 'asym_integral'
        try:
            temp_list = setup["asym_desc_r_list"]
            if len(temp_list) > 0:
                for r_list_count in temp_list:
                    dataset_name = 'asym_integral_x_{}'.format(str(r_list_count).replace('.','-'))
                    temp_data = np.asarray(data[group_name][dataset_name])
                    result_list.append(transform_data(temp_data, setup['asym_desc_transform']))

                    dataset_name = 'asym_integral_y_{}'.format(str(r_list_count).replace('.','-'))
                    temp_data = np.asarray(data[group_name][dataset_name])
                    result_list.append(transform_data(temp_data, setup['asym_desc_transform']))

                    dataset_name = 'asym_integral_z_{}'.format(str(r_list_count).replace('.','-'))
                    temp_data = np.asarray(data[group_name][dataset_name])
                    result_list.append(transform_data(temp_data, setup['asym_desc_transform']))
        except:
            pass



        group_name = 'asym_integral'
        try:
            temp_list = setup["asymsum_desc_r_list"]
            if len(temp_list) > 0:
                for r_list_count in temp_list:
                    dataset_name = 'asym_integral_x_{}'.format(str(r_list_count).replace('.','-'))
                    temp_data1 = np.asarray(data[group_name][dataset_name])

                    dataset_name = 'asym_integral_y_{}'.format(str(r_list_count).replace('.','-'))
                    temp_data2 = np.asarray(data[group_name][dataset_name])

                    dataset_name = 'asym_integral_z_{}'.format(str(r_list_count).replace('.','-'))
                    temp_data3 = np.asarray(data[group_name][dataset_name])
                    result_list.append(transform_data(temp_data1 + temp_data2 + temp_data3 , setup['asymsum_desc_transform']))
        except:
            pass



        group_name = 'MCSH'
        try:
            temp_list = setup["MCSH_1_1_r_list"]
            if len(temp_list) > 0:
                for r_list_count in temp_list:
                    dataset_name = 'MCSH_1_1_{}'.format(str(r_list_count).replace('.','-'))
                    temp_data = np.asarray(data[group_name][dataset_name])
                    result_list.append(transform_data(temp_data, setup['MCSH_1_1_transform']))
        except:
            pass


        try:
            temp_list = setup["MCSH_2_1_r_list"]
            if len(temp_list) > 0:
                for r_list_count in temp_list:
                    dataset_name = 'MCSH_2_1_{}'.format(str(r_list_count).replace('.','-'))
                    temp_data = np.asarray(data[group_name][dataset_name])
                    result_list.append(transform_data(temp_data, setup['MCSH_2_1_transform']))
        except:
            pass

        try:
            temp_list = setup["MCSH_3_1_r_list"]
            if len(temp_list) > 0:
                for r_list_count in temp_list:
                    dataset_name = 'MCSH_3_1_{}'.format(str(r_list_count).replace('.','-'))
                    temp_data = np.asarray(data[group_name][dataset_name])
                    result_list.append(transform_data(temp_data, setup['MCSH_3_1_transform']))
        except:
            pass

        try:
            temp_list = setup["MCSH_3_2_r_list"]
            if len(temp_list) > 0:
                for r_list_count in temp_list:
                    dataset_name = 'MCSH_3_2_{}'.format(str(r_list_count).replace('.','-'))
                    temp_data = np.asarray(data[group_name][dataset_name])
                    result_list.append(transform_data(temp_data, setup['MCSH_3_2_transform']))
        except:
            pass



        
        result = zip(*result_list)
        print "done picking: {} {} {}".format(i,j,k)
        

    #    molecule_dir = setup["working_dir"] + '/data/' + molecule
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

    data_dir_full = database_name + '/' + sub_database_name + '/' + data_dir_name
    print data_dir_full
    
    if os.path.isdir(data_dir_full) == False:
        print '\n****Error: Cant find the database directory! ****\n'
        raise NotImplementedError
    
    os.chdir(data_dir_full)
    
    Nx = Ny = Nz = N
    
    i_li = range(Nx)
    j_li = range(Ny)
    k_li = range(Nz)
    
    paramlist = list(itertools.product(i_li,j_li,k_li))
    
#    pool = multiprocessing.Pool()
#    for i,j,k in paramlist:
#        pool.apply_async(process_each_block, args=(molecule,functional,i,j,k, setup, data_dir_full))
#    pool.close()
#    pool.join()

    process_each_block(molecule,functional,2,3,2, setup, data_dir_full)
    process_each_block(molecule,functional,2,1,3, setup, data_dir_full)

#    for i,j,k in paramlist:
#        process_each_block(molecule,functional,i,j,k, setup, data_dir_full)


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
        #test
        
        
        
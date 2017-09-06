# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 17:20:15 2017

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

import itertools
import multiprocessing


def log(log_filename, text):
    with open(log_filename, "a") as myfile:
        myfile.write(text)
    return



def write_data_to_file2(i,j,k,data,random_data, molecule,functional, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform):
    temp = "{}_{}_{}_{}_{}_{}_gamma{}_dev{}_devsq{}_inte{}_{}_{}".format(molecule,functional,i,j,k, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform)
    with open(temp + "_subsampled_data.p", 'wb') as handle:
        pickle.dump(data, handle, protocol=2)

    with open(temp + "_subsampled_random_data.p", 'wb') as handle:
        pickle.dump(data + random_data, handle, protocol=2)
    return

def write_data_to_file_random(i,j,k,random_data, molecule,functional, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform):
    temp = "{}_{}_{}_{}_{}_{}_gamma{}_dev{}_devsq{}_inte{}_{}_{}".format(molecule,functional,i,j,k, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform)
    
    with open(temp + "_random_data.p", 'wb') as handle:
        pickle.dump(random_data, handle, protocol=2)

    return



def process_data(i,j,k,random_rate, processed_data,log_filename,list_subsample,molecule,functional, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform):
    random_pick_number = int(math.ceil((float(len(processed_data)) * random_rate)))
    temp_random_data = random_subsampling(processed_data, random_pick_number)
    write_data_to_file_random(i,j,k,temp_random_data, molecule,functional, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform)
    
    log(log_filename,"\nstart sub-sampling") 
    sample_start = time.time() 
    log(log_filename,"\nmolecule length before: " + str(len(processed_data)))
    if len(list_subsample) <= 10:
        processed_data = subsampling_system(processed_data, list_desc = list_subsample, cutoff_sig = 0.01, rate = 0.2)
    else:
        processed_data = subsampling_system_with_PCA(processed_data, list_desc = list_subsample, cutoff_sig = 0.01, rate = 0.2,start_trial_component = 9)
    log(log_filename,"\nmolecule length after: " + str(len(processed_data)))  
    log(log_filename,"\nfinished sampling, took: " + str(time.time()-sample_start))
    write_data_to_file2(i,j,k,processed_data, temp_random_data, molecule,functional, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform)
    
    return
    
    
def subsample_one_molecule(data,i,j,k,molecule,functional, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform):
    log_filename = "{}_{}_{}_{}_{}_{}_gamma{}_dev{}_devsq{}_inte{}_{}_{}_subsample_log.log".format(molecule,functional,i,j,k, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform)
    with open(log_filename, "w") as myfile:
        myfile.write('')
    log(log_filename,"\nstart reading: " + molecule) 
    print "start reading: " + molecule
    process_data(i,j,k,0.05, data,log_filename,[], molecule,functional, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform)
    return




def process_each_block(molecule,functional,i,j,k, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform):
    data_filename = "{}_{}_{}_{}_{}_all_descriptors.hdf5".format(molecule,functional,i,j,k)
    print data_filename
    data =  h5py.File(data_filename,'r')
    
    r_list = [0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.1,0.15,0.2,0.3,0.4,0.5]
    
    result_list = []
  
    
    if target == 'Vxc':
        target_set_name = 'V_xc'   
    if target == 'epxc':
        target_set_name = 'epsilon_xc'
    if target == 'tau':
        target_set_name = 'tau'
    
    temp_data = np.asarray(data[target_set_name])
#    print np.sum(np.sort(temp_data))*0.1*0.1*0.1
    
    
    if target_transform == 'real':
        result_list.append(temp_data.flatten().tolist())
    elif target_transform == 'log':
        result_list.append(np.log10(np.multiply(-1., temp_data.flatten())).tolist())

    
    temp_data = np.asarray(data['rho'])

    if desc_transform == 'real':
        result_list.append(temp_data.flatten().tolist())
    elif desc_transform == 'log':
        result_list.append(np.log10(temp_data.flatten()).tolist())
    
    if gamma == 1:
        temp_data = np.asarray(data['gamma'])
#        print 'gamma'
#        print temp_data.shape
        if desc_transform == 'real':
                result_list.append(temp_data.flatten().tolist())
        elif desc_transform == 'log':
            result_list.append(np.log10(temp_data.flatten()).tolist())
    
    if num_desc_deri > 0:
        group_name = 'derivative'
        for desc_deri_count in range(num_desc_deri):
            dataset_name = 'derivative_{}'.format(desc_deri_count+1)
            temp_data = np.asarray(data[group_name][dataset_name])
#            print 'deriv'
#            print temp_data.shape
            temp = temp_data.flatten()
            result_list.append(temp.tolist())
    
    if num_desc_deri_squa > 0:
        group_name = 'derivative'
        for desc_deri_count in range(num_desc_deri_squa):
            print str(i) + ' start' 
            dataset_name = 'derivative_{}'.format(desc_deri_count+1)
            temp_data = np.asarray(data[group_name][dataset_name])
#            print 'deriv squared'
#            print temp_data.shape
            temp = np.power(temp_data,2.)
            result_list.append(temp.flatten().tolist())
   
    if num_desc_ave_dens > 0:
        group_name = 'average_density'
        for desc_ave_dens_count in range(num_desc_ave_dens):
            
            dataset_name = 'average_density_{}'.format(str(r_list[desc_ave_dens_count]).replace('.','-'))
            temp_data = np.asarray(data[group_name][dataset_name])
#            print 'ave_dens'
#            print temp_data.shape
            if desc_transform == 'real':
                result_list.append(temp_data.flatten().tolist())
            elif desc_transform == 'log':
                result_list.append(np.log10(temp_data.flatten()).tolist())
        
    
    
    result = zip(*result_list)
    
    parent = os.getcwd()
    dir_name = "{}_{}_gamma{}_dev{}_devsq{}_inte{}_{}_{}".format(functional, target, gamma,num_desc_deri,num_desc_deri_squa,num_desc_ave_dens,desc_transform,target_transform)    
    if os.path.isdir(dir_name) == False:
        os.makedirs(parent + '/' + dir_name)      
    os.chdir(parent + '/' + dir_name)
    
    print "done picking: {} {} {}".format(i,j,k)
    subsample_one_molecule(result,i,j,k,molecule,functional, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform)

    
    os.chdir(parent)
    return


        
        
        
def process_one_molecule(molecule, functional,h,L,N, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform):
    cwd = os.getcwd()
    dir_name = "{}_{}_{}_{}_{}".format(molecule,functional,str(L).replace('.','-'),str(h).replace('.','-'),N)
    print dir_name
    
    if os.path.isdir(dir_name) == False:
        print '\n****Error: Cant find the directory! ****\n'
        raise NotImplementedError
    
    os.chdir(cwd + '/' + dir_name)
    
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
        pool.apply_async(process_each_block, args=(molecule,functional,i,j,k, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform))
    pool.close()
    pool.join()


    return


if __name__ == "__main__":
    choice = sys.argv[1]
    if choice not in ['single','set']:
        raise NotImplementedError
    
    if choice == 'single':
        molecule = sys.argv[2]
        h = float(sys.argv[3])
        L = float(sys.argv[4])
        N = int(sys.argv[5])
        
        functional = sys.argv[6]
        gamma = int(sys.argv[7])
        num_desc_deri = int(sys.argv[8])
        num_desc_deri_squa = int(sys.argv[9])
        num_desc_ave_dens = int(sys.argv[10])
        target = sys.argv[11]
        desc_transform = sys.argv[12]
        target_transform = sys.argv[13]
        
        if target not in ['Vxc','epxc','tau']:
            raise ValueError
        if gamma > 1:
            raise ValueError
        if num_desc_ave_dens > 80:
            raise ValueError
        if num_desc_deri_squa > 3:
            raise ValueError
        if num_desc_deri > 3:
            raise ValueError
        if desc_transform not in ['log','real']:
            raise ValueError
        if target_transform not in ['log','real']:
            raise ValueError

        process_one_molecule(molecule, functional,h,L,N, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform)
        
        
        
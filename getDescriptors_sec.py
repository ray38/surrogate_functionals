# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 15:53:10 2017

@author: ray
"""

import numpy as np
import sys
import math

from convolutions import get_differenciation_conv, get_integration_stencil,get_auto_accuracy,get_fftconv_with_known_stencil_no_wrap
import h5py
import os
#from joblib import Parallel, delayed
import multiprocessing
import itertools
import json


def read_system(mol,xc,Nx_center,Ny_center,Nz_center,Nx,Ny,Nz,data_name):
    def get_index(center,maxx,minn):
        if center > maxx or center < minn:
            raise NotImplementedError
        temp_first = center - 1
        if temp_first <0:
            temp_first = maxx
        temp_second = center
        temp_third = center +1
        if temp_third >maxx:
            temp_third = minn
        return [temp_first,temp_second,temp_third]
    
    Nx_max = Nx-1
    Ny_max = Ny-1
    Nz_max = Nz-1
    Nx_min = 0
    Ny_min = 0
    Nz_min = 0
    
    Nx_list = get_index(Nx_center,Nx_max,Nx_min)
    Ny_list = get_index(Ny_center,Ny_max,Ny_min)
    Nz_list = get_index(Nz_center,Nz_max,Nz_min)
    
    temp_x = None
    x_start = True
    for i in Nx_list:
        
        temp_y = None
        y_start = True
        for j in Ny_list:
            
            temp_z = None
            z_start = True
            for k in Nz_list:
                
                temp_filename =  '{}_{}_{}_{}_{}.hdf5'.format(mol,xc,i,j,k)            
                if os.path.isfile(temp_filename):
                    raw_data =  h5py.File(temp_filename,'r')
                    temp_data = np.asarray(raw_data[data_name])
                if z_start:
                    temp_z = temp_data
                    z_start = False
                else:
                    temp_z = np.concatenate((temp_z,temp_data),axis=2)
            
            if y_start:
                temp_y = temp_z
                y_start = False
            else:
                temp_y = np.concatenate((temp_y,temp_z),axis=1)
       
        if x_start:
            temp_x = temp_y
            x_start = False
        else:
            temp_x = np.concatenate((temp_x,temp_y),axis=0)
    
    return temp_x
    
def carve_out_matrix(matrix):
    old_shape_x, old_shape_y, old_shape_z = matrix.shape
    x = int(round(old_shape_x / 3. ,0))
    y = int(round(old_shape_y / 3. ,0))
    z = int(round(old_shape_z / 3. ,0))
   
    return matrix[x:2*x,y:2*y,z:2*z]

def get_homogeneous_gas_integral(n,r):
    return r*r*n*math.pi

def get_homo_nondimensional(int_arr, n_arr, r):
    temp = (4./3.)*r*r*r*math.pi
    result = np.divide(int_arr, n_arr)
    return np.divide(result, temp)

def get_homo_nondimensional_nave(int_arr, n_ave, r):
    temp = (4./3.)*r*r*r*math.pi*n_ave
    return np.divide(int_arr, temp)


def calculate_ave_density_desc(n,r,hx,hy,hz,stencil,pad):
    print 'start ave dens desc : ' + str(r) 
#    integration, temp_pad  = get_integration_fftconv(n, hx, hy, hz, r, accuracy = get_auto_accuracy(hx,hy,hz, r))
    integration, temp_pad = get_fftconv_with_known_stencil_no_wrap(n,hx,hy,hz,r,stencil,pad)
    ave_density = get_homo_nondimensional_nave(integration, 1.0, r)
    return ave_density, temp_pad


def process_normal_descriptors(molecule, functional,i,j,k):
    
    raw_data_filename = "{}_{}_{}_{}_{}.hdf5".format(molecule,functional,i,j,k)
    result_filename = "{}_{}_{}_{}_{}_all_descriptors.hdf5".format(molecule,functional,i,j,k)
    raw_data =  h5py.File(raw_data_filename,'r')
    n = np.asarray(raw_data['rho'])
    V_xc = np.asarray(raw_data['V_xc'])
    ep_xc =  np.asarray(raw_data['epsilon_xc'])
    gamma =  np.asarray(raw_data['gamma'])
    tau =  np.asarray(raw_data['tau'])
    raw_data.close()


    with h5py.File(result_filename,'a') as data:
        data.create_dataset('V_xc',data=V_xc)
        data.create_dataset('epsilon_xc',data=ep_xc)
        data.create_dataset('rho',data=n)
        data.create_dataset('gamma',data=gamma)
        data.create_dataset('tau',data=tau)
        
    return

def process_range_descriptor(molecule, functional,i,j,k,h,N,r_list,stencil_list,pad_list):
    result_filename = "{}_{}_{}_{}_{}_all_descriptors.hdf5".format(molecule,functional,i,j,k)
    Nx = Ny = Nz = N
    extented_n = read_system(molecule,functional,i,j,k,Nx,Ny,Nz,'rho')


    with h5py.File(result_filename,'a') as data:
#        temp_first_deri = np.gradient(extented_n.copy())
        ave_dens_grp = data.create_group('average_density')
        derivative_grp = data.create_group('derivative')
        for index, r in enumerate(r_list):
            dataset_name = 'average_density_{}'.format(str(r).replace('.','-'))
            if dataset_name not in ave_dens_grp.keys():
                temp_data, temp_pad = calculate_ave_density_desc(extented_n.copy(),r,h,h,h,stencil_list[index],pad_list[index])
                ave_dens_grp.create_dataset(dataset_name,data=carve_out_matrix(temp_data))

                
        temp_first_deri, temp_pad = get_differenciation_conv(extented_n.copy(), h, h, h, gradient = 'first',
                                               stencil_type = 'mid', accuracy = '2')
        temp_sec_deri, temp_pad   = get_differenciation_conv(extented_n.copy(), h, h, h, gradient = 'second',
                                               stencil_type = 'times2', accuracy = '2')
#        temp_third_deri, temp_pad = get_differenciation_conv(extented_n.copy(), h, h, h, gradient = 'third',
#                                               stencil_type = 'times2', accuracy = '2')
        derivative_grp.create_dataset('derivative_1',data=carve_out_matrix(temp_first_deri))
        derivative_grp.create_dataset('derivative_2',data=carve_out_matrix(temp_sec_deri))
#        derivative_grp.create_dataset('derivative_3',data=carve_out_matrix(temp_third_deri))
        print data.keys()
        print derivative_grp.keys()
        print ave_dens_grp.keys()

        
    return


def prepare_integral_stencils(r_list,h):
    print 'start preparing integral stencils'
    stencil_list = []
    pad_list = []
    for r in r_list:
        temp_stencil,temp_pad = get_integration_stencil(h, h, h, r, accuracy = get_auto_accuracy(h,h,h, r))
        stencil_list.append(temp_stencil)
        pad_list.append(temp_pad)
    return stencil_list, pad_list

def process(molecule, functional,i,j,k,h,N,r_list,stencil_list,pad_list):
    result_filename = "{}_{}_{}_{}_{}_all_descriptors.hdf5".format(molecule,functional,i,j,k)
    if os.path.isfile(result_filename) == False:
        print 'start {} {} {}'.format(i,j,k)
        process_normal_descriptors(molecule, functional,i,j,k)
        process_range_descriptor(molecule, functional,i,j,k,h,N,r_list,stencil_list,pad_list)
    

def process_one_molecule(molecule, functional,h,L,N):
    cwd = os.getcwd()
    dir_name = "{}_{}_{}_{}_{}".format(molecule,functional,str(L).replace('.','-'),str(h).replace('.','-'),N)
    print dir_name
    
    if os.path.isdir(dir_name) == False:
        print '\n****Error: Cant find the directory! ****\n'
        raise NotImplementedError
    
    os.chdir(cwd + '/' + dir_name)
#    r_list = np.linspace(0.05, 0.1, 2)
#    r_list = [0.01]
    r_list = [0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.1,0.15,0.2,0.3,0.4,0.5]
    stencil_list,pad_list = prepare_integral_stencils(r_list,h)
    
    num_cores = multiprocessing.cpu_count()
    print "number of cores: {}".format(num_cores)

#    Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)
    
    Nx = Ny = Nz = N
    i_li = range(Nx)
    j_li = range(Ny)
    k_li = range(Nz)
    
    paramlist = list(itertools.product(i_li,j_li,k_li))
    
    pool = multiprocessing.Pool()
    for i,j,k in paramlist:
        pool.apply_async(process, args=(molecule, functional,i,j,k,h,N,r_list,stencil_list,pad_list))
    pool.close()
    pool.join()
        
    
    
    os.chdir(cwd)
    return


if __name__ == "__main__":
    choice = sys.argv[1]
    if choice not in ['single','set']:
        raise NotImplementedError
    
    if choice == 'single':
        molecule = sys.argv[2]
        functional = sys.argv[3]
        h = float(sys.argv[4])
        L = float(sys.argv[5])
        N = int(sys.argv[6])
        functionals = ['B3LYP']#'PBE','PBE0','SVWN',
        if functional in functionals:
            process_one_molecule(molecule, functional,h,L,N)

    elif choice == 'set':
        list_molecule_filename = sys.argv[2]
        h = float(sys.argv[3])
        L = float(sys.argv[4])
        N = int(sys.argv[5])
        with open(list_molecule_filename) as f:
            molecule_names = f.readlines()
        molecule_names = [x.strip() for x in molecule_names]
        functionals = ['PBE','PBE0','SVWN','B3LYP']
        for functional in functionals:
            for molecule in molecule_names:
                result_filename = "{}_{}_all_descriptors.hdf5".format(molecule,functional)
                try:
                    if os.path.isfile(result_filename) == False:
                        process_one_molecule(molecule, functional,h,L,N)
                    else:
                        print result_filename + ' already exist'
                except:
                    print "failed: {}\t{}".format(molecule,functional)
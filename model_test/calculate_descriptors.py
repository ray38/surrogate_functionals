# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 15:53:10 2017

@author: ray
"""

import numpy as np
import sys
import math

from convolutions import get_integration_stencil,get_auto_accuracy,get_fftconv_with_known_stencil_no_wrap,calc_MC_surface_harmonic_stencil
import h5py
import os
#from joblib import Parallel, delayed
#import multiprocessing
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
    print Nx_list, Ny_list,Nz_list
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

def create_dataset(database, dataset_name, data):
    if dataset_name not in database.keys():
        database.create_dataset(dataset_name,data=data)
    return


def process_normal_descriptors(molecule, functional,i,j,k):
    
    raw_data_filename = "{}_{}_{}_{}_{}.hdf5".format(molecule,functional,i,j,k)
    result_filename = "{}_{}_{}_{}_{}_all_descriptors.hdf5".format(molecule,functional,i,j,k)
    raw_data =  h5py.File(raw_data_filename,'r')
    rho = np.asarray(raw_data['rho'])
    epxc = np.asarray(raw_data['epsilon_xc'])
    raw_data.close()

    with h5py.File(result_filename,'a') as database:
        print 'get normal'
        create_dataset(database, 'rho', rho)
        create_dataset(database, 'epsilon_xc', epxc)
        
    return

def process_range_descriptor(molecule, functional,i,j,k,h,N,r_list,stencil_list,pad_list,MC_surface_harmonic_stencil_dict):
    
    result_filename = "{}_{}_{}_{}_{}_all_descriptors.hdf5".format(molecule,functional,i,j,k)
    Nx = Ny = Nz = N
    extented_n = read_system(molecule,functional,i,j,k,Nx,Ny,Nz,'rho')

    with h5py.File(result_filename,'a') as data:
        
        ave_dens_grp = data.create_group('average_density')

        try:
            ave_dens_grp = data['average_density']
        except:
            ave_dens_grp = data.create_group('average_density')

        try:
            MC_surface_spherical_harmonic_grp = data['MC_surface_spherical_harmonic']
        except:
            MC_surface_spherical_harmonic_grp = data.create_group('MC_surface_spherical_harmonic')

 
        for index, r in enumerate(r_list):
            dataset_name = 'average_density_{}'.format(str(r).replace('.','-'))
            if dataset_name not in ave_dens_grp.keys():
                temp_data, temp_pad = calculate_ave_density_desc(extented_n.copy(),r,h,h,h,stencil_list[index],pad_list[index])
                ave_dens_grp.create_dataset(dataset_name,data=carve_out_matrix(temp_data))


        for r in r_list:
            dataset_name = 'MC_surface_shperical_harmonic_1_{}'.format(str(r).replace('.','-'))
            if dataset_name not in MC_surface_spherical_harmonic_grp.keys():
                print "start: {} MC_surface_shperical 1 ".format(r)
                stencils = MC_surface_harmonic_stencil_dict["1"][str(r)][0]
                pad = MC_surface_harmonic_stencil_dict["1"][str(r)][1]

                temp_result = np.zeros_like(carve_out_matrix(extented_n.copy()))

                for temp_stencil in stencils:
                    temp_temp_result_extend,_ = get_fftconv_with_known_stencil_no_wrap(extented_n,h,h,h,1,temp_stencil,0)
                    temp_temp_result = carve_out_matrix(temp_temp_result_extend)
                    temp_result = np.add(temp_result, np.square(temp_temp_result))

                temp_result = np.sqrt(temp_result)

                MC_surface_spherical_harmonic_grp.create_dataset(dataset_name,data=temp_result)
                

        
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


def prepare_MC_surface_harmonic_stencil_stencils(r_list,h):

    MC_surface_harmonic_stencil_dict = {}
    MC_surface_harmonic_stencil_dict["1"] = {}
    MC_surface_harmonic_stencil_dict["2"] = {}
    MC_surface_harmonic_stencil_dict["3"] = {}
    MC_surface_harmonic_stencil_dict["4"] = {}

    for r in r_list:

        stencil_Re_1, pad =  calc_MC_surface_harmonic_stencil(h, h, h, r, 1, 1, accuracy = 6)
        stencil_Re_2, pad =  calc_MC_surface_harmonic_stencil(h, h, h, r, 1, 2, accuracy = 6)
        stencil_Re_3, pad =  calc_MC_surface_harmonic_stencil(h, h, h, r, 1, 3, accuracy = 6)
        MC_surface_harmonic_stencil_dict["1"][str(r)] = [[stencil_Re_1,stencil_Re_2,stencil_Re_3], pad ]

    return MC_surface_harmonic_stencil_dict


def process(molecule, functional,i,j,k,h,N,r_list,stencil_list,pad_list,asym_stencil_list,asym_pad_list):
    result_filename = "{}_{}_{}_{}_{}_all_descriptors.hdf5".format(molecule,functional,i,j,k)
    if os.path.isfile(result_filename) == False:
        print 'start {} {} {}'.format(i,j,k)
        process_normal_descriptors(molecule, functional,i,j,k)
        process_range_descriptor(molecule, functional,i,j,k,h,N,r_list,stencil_list,pad_list,asym_stencil_list,asym_pad_list)
    

def process_one_molecule(molecule, functional,h,L,N,r_list):
    cwd = os.getcwd()
    dir_name = "{}_{}_{}_{}_{}".format(molecule,functional,str(L).replace('.','-'),str(h).replace('.','-'),N)
    print dir_name
    
    if os.path.isdir(dir_name) == False:
        print '\n****Error: Cant find the directory! ****\n'
        raise NotImplementedError
    
    os.chdir(cwd + '/' + dir_name)
    stencil_list,pad_list = prepare_integral_stencils(r_list,h)
    MC_surface_harmonic_stencil_dict = prepare_MC_surface_harmonic_stencil_stencils(r_list,h)

    
    Nx = Ny = Nz = N
    i_li = range(Nx)
    j_li = range(Ny)
    k_li = range(Nz)
    
    paramlist = list(itertools.product(i_li,j_li,k_li))
    


    for i,j,k in paramlist:
        process(molecule, functional,i,j,k,h,N,r_list,stencil_list,pad_list,MC_surface_harmonic_stencil_dict)
        
    
    
    os.chdir(cwd)
    return


if __name__ == "__main__":

    setup_filename = sys.argv[1]
    molecule = sys.argv[2]


    #with open(setup_filename, encoding='utf-8') as f:
    with open(setup_filename) as f:
        setup = json.load(f)

    print setup

    
    print "start"
    h = float(setup['grid_spacing'])
    L = float(setup['box_dimension'])
    N = int(setup['number_segment_per_side'])
    functional = setup['functionals']
    r_list = setup['r_list']

    #for functional in functionals:
    print "start process molecule"
    process_one_molecule(molecule, functional,h,L,N,r_list)

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 15:53:10 2017

@author: ray
"""

import numpy as np
import sys
import math

from convolutions import get_differenciation_conv, get_integration_stencil,get_auto_accuracy,get_fftconv_with_known_stencil_no_wrap,get_asym_integration_stencil,get_asym_integration_fftconv,get_asym_integral_fftconv_with_known_stencil,calc_MC_surface_harmonic_stencil
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



def process_range_descriptor(molecule, functional,i,j,k,h,N,r_list,MCSH_stencil_dict):
    
    result_filename = "{}_{}_{}_{}_{}_all_descriptors.hdf5".format(molecule,functional,i,j,k)
    Nx = Ny = Nz = N
#    print 'get range1'
    extented_n = read_system(molecule,functional,i,j,k,Nx,Ny,Nz,'rho')

#    print 'get range2'
    with h5py.File(result_filename,'a') as data:
        

        MCSH_grp = data['MCSH']
        print "MCSH"


        for r in r_list:


            dataset_name = 'MCSH_4_1_{}'.format(str(r).replace('.','-'))
            if dataset_name not in MCSH_grp.keys():
                print "start: {} MCSH 4 1 ".format(r)
                stencils = MCSH_stencil_dict["4_1"][str(r)][0]
                pad = MCSH_stencil_dict["4_1"][str(r)][1]

                temp_result = np.zeros_like(carve_out_matrix(extented_n.copy()))

                for temp_stencil in stencils:
                    temp_temp_result_extend,_ = get_fftconv_with_known_stencil_no_wrap(extented_n,h,h,h,1,temp_stencil,0)
                    temp_temp_result = carve_out_matrix(temp_temp_result_extend)
                    temp_result = np.add(temp_result, np.square(temp_temp_result))

                temp_result = np.sqrt(temp_result)

                MCSH_grp.create_dataset(dataset_name,data=temp_result)

            dataset_name = 'MCSH_4_2_{}'.format(str(r).replace('.','-'))
            if dataset_name not in MCSH_grp.keys():
                print "start: {} MCSH 4 2 ".format(r)
                stencils = MCSH_stencil_dict["4_2"][str(r)][0]
                pad = MCSH_stencil_dict["4_2"][str(r)][1]

                temp_result = np.zeros_like(carve_out_matrix(extented_n.copy()))

                for temp_stencil in stencils:
                    temp_temp_result_extend,_ = get_fftconv_with_known_stencil_no_wrap(extented_n,h,h,h,1,temp_stencil,0)
                    temp_temp_result = carve_out_matrix(temp_temp_result_extend)
                    temp_result = np.add(temp_result, np.square(temp_temp_result))

                temp_result = np.sqrt(temp_result)

                MCSH_grp.create_dataset(dataset_name,data=temp_result)


            dataset_name = 'MCSH_4_3_{}'.format(str(r).replace('.','-'))
            if dataset_name not in MCSH_grp.keys():
                print "start: {} MCSH 4 3 ".format(r)
                stencils = MCSH_stencil_dict["4_3"][str(r)][0]
                pad = MCSH_stencil_dict["4_3"][str(r)][1]

                temp_result = np.zeros_like(carve_out_matrix(extented_n.copy()))

                for temp_stencil in stencils:
                    temp_temp_result_extend,_ = get_fftconv_with_known_stencil_no_wrap(extented_n,h,h,h,1,temp_stencil,0)
                    temp_temp_result = carve_out_matrix(temp_temp_result_extend)
                    temp_result = np.add(temp_result, np.square(temp_temp_result))

                temp_result = np.sqrt(temp_result)

                MCSH_grp.create_dataset(dataset_name,data=temp_result)


            dataset_name = 'MCSH_4_4_{}'.format(str(r).replace('.','-'))
            if dataset_name not in MCSH_grp.keys():
                print "start: {} MCSH 4 4 ".format(r)
                stencils = MCSH_stencil_dict["4_4"][str(r)][0]
                pad = MCSH_stencil_dict["4_4"][str(r)][1]

                temp_result = np.zeros_like(carve_out_matrix(extented_n.copy()))

                for temp_stencil in stencils:
                    temp_temp_result_extend,_ = get_fftconv_with_known_stencil_no_wrap(extented_n,h,h,h,1,temp_stencil,0)
                    temp_temp_result = carve_out_matrix(temp_temp_result_extend)
                    temp_result = np.add(temp_result, np.square(temp_temp_result))

                temp_result = np.sqrt(temp_result)

                MCSH_grp.create_dataset(dataset_name,data=temp_result)


        
    return






def prepare_MCSH_stencils(r_list,h):

    MCSH_stencil_dict = {}
    MCSH_stencil_dict["1_1"] = {}
    MCSH_stencil_dict["2_1"] = {}
    MCSH_stencil_dict["3_1"] = {}
    MCSH_stencil_dict["3_2"] = {}
    MCSH_stencil_dict["3_3"] = {}

    MCSH_stencil_dict["4_1"] = {}
    MCSH_stencil_dict["4_2"] = {}
    MCSH_stencil_dict["4_3"] = {}
    MCSH_stencil_dict["4_1"] = {}

    for r in r_list:

        stencil_Re_1, pad =  calc_MC_surface_harmonic_stencil(h, h, h, r, 4, 1, accuracy = 6)
        stencil_Re_11, pad =  calc_MC_surface_harmonic_stencil(h, h, h, r, 4, 11, accuracy = 6)
        stencil_Re_15, pad =  calc_MC_surface_harmonic_stencil(h, h, h, r, 4, 15, accuracy = 6)
        MCSH_stencil_dict["4_1"][str(r)] = [[stencil_Re_1,stencil_Re_11,stencil_Re_15], pad ]


        stencil_Re_2, pad =  calc_MC_surface_harmonic_stencil(h, h, h, r, 4, 2, accuracy = 6)
        stencil_Re_3, pad =  calc_MC_surface_harmonic_stencil(h, h, h, r, 4, 3, accuracy = 6)
        stencil_Re_7, pad =  calc_MC_surface_harmonic_stencil(h, h, h, r, 4, 7, accuracy = 6)
        stencil_Re_10, pad =  calc_MC_surface_harmonic_stencil(h, h, h, r, 4, 10, accuracy = 6)
        stencil_Re_12, pad =  calc_MC_surface_harmonic_stencil(h, h, h, r, 4, 12, accuracy = 6)
        stencil_Re_14, pad =  calc_MC_surface_harmonic_stencil(h, h, h, r, 4, 14, accuracy = 6)
        MCSH_stencil_dict["4_2"][str(r)] = [[stencil_Re_2,stencil_Re_3,stencil_Re_7,stencil_Re_10,stencil_Re_12,stencil_Re_14], pad ]


        stencil_Re_4, pad =  calc_MC_surface_harmonic_stencil(h, h, h, r, 4, 4, accuracy = 6)
        stencil_Re_6, pad =  calc_MC_surface_harmonic_stencil(h, h, h, r, 4, 6, accuracy = 6)
        stencil_Re_13, pad =  calc_MC_surface_harmonic_stencil(h, h, h, r, 4, 13, accuracy = 6)
        MCSH_stencil_dict["4_3"][str(r)] = [[stencil_Re_4,stencil_Re_6,stencil_Re_13], pad ]




        stencil_Re_5, pad =  calc_MC_surface_harmonic_stencil(h, h, h, r, 4, 5, accuracy = 6)
        stencil_Re_8, pad =  calc_MC_surface_harmonic_stencil(h, h, h, r, 4, 8, accuracy = 6)
        stencil_Re_9, pad =  calc_MC_surface_harmonic_stencil(h, h, h, r, 4, 9, accuracy = 6)
        MCSH_stencil_dict["4_4"][str(r)] = [[stencil_Re_5,stencil_Re_8,stencil_Re_9], pad ]

    return MCSH_stencil_dict



def process(molecule, functional,i,j,k,h,N,r_list,MC_surface_harmonic_stencil_dict):
    result_filename = "{}_{}_{}_{}_{}_all_descriptors.hdf5".format(molecule,functional,i,j,k)
    #if os.path.isfile(result_filename) == False:
    print 'start {} {} {}'.format(i,j,k)
    process_range_descriptor(molecule, functional,i,j,k,h,N,r_list,MC_surface_harmonic_stencil_dict)
    

def process_one_molecule(molecule, functional,h,L,N,r_list):
    cwd = os.getcwd()
    dir_name = "{}_{}_{}_{}_{}".format(molecule,functional,str(L).replace('.','-'),str(h).replace('.','-'),N)
    print dir_name
    
    if os.path.isdir(dir_name) == False:
        print '\n****Error: Cant find the directory! ****\n'
        raise NotImplementedError
    
    os.chdir(cwd + '/' + dir_name)
    MCSH_stencil_dict = prepare_MCSH_stencils(r_list,h)
    
    
    Nx = Ny = Nz = N
    i_li = range(Nx)
    j_li = range(Ny)
    k_li = range(Nz)
    
    paramlist = list(itertools.product(i_li,j_li,k_li))



    for i,j,k in paramlist:
        process(molecule, functional,i,j,k,h,N,r_list,MCSH_stencil_dict)
        
    #process(molecule, functional,0,0,0,h,N,r_list,MCSH_stencil_dict)
    
    os.chdir(cwd)
    return


if __name__ == "__main__":

    print "start adding dataset"

    setup_filename = sys.argv[1]
    choice = sys.argv[2]

    #with open(setup_filename, encoding='utf-8') as f:
    with open(setup_filename) as f:
        setup = json.load(f)

    print setup

    if choice not in ['single','set']:
        raise NotImplementedError
    
    if choice == 'single':
        molecule = sys.argv[3]
        print "start"
        h = float(setup['grid_spacing'])
        L = float(setup['box_dimension'])
        N = int(setup['number_segment_per_side'])
        functional = setup['functionals']
        r_list = setup['r_list']

        #for functional in functionals:
        print "start process molecule"
        process_one_molecule(molecule, functional,h,L,N,r_list)

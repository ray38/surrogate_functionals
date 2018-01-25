# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 15:53:10 2017

@author: ray
"""

import numpy as np
import sys
import math

import csv
import h5py
import os

import itertools
import json



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

def predict(n,x):

    return lda_x(n,x) + lda_c(n,x)


def process_normal_descriptors(molecule, functional,i,j,k):
    result = []
    
    raw_data_filename = "{}_{}_{}_{}_{}.hdf5".format(molecule,functional,i,j,k)
    result_filename = "{}_{}_{}_{}_{}_all_descriptors.hdf5".format(molecule,functional,i,j,k)
    raw_data =  h5py.File(raw_data_filename,'r')
    x = np.asarray(raw_data['x'])[::5,::5,::5]
    y = np.asarray(raw_data['y'])[::5,::5,::5]
    z = np.asarray(raw_data['z'])[::5,::5,::5]
    n = np.asarray(raw_data['rho'])[::5,::5,::5]
    V_xc = np.asarray(raw_data['V_xc'])[::5,::5,::5]
    ep_xc =  np.asarray(raw_data['epsilon_xc'])[::5,::5,::5]
    gamma =  np.asarray(raw_data['gamma'])[::5,::5,::5]
    tau =  np.asarray(raw_data['tau'])[::5,::5,::5]
    raw_data.close()

    LDA_x = [-0.33080996,  0.02474374,  1.4517462,   0.3657363,  -2.31230322,  3.56469899, 0.3858979 ]

    LDA_residual = predict(n,LDA_x) - ep_xc

    result.append( np.around(x,2).flatten().tolist())
    result.append( np.around(y,2).flatten().tolist())
    result.append( np.around(z,2).flatten().tolist())
    result.append( np.around(n,9).flatten().tolist())
    result.append(np.around(gamma,9).flatten().tolist())
#    result.append(tau.flatten().tolist())
#    result.append(V_xc.flatten().tolist())
    result.append(np.around(ep_xc,9).flatten().tolist())
    result.append(np.around(LDA_residual,9).flatten().tolist())
        
    return result

    return np.around(x,2).flatten().tolist(), np.around(y,2).flatten().tolist(), np.around(z,2).flatten().tolist(), np.around(n,2).flatten().tolist(), np.around(gamma,2).flatten().tolist(), np.around(ep_xc,2).flatten().tolist(), np.around(LDA_residual,2).flatten().tolist()



def process(molecule, functional,i,j,k,h,N):

    print 'start {} {} {}'.format(i,j,k)
    result  = process_normal_descriptors(molecule, functional,i,j,k)
    return result
    

def process_one_molecule(molecule, functional,h,L,N):
    cwd = os.getcwd()
    dir_name = "{}_{}_{}_{}_{}".format(molecule,functional,str(L).replace('.','-'),str(h).replace('.','-'),N)
    print dir_name
    
    if os.path.isdir(dir_name) == False:
        print '\n****Error: Cant find the directory! ****\n'
        raise NotImplementedError
    
    os.chdir(cwd + '/' + dir_name)


    
    Nx = Ny = Nz = N
    i_li = range(Nx)
    j_li = range(Ny)
    k_li = range(Nz)
    
    paramlist = list(itertools.product(i_li,j_li,k_li))


    overall_list = []
    x = []
    y = []
    z = []
    n = []
    gamma = []
    epxc = []
    LDA_residual = []
    for i,j,k in paramlist:
        temp_x, temp_y, temp_z,  temp_n,  temp_gamma,  temp_epxc,  temp_LDAresidual = process(molecule, functional,i,j,k,h,N)
        x += temp_x
        y += temp_y
        z += temp_z
        n += temp_n
        gamma += temp_gamma
        epxc += temp_epxc
        LDA_residual += temp_LDAresidual

    overall_list.append(x)
    overall_list.append(y)
    overall_list.append(z)
    overall_list.append(n)
    overall_list.append(gamma)
    overall_list.append(epxc)
    overall_list.append(LDA_residual)
    
    overall_list = np.stack(overall_list,axis=1).tolist()
    with open("{}_{}_downsampled_data.csv".format(molecule,functional), "wb") as f:
        writer = csv.writer(f)
#            writer.writerow(['x','y','z','rho','gamma','tau','Vxc','epxc','ad_0-01','ad_0-02','ad_0-03','ad_0-04','ad_0-05','ad_0-06','ad_0-08','ad_0-1','ad_0-15','ad_0-2','ad_0-3','ad_0-4','ad_0-5','deriv_1','deriv_2'])
        writer.writerow(['x','y','z','rho','gamma','epxc','LDA_residual'])
        writer.writerows(overall_list)
    
    os.chdir(cwd)
    return


if __name__ == "__main__":

    setup_filename = sys.argv[1]

    #with open(setup_filename, encoding='utf-8') as f:
    with open(setup_filename) as f:
        setup = json.load(f)

    print setup

    molecule = sys.argv[2]
    h = float(setup['grid_spacing'])
    L = float(setup['box_dimension'])
    N = int(setup['number_segment_per_side'])
    functional = setup['functionals']

    #for functional in functionals:
    process_one_molecule(molecule, functional,h,L,N)


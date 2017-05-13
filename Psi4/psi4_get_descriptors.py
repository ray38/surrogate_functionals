# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 13:30:00 2017

@author: ray
"""


import sys
from convolutions import get_differenciation_conv, get_integration_conv, get_integration_fftconv, read_integration_stencil_file, get_integral_fftconv_with_known_stencil ,get_auto_accuracy
from getDescriptors import get_discriptors_from_density_integral_simple_norm_psi4_test,get_discriptors_from_density_integral_simple_norm_psi4_test_short,get_discriptors_from_density_integral_simple_norm_psi4_test_extra

try: import cPickle as pickle
except: import pickle
import time
import h5py
import numpy as np
import os
import random


list_molecule_filename = sys.argv[1]
method = sys.argv[2]

#try:
#    temp = sys.argv[3]
#    if temp == 'periodic':
#        periodic = True
#    else:
#        periodic = False
#        raise ValueError
#except:
#    periodic = False

#if periodic == True:
#    print "All analysis are with PBC turned on"

with open(list_molecule_filename) as f:
    molecule_names = f.readlines()
molecule_names = [x.strip() for x in molecule_names]

xc_funcs = ['SVWN','PBE0','PBE','B3LYP']
all_data = {}

successful_list = []
exist_list = []
failed_list = []

random.shuffle(molecule_names)
for molecule in molecule_names:
    for xc in reversed(xc_funcs):
        
        psi4_filename = '{}_{}.hdf5'.format(molecule,xc)
        print(psi4_filename)
        descriptor_filename = '{}_{}_integral_descriptors.p'.format(molecule,xc)

#        try:
#            temp = pickle.load( open( descriptor_filename, "rb" ) )
#            exist_list.append(molecule)
        if os.path.isfile(descriptor_filename) == False:
#        except:
#            try:
            data = h5py.File(psi4_filename,'r')
            n = np.asarray(data['rho'])
            V_xc = np.asarray(data['V_xc'])
            ep_xc =  np.asarray(data['epsilon_xc'])
            tau = np.asarray(data['tau'])
            gamma =  np.asarray(data['gamma'])
            h = data['h_x'][0]
            num_e = np.multiply(np.sum(n),(h*h*h))

            
            result_descriptors = get_discriptors_from_density_integral_simple_norm_psi4_test_extra(h,h,h, n, num_e,  V_xc, ep_xc, tau, gamma, periodic = False)
    
            with open(descriptor_filename, 'wb') as handle:
                pickle.dump(result_descriptors, handle, protocol=2)


#            except:
#                print("failed " + molecule)


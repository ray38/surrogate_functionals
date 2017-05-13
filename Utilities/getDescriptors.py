# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 11:34:33 2017

@author: ray
"""
import sys
try:
    path = "/home/ray/Documents/gpaw_test/gpawDFT"
    path_ase = "/home/ray/Documents/gpaw_test/ase-master-Jan-2017"
    path_utility = "/home/ray/Documents/Utilities"
    
    sys.path.insert(0,path)
    sys.path.insert(0,path_ase)
    sys.path.insert(0,path_utility)
    
    from gpaw import GPAW, restart

except:
    print('not imported gpaw')

#from convolutions import *
#from integration import *

#from gpaw import GPAW, restart
import numpy as np
from operator import itemgetter

from scipy.ndimage.filters import convolve
import math

from convolutions import get_differenciation_conv, get_integration_conv, get_integration_fftconv, read_integration_stencil_file, get_integral_fftconv_with_known_stencil ,get_auto_accuracy
import time
#import math


#def get_h_1s_desity_integral(r):
#    return -(math.exp(-2.*r)-1.) 

#
#def get_auto_accuracy(hx,hy,hz, r):
#    h = max([hx,hy,hz])
#    temp = 5 - int(math.floor((r/h)/3.))
#    if temp < 1:
#        return 1
#    else:
#        return temp




def get_homogeneous_gas_integral(n,r):
    return r*r*n*math.pi

def get_homo_nondimensional(int_arr, n_arr, r):
    temp = (4./3.)*r*r*r*math.pi
    result = np.divide(int_arr, n_arr)
    return np.divide(result, temp)

def get_homo_nondimensional_nave(int_arr, n_ave, r):
    temp = (4./3.)*r*r*r*math.pi*n_ave
    return np.divide(int_arr, temp)

 
def get_h_1s_desity_integral(r):
    r_bohr = r * 1.88973
    return -(math.exp(-2.*r_bohr)-1.)    
    
def correct_s(x, order_correct):
    temp = -1.*(x-0.5)**2 +0.25 +x
    if temp >=1.:
        return 1.
    else:
        return temp



def get_inhomogeneity_parameter(dn_arr, n_arr, hx,hy,hz):
    k = (dn_arr / n_arr)
    temp, temp2= get_differenciation_conv((k.copy())**2., hx, hy, hz, gradient = 'first',
                                               stencil_type = 'mid', accuracy = '2')
    numer = temp ** 2.
    denom = k ** 6.
    
    return numer/denom

def calculate_dimensionless_reduced_gradient_entry(dn,n):
    kf = math.pow((3. * math.pi * math.pi * n) , (1/3))
    s  = abs(n) / (2. * kf * n)
    return s
    
def calculate_dimensionless_reduced_gradient(dn_arr, n_arr):
    result = np.zeros_like(dn_arr)
    for index, num in np.ndenumerate(n_arr):
        result[index[0]][index[1]][index[2]] = calculate_dimensionless_reduced_gradient_entry( num,
                                                     dn_arr[index[0]][index[1]][index[2]])
    return result

def get_pads(pad_list):
    padx = max(pad_list,key=itemgetter(1))[0]
    pady = max(pad_list,key=itemgetter(1))[1]
    padz = max(pad_list,key=itemgetter(1))[2]
    return padx, pady, padz



def get_discriptors_from_density_generic(n,hx,hy,hz, nt, num_e, periodic = False):
    '''
    get the first-fourth derivative from the density matrix using convolution
    get the integration convolution at 0.5, 1.0, 1.5, 2.0, 2.5
    '''
    def get_xyz_descriptors(n, hx, hy, hz, num_e):
#        dimx, dimy, dimz = n.shape
        dimx = np.ones(n.shape)*n.shape[0]
        dimy = np.ones(n.shape)*n.shape[1]
        dimz = np.ones(n.shape)*n.shape[2]
        
        hx_ = np.ones_like(n)*hx
        hy_ = np.ones_like(n)*hy
        hz_ = np.ones_like(n)*hz
        num_e_ = np.ones_like(n)*float(num_e)

        
        
        x = np.ones(n.shape)
        y = np.ones(n.shape)
        z = np.ones(n.shape)

        
        for index, density in np.ndenumerate(n):
            x[index[0]][index[1]][index[2]] = index[0]
            y[index[0]][index[1]][index[2]] = index[1]
            z[index[0]][index[1]][index[2]] = index[2]
        
        return x, y, z, dimx, dimy, dimz, hx_, hy_, hz_, num_e_
    
    
    pad_list = []  
    result = []
    x, y, z, dimx, dimy, dimz, hx_, hy_, hz_, num_e_ = get_xyz_descriptors(n, hx, hy, hz, num_e)
    
    
    
    print('\n\ngetting differentiation convolutions...')
    first_grad_conv, temp_pad  = get_differenciation_conv(n.copy(), hx, hy, hz, gradient = 'first',
                                               stencil_type = 'mid', accuracy = '2')
    pad_list.append(temp_pad)
    print(first_grad_conv.shape)
    second_grad_conv, temp_pad  = get_differenciation_conv(n.copy(), hx, hy, hz, gradient = 'second',
                                               stencil_type = 'mid', accuracy = '2')                                               
    pad_list.append(temp_pad)              
                             
    third_grad_conv, temp_pad  = get_differenciation_conv(n.copy(), hx, hy, hz, gradient = 'third',
                                               stencil_type = 'mid', accuracy = '2')                                               
    pad_list.append(temp_pad)                  
                         
    fourth_grad_conv, temp_pad  = get_differenciation_conv(n.copy(), hx, hy, hz, gradient = 'fourth',
                                               stencil_type = 'mid', accuracy = '2')    
    pad_list.append(temp_pad)       
    
    
    print('\n\ngetting dimensionless quantities')
    dimless_grad = calculate_dimensionless_reduced_gradient(first_grad_conv.copy(), n.copy())
    dimless_sec_grad, temp_pad  = get_differenciation_conv(dimless_grad.copy(), hx, hy, hz, gradient = 'first',
                                               stencil_type = 'mid', accuracy = '2')    
    pad_list.append(temp_pad)
    
    dimless_third_grad, temp_pad  = get_differenciation_conv(dimless_grad.copy(), hx, hy, hz, gradient = 'second',
                                               stencil_type = 'mid', accuracy = '2')    
    pad_list.append(temp_pad)
    
    dimless_fourth_grad, temp_pad  = get_differenciation_conv(dimless_grad.copy(), hx, hy, hz, gradient = 'third',
                                               stencil_type = 'mid', accuracy = '2')    
    pad_list.append(temp_pad)

    inhomogeneity_param = get_inhomogeneity_parameter(first_grad_conv.copy(), n.copy(), hx, hy, hz)

    print('\n\ngetting integration convolutions...')
    integration_0_2, temp_pad  = get_integration_conv(n.copy(), hx, hy, hz, 0.2, accuracy = 4)
    pad_list.append(temp_pad)
  
    integration_0_4, temp_pad  = get_integration_conv(n.copy(), hx, hy, hz, 0.4, accuracy = 4)
    pad_list.append(temp_pad)
    
    integration_0_6, temp_pad  = get_integration_conv(n.copy(), hx, hy, hz, 0.6, accuracy = 4)
    pad_list.append(temp_pad)
    
    integration_0_8, temp_pad  = get_integration_conv(n.copy(), hx, hy, hz, 0.8, accuracy = 4)
    pad_list.append(temp_pad)
    
    integration_1_0, temp_pad  = get_integration_conv(n.copy(), hx, hy, hz, 1.0, accuracy = 4)
    pad_list.append(temp_pad)                                     

    integration_homo_non_0_2 = get_homo_nondimensional(integration_0_2.copy(), n.copy(), 0.2)
    integration_homo_non_0_4 = get_homo_nondimensional(integration_0_4.copy(), n.copy(), 0.4)
    integration_homo_non_0_6 = get_homo_nondimensional(integration_0_6.copy(), n.copy(), 0.6)
    integration_homo_non_0_8 = get_homo_nondimensional(integration_0_8.copy(), n.copy(), 0.8)
    integration_homo_non_1_0 = get_homo_nondimensional(integration_1_0.copy(), n.copy(), 1.0)
    
    a = get_h_1s_desity_integral(0.2)
    b = get_h_1s_desity_integral(0.4)
    c = get_h_1s_desity_integral(0.6)
    d = get_h_1s_desity_integral(0.8)
    e = get_h_1s_desity_integral(1.0)
    
    integration_h1s_non_0_2 = integration_0_2.copy() / a
    integration_h1s_non_0_4 = integration_0_4.copy() / b
    integration_h1s_non_0_6 = integration_0_6.copy() / c
    integration_h1s_non_0_8 = integration_0_8.copy() / d
    integration_h1s_non_1_0 = integration_1_0.copy() / e
    
    
    if periodic:
        padx = 1
        pady = 1
        padz = 1
    else:
        padx, pady, padz = get_pads(pad_list)

    
    result = zip(   x[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    y[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    z[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    dimx[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    dimy[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    dimz[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    hx_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    hy_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    hz_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    num_e_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    n[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    nt[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    first_grad_conv[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    second_grad_conv[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    third_grad_conv[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    fourth_grad_conv[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    dimless_grad[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    dimless_sec_grad[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    dimless_third_grad[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    dimless_fourth_grad[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    inhomogeneity_param[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    integration_0_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    integration_0_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    integration_0_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    integration_0_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    integration_1_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    integration_homo_non_0_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    integration_homo_non_0_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    integration_homo_non_0_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    integration_homo_non_0_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    integration_homo_non_1_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    integration_h1s_non_0_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    integration_h1s_non_0_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    integration_h1s_non_0_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    integration_h1s_non_0_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                    integration_h1s_non_1_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist())
    
    print('\n\n\n######\nafter convolution')
    print(type(result))
    print(len(result))
    print(result[0])
    return result




def get_discriptors_from_density_dense_integral(nt,hx,hy,hz, n, It, num_e, periodic = False, integral_accuracy = 4):
    '''
    get the first-fourth derivative from the density matrix using convolution
    get the integration convolution at 0.5, 1.0, 1.5, 2.0, 2.5
    '''
    def get_xyz_descriptors(nt, hx, hy, hz, It, num_e):

        dimx = np.ones(nt.shape)*nt.shape[0]
        dimy = np.ones(nt.shape)*nt.shape[1]
        dimz = np.ones(nt.shape)*nt.shape[2]
      
        nt_ave = It / float(hx*hy*hz) 
                
        hx_ = np.ones_like(nt)*hx
        hy_ = np.ones_like(nt)*hy
        hz_ = np.ones_like(nt)*hz
        num_e_ = np.ones_like(nt)*float(num_e)
        It_ = np.ones_like(nt)*float(It)
             
        x = np.ones(nt.shape)
        y = np.ones(nt.shape)
        z = np.ones(nt.shape)
        
        for index, density in np.ndenumerate(nt):
            x[index[0]][index[1]][index[2]] = index[0]
            y[index[0]][index[1]][index[2]] = index[1]
            z[index[0]][index[1]][index[2]] = index[2]
        
        return x, y, z, dimx, dimy, dimz, hx_, hy_, hz_, num_e_, It_, nt_ave
#    
#    def get_auto_accuracy(hx,hy,hz, r):
#        h = max([hx,hy,hz])
#        temp = 5 - int(math.floor((r/h)/3.))
#        if temp < 1:
#            return 1
#        else:
#            return temp
    
    n_plot = n.copy()
    nt_plot = nt.copy()    
    
 
    pad_list = []  
    result = []
    x, y, z, dimx, dimy, dimz, hx_, hy_, hz_, num_e_, It_, nt_ave = get_xyz_descriptors(n, hx, hy, hz, It, num_e)
    
    
    start = time.time()    
    
    print('\n\ngetting integration convolutions...')
    integration_0_05, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.05, accuracy = get_auto_accuracy(hx,hy,hz, 0.05))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_0_1, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.1, accuracy = get_auto_accuracy(hx,hy,hz, 0.1))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_0_15, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.15, accuracy = get_auto_accuracy(hx,hy,hz, 0.15))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_0_2, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.2, accuracy = get_auto_accuracy(hx,hy,hz, 0.2))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_0_25, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.25, accuracy = get_auto_accuracy(hx,hy,hz, 0.25))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_0_3, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.3, accuracy = get_auto_accuracy(hx,hy,hz, 0.3))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_0_35, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.35, accuracy = get_auto_accuracy(hx,hy,hz, 0.35))
    pad_list.append(temp_pad)
    print('done 0.2')
    
    integration_0_4, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.4, accuracy = get_auto_accuracy(hx,hy,hz, 0.4))
    pad_list.append(temp_pad)
    print('done 0.4')

    integration_0_45, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.45, accuracy = get_auto_accuracy(hx,hy,hz, 0.45))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_0_5, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.5, accuracy = get_auto_accuracy(hx,hy,hz, 0.5))
    pad_list.append(temp_pad)
    print('done 0.2')
  
    integration_0_6, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.6, accuracy = get_auto_accuracy(hx,hy,hz, 0.6))
    pad_list.append(temp_pad)
    print('done 0.6')

    integration_0_7, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.7, accuracy = get_auto_accuracy(hx,hy,hz, 0.7))
    pad_list.append(temp_pad)
    print('done 0.8')
    
    integration_0_8, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.8, accuracy = get_auto_accuracy(hx,hy,hz, 0.8))
    pad_list.append(temp_pad)
    print('done 0.8')

    integration_0_9, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.9, accuracy = get_auto_accuracy(hx,hy,hz, 0.9))
    pad_list.append(temp_pad)
    print('done 0.8')
    
    integration_1_0, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.0, accuracy = get_auto_accuracy(hx,hy,hz, 1.0))
    pad_list.append(temp_pad)
    print('done 1.0')

    integration_1_1, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.1, accuracy = get_auto_accuracy(hx,hy,hz, 1.1))
    pad_list.append(temp_pad)
    print('done 0.8')
    
    integration_1_2, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.2, accuracy = get_auto_accuracy(hx,hy,hz, 1.2))
    pad_list.append(temp_pad)
    print('done 1.2')

    integration_1_3, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.3, accuracy = get_auto_accuracy(hx,hy,hz, 1.3))
    pad_list.append(temp_pad)
    print('done 0.8')
    
    integration_1_4, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.4, accuracy = get_auto_accuracy(hx,hy,hz, 1.4))
    pad_list.append(temp_pad)
    print('done 1.4')

    integration_1_5, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.5, accuracy = get_auto_accuracy(hx,hy,hz, 1.5))
    pad_list.append(temp_pad)
    print('done 0.8')
    
    integration_1_6, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.6, accuracy = get_auto_accuracy(hx,hy,hz, 1.6))
    pad_list.append(temp_pad)
    print('done 1.6')

    integration_1_7, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.7, accuracy = get_auto_accuracy(hx,hy,hz, 1.7))
    pad_list.append(temp_pad)
    print('done 1.8')
    
    integration_1_8, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.8, accuracy = get_auto_accuracy(hx,hy,hz, 1.8))
    pad_list.append(temp_pad)
    print('done 1.8')

    integration_1_9, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.9, accuracy = get_auto_accuracy(hx,hy,hz, 1.9))
    pad_list.append(temp_pad)
    print('done 1.8')
    
    integration_2_0, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 2.0, accuracy = get_auto_accuracy(hx,hy,hz, 2.0))
    pad_list.append(temp_pad)
    print('done 2.0')

    integration_2_2, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 2.2, accuracy = get_auto_accuracy(hx,hy,hz, 2.2))
    pad_list.append(temp_pad)
    print('done 2.2')
  
    integration_2_4, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 2.4, accuracy = get_auto_accuracy(hx,hy,hz, 2.4))
    pad_list.append(temp_pad)
    print('done 2.4')
    
    integration_2_6, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 2.6, accuracy = get_auto_accuracy(hx,hy,hz, 2.6))
    pad_list.append(temp_pad)
    print('done 2.6')
    
    integration_2_8, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 2.8, accuracy = get_auto_accuracy(hx,hy,hz, 2.8))
    pad_list.append(temp_pad)
    print('done 2.8')
    
    integration_3_0, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 3.0, accuracy = get_auto_accuracy(hx,hy,hz, 3.0))
    pad_list.append(temp_pad)
    print('done 3.0'                           )

    integration_3_2, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 3.2, accuracy = get_auto_accuracy(hx,hy,hz, 3.2))
    pad_list.append(temp_pad)
    print('done 1.2')
  
    integration_3_4, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 3.4, accuracy = get_auto_accuracy(hx,hy,hz, 3.4))
    pad_list.append(temp_pad)
    print('done 1.4')
    
    integration_3_6, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 3.6, accuracy = get_auto_accuracy(hx,hy,hz, 3.6))
    pad_list.append(temp_pad)
    print('done 1.6')
    
    integration_3_8, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 3.8, accuracy = get_auto_accuracy(hx,hy,hz, 3.8))
    pad_list.append(temp_pad)
    print('done 1.8')
    
    integration_4_0, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 4.0, accuracy = get_auto_accuracy(hx,hy,hz, 4.0))
    pad_list.append(temp_pad)
    print('done 2.0')

    integration_4_2, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 4.2, accuracy = get_auto_accuracy(hx,hy,hz, 4.2))
    pad_list.append(temp_pad)
    print('done 2.2')
  
    integration_4_4, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 4.4, accuracy = get_auto_accuracy(hx,hy,hz, 4.4))
    pad_list.append(temp_pad)
    print('done 2.4')
    
    integration_4_6, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 4.6, accuracy = get_auto_accuracy(hx,hy,hz, 4.6))
    pad_list.append(temp_pad)
    print('done 2.6')
    
    integration_4_8, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 4.8, accuracy = get_auto_accuracy(hx,hy,hz, 4.8))
    pad_list.append(temp_pad)
    print('done 2.8')
    
    integration_5_0, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 5.0, accuracy = get_auto_accuracy(hx,hy,hz, 5.0))
    pad_list.append(temp_pad)
    print('done 3.0'   )

    integration_5_2, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 5.2, accuracy = get_auto_accuracy(hx,hy,hz, 5.2))
    pad_list.append(temp_pad)
    print('done 2.2')
  
    integration_5_4, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 5.4, accuracy = get_auto_accuracy(hx,hy,hz, 5.4))
    pad_list.append(temp_pad)
    print('done 2.4')
    
    integration_5_6, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 5.6, accuracy = get_auto_accuracy(hx,hy,hz, 5.6))
    pad_list.append(temp_pad)
    print('done 2.6')
    
    integration_5_8, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 5.8, accuracy = get_auto_accuracy(hx,hy,hz, 5.8))
    pad_list.append(temp_pad)
    print('done 2.8')
    
    integration_6_0, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 6.0, accuracy = get_auto_accuracy(hx,hy,hz, 6.0))
    pad_list.append(temp_pad)
    print('done 3.0' )

    print("after first conv: " + str(time.time()-start))
    
    integration_homo_non_0_05 = get_homo_nondimensional(integration_0_05.copy(), nt.copy(), 0.05)
    integration_homo_non_0_1 = get_homo_nondimensional(integration_0_1.copy(), nt.copy(), 0.1)
    integration_homo_non_0_15 = get_homo_nondimensional(integration_0_15.copy(), nt.copy(), 0.15)
    integration_homo_non_0_2 = get_homo_nondimensional(integration_0_2.copy(), nt.copy(), 0.2)
    integration_homo_non_0_25 = get_homo_nondimensional(integration_0_25.copy(), nt.copy(), 0.25)
    integration_homo_non_0_3 = get_homo_nondimensional(integration_0_3.copy(), nt.copy(), 0.3)
    integration_homo_non_0_35 = get_homo_nondimensional(integration_0_35.copy(), nt.copy(), 0.35)
    integration_homo_non_0_4 = get_homo_nondimensional(integration_0_4.copy(), nt.copy(), 0.4)
    integration_homo_non_0_45 = get_homo_nondimensional(integration_0_45.copy(), nt.copy(), 0.45)
    integration_homo_non_0_5 = get_homo_nondimensional(integration_0_5.copy(), nt.copy(), 0.5)
    integration_homo_non_0_6 = get_homo_nondimensional(integration_0_6.copy(), nt.copy(), 0.6)
    integration_homo_non_0_7 = get_homo_nondimensional(integration_0_7.copy(), nt.copy(), 0.7)
    integration_homo_non_0_8 = get_homo_nondimensional(integration_0_8.copy(), nt.copy(), 0.8)
    integration_homo_non_0_9 = get_homo_nondimensional(integration_0_9.copy(), nt.copy(), 0.9)
    integration_homo_non_1_0 = get_homo_nondimensional(integration_1_0.copy(), nt.copy(), 1.0)
    integration_homo_non_1_1 = get_homo_nondimensional(integration_1_1.copy(), nt.copy(), 1.1)
    integration_homo_non_1_2 = get_homo_nondimensional(integration_1_2.copy(), nt.copy(), 1.2)
    integration_homo_non_1_3 = get_homo_nondimensional(integration_1_3.copy(), nt.copy(), 1.3)
    integration_homo_non_1_4 = get_homo_nondimensional(integration_1_4.copy(), nt.copy(), 1.4)
    integration_homo_non_1_5 = get_homo_nondimensional(integration_1_5.copy(), nt.copy(), 1.5)
    integration_homo_non_1_6 = get_homo_nondimensional(integration_1_6.copy(), nt.copy(), 1.6)
    integration_homo_non_1_7 = get_homo_nondimensional(integration_1_7.copy(), nt.copy(), 1.7)
    integration_homo_non_1_8 = get_homo_nondimensional(integration_1_8.copy(), nt.copy(), 1.8)
    integration_homo_non_1_9 = get_homo_nondimensional(integration_1_9.copy(), nt.copy(), 1.9)
    integration_homo_non_2_0 = get_homo_nondimensional(integration_2_0.copy(), nt.copy(), 2.0)
    integration_homo_non_2_2 = get_homo_nondimensional(integration_2_2.copy(), nt.copy(), 2.2)
    integration_homo_non_2_4 = get_homo_nondimensional(integration_2_4.copy(), nt.copy(), 2.4)
    integration_homo_non_2_6 = get_homo_nondimensional(integration_2_6.copy(), nt.copy(), 2.6)
    integration_homo_non_2_8 = get_homo_nondimensional(integration_2_8.copy(), nt.copy(), 2.8)
    integration_homo_non_3_0 = get_homo_nondimensional(integration_3_0.copy(), nt.copy(), 3.0)
    integration_homo_non_3_2 = get_homo_nondimensional(integration_3_2.copy(), nt.copy(), 3.2)
    integration_homo_non_3_4 = get_homo_nondimensional(integration_3_4.copy(), nt.copy(), 3.4)
    integration_homo_non_3_6 = get_homo_nondimensional(integration_3_6.copy(), nt.copy(), 3.6)
    integration_homo_non_3_8 = get_homo_nondimensional(integration_3_8.copy(), nt.copy(), 3.8)
    integration_homo_non_4_0 = get_homo_nondimensional(integration_4_0.copy(), nt.copy(), 4.0)
    integration_homo_non_4_2 = get_homo_nondimensional(integration_4_2.copy(), nt.copy(), 4.2)
    integration_homo_non_4_4 = get_homo_nondimensional(integration_4_4.copy(), nt.copy(), 4.4)
    integration_homo_non_4_6 = get_homo_nondimensional(integration_4_6.copy(), nt.copy(), 4.6)
    integration_homo_non_4_8 = get_homo_nondimensional(integration_4_8.copy(), nt.copy(), 4.8)
    integration_homo_non_5_0 = get_homo_nondimensional(integration_5_0.copy(), nt.copy(), 5.0)
    integration_homo_non_5_2 = get_homo_nondimensional(integration_5_2.copy(), nt.copy(), 5.2)
    integration_homo_non_5_4 = get_homo_nondimensional(integration_5_4.copy(), nt.copy(), 5.4)
    integration_homo_non_5_6 = get_homo_nondimensional(integration_5_6.copy(), nt.copy(), 5.6)
    integration_homo_non_5_8 = get_homo_nondimensional(integration_5_8.copy(), nt.copy(), 5.8)
    integration_homo_non_6_0 = get_homo_nondimensional(integration_6_0.copy(), nt.copy(), 6.0)    

    integration_homo_non_0_05_ntave = get_homo_nondimensional_nave(integration_0_05.copy(), nt_ave, 0.05)
    integration_homo_non_0_1_ntave = get_homo_nondimensional_nave(integration_0_1.copy(), nt_ave, 0.1)
    integration_homo_non_0_15_ntave = get_homo_nondimensional_nave(integration_0_15.copy(), nt_ave, 0.15)
    
    integration_homo_non_0_2_ntave = get_homo_nondimensional_nave(integration_0_2.copy(), nt_ave, 0.2)
    integration_homo_non_0_25_ntave = get_homo_nondimensional_nave(integration_0_25.copy(), nt_ave, 0.25)
    integration_homo_non_0_3_ntave = get_homo_nondimensional_nave(integration_0_3.copy(), nt_ave, 0.3)
    integration_homo_non_0_35_ntave = get_homo_nondimensional_nave(integration_0_35.copy(), nt_ave, 0.35)
    
    integration_homo_non_0_4_ntave = get_homo_nondimensional_nave(integration_0_4.copy(), nt_ave, 0.4)
    integration_homo_non_0_45_ntave = get_homo_nondimensional_nave(integration_0_45.copy(), nt_ave, 0.45)
    integration_homo_non_0_5_ntave = get_homo_nondimensional_nave(integration_0_5.copy(), nt_ave, 0.5)
    integration_homo_non_0_6_ntave = get_homo_nondimensional_nave(integration_0_6.copy(), nt_ave, 0.6)
    integration_homo_non_0_7_ntave = get_homo_nondimensional_nave(integration_0_7.copy(), nt_ave, 0.7)
    integration_homo_non_0_8_ntave = get_homo_nondimensional_nave(integration_0_8.copy(), nt_ave, 0.8)
    integration_homo_non_0_9_ntave = get_homo_nondimensional_nave(integration_0_9.copy(), nt_ave, 0.9)
    integration_homo_non_1_0_ntave = get_homo_nondimensional_nave(integration_1_0.copy(), nt_ave, 1.0)
    integration_homo_non_1_1_ntave = get_homo_nondimensional_nave(integration_1_1.copy(), nt_ave, 1.1)
    integration_homo_non_1_2_ntave = get_homo_nondimensional_nave(integration_1_2.copy(), nt_ave, 1.2)
    integration_homo_non_1_3_ntave = get_homo_nondimensional_nave(integration_1_3.copy(), nt_ave, 1.3)
    integration_homo_non_1_4_ntave = get_homo_nondimensional_nave(integration_1_4.copy(), nt_ave, 1.4)
    integration_homo_non_1_5_ntave = get_homo_nondimensional_nave(integration_1_5.copy(), nt_ave, 1.5)
    integration_homo_non_1_6_ntave = get_homo_nondimensional_nave(integration_1_6.copy(), nt_ave, 1.6)
    integration_homo_non_1_7_ntave = get_homo_nondimensional_nave(integration_1_7.copy(), nt_ave, 1.7)
    integration_homo_non_1_8_ntave = get_homo_nondimensional_nave(integration_1_8.copy(), nt_ave, 1.8)
    integration_homo_non_1_9_ntave = get_homo_nondimensional_nave(integration_1_9.copy(), nt_ave, 1.9)
    integration_homo_non_2_0_ntave = get_homo_nondimensional_nave(integration_2_0.copy(), nt_ave, 2.0)
    integration_homo_non_2_2_ntave = get_homo_nondimensional_nave(integration_2_2.copy(), nt_ave, 2.2)
    integration_homo_non_2_4_ntave = get_homo_nondimensional_nave(integration_2_4.copy(), nt_ave, 2.4)
    integration_homo_non_2_6_ntave = get_homo_nondimensional_nave(integration_2_6.copy(), nt_ave, 2.6)
    integration_homo_non_2_8_ntave = get_homo_nondimensional_nave(integration_2_8.copy(), nt_ave, 2.8)
    integration_homo_non_3_0_ntave = get_homo_nondimensional_nave(integration_3_0.copy(), nt_ave, 3.0)
    integration_homo_non_3_2_ntave = get_homo_nondimensional_nave(integration_3_2.copy(), nt_ave, 3.2)
    integration_homo_non_3_4_ntave = get_homo_nondimensional_nave(integration_3_4.copy(), nt_ave, 3.4)
    integration_homo_non_3_6_ntave = get_homo_nondimensional_nave(integration_3_6.copy(), nt_ave, 3.6)
    integration_homo_non_3_8_ntave = get_homo_nondimensional_nave(integration_3_8.copy(), nt_ave, 3.8)
    integration_homo_non_4_0_ntave = get_homo_nondimensional_nave(integration_4_0.copy(), nt_ave, 4.0)
    integration_homo_non_4_2_ntave = get_homo_nondimensional_nave(integration_4_2.copy(), nt_ave, 4.2)
    integration_homo_non_4_4_ntave = get_homo_nondimensional_nave(integration_4_4.copy(), nt_ave, 4.4)
    integration_homo_non_4_6_ntave = get_homo_nondimensional_nave(integration_4_6.copy(), nt_ave, 4.6)
    integration_homo_non_4_8_ntave = get_homo_nondimensional_nave(integration_4_8.copy(), nt_ave, 4.8)
    integration_homo_non_5_0_ntave = get_homo_nondimensional_nave(integration_5_0.copy(), nt_ave, 5.0)
    integration_homo_non_5_2_ntave = get_homo_nondimensional_nave(integration_5_2.copy(), nt_ave, 5.2)
    integration_homo_non_5_4_ntave = get_homo_nondimensional_nave(integration_5_4.copy(), nt_ave, 5.4)
    integration_homo_non_5_6_ntave = get_homo_nondimensional_nave(integration_5_6.copy(), nt_ave, 5.6)
    integration_homo_non_5_8_ntave = get_homo_nondimensional_nave(integration_5_8.copy(), nt_ave, 5.8)
    integration_homo_non_6_0_ntave = get_homo_nondimensional_nave(integration_6_0.copy(), nt_ave, 6.0)

    print("after everything: " + str(time.time()-start)    )
    
    if periodic:
        result = zip(   x.flatten().tolist() ,\
                        y.flatten().tolist() ,\
                        z.flatten().tolist() ,\
                        dimx.flatten().tolist() ,\
                        dimy.flatten().tolist() ,\
                        dimz.flatten().tolist() ,\
                        hx_.flatten().tolist() ,\
                        hy_.flatten().tolist() ,\
                        hz_.flatten().tolist() ,\
                        It_.flatten().tolist() ,\
                        num_e_.flatten().tolist() ,\
                        nt_plot.flatten().tolist() ,\
                        n_plot.flatten().tolist() ,\
                        integration_0_05.flatten().tolist() ,\
                        integration_0_1.flatten().tolist() ,\
                        integration_0_15.flatten().tolist() ,\
                        integration_0_2.flatten().tolist() ,\
                        integration_0_25.flatten().tolist() ,\
                        integration_0_3.flatten().tolist() ,\
                        integration_0_35.flatten().tolist() ,\
                        integration_0_4.flatten().tolist() ,\
                        integration_0_45.flatten().tolist() ,\
                        integration_0_5.flatten().tolist() ,\
                        integration_0_6.flatten().tolist() ,\
                        integration_0_2.flatten().tolist() ,\
                        integration_0_8.flatten().tolist() ,\
                        integration_0_2.flatten().tolist() ,\
                        integration_1_0.flatten().tolist() ,\
                        integration_1_1.flatten().tolist() ,\
                        integration_1_2.flatten().tolist() ,\
                        integration_1_3.flatten().tolist() ,\
                        integration_1_4.flatten().tolist() ,\
                        integration_1_5.flatten().tolist() ,\
                        integration_1_6.flatten().tolist() ,\
                        integration_1_7.flatten().tolist() ,\
                        integration_1_8.flatten().tolist() ,\
                        integration_1_9.flatten().tolist() ,\
                        integration_2_0.flatten().tolist() ,\
                        integration_2_2.flatten().tolist() ,\
                        integration_2_4.flatten().tolist() ,\
                        integration_2_6.flatten().tolist() ,\
                        integration_2_8.flatten().tolist() ,\
                        integration_3_0.flatten().tolist() ,\
                        integration_3_2.flatten().tolist() ,\
                        integration_3_4.flatten().tolist() ,\
                        integration_3_6.flatten().tolist() ,\
                        integration_3_8.flatten().tolist() ,\
                        integration_4_0.flatten().tolist() ,\
                        integration_4_2.flatten().tolist() ,\
                        integration_4_4.flatten().tolist() ,\
                        integration_4_6.flatten().tolist() ,\
                        integration_4_8.flatten().tolist() ,\
                        integration_5_0.flatten().tolist() ,\
                        integration_5_2.flatten().tolist() ,\
                        integration_5_4.flatten().tolist() ,\
                        integration_5_6.flatten().tolist() ,\
                        integration_5_8.flatten().tolist() ,\
                        integration_6_0.flatten().tolist() ,\
                        integration_homo_non_0_05.flatten().tolist() ,\
                        integration_homo_non_0_1.flatten().tolist() ,\
                        integration_homo_non_0_15.flatten().tolist() ,\
                        integration_homo_non_0_2.flatten().tolist() ,\
                        integration_homo_non_0_25.flatten().tolist() ,\
                        integration_homo_non_0_3.flatten().tolist() ,\
                        integration_homo_non_0_35.flatten().tolist() ,\
                        integration_homo_non_0_4.flatten().tolist() ,\
                        integration_homo_non_0_45.flatten().tolist() ,\
                        integration_homo_non_0_5.flatten().tolist() ,\
                        integration_homo_non_0_6.flatten().tolist() ,\
                        integration_homo_non_0_7.flatten().tolist() ,\
                        integration_homo_non_0_8.flatten().tolist() ,\
                        integration_homo_non_0_9.flatten().tolist() ,\
                        integration_homo_non_1_0.flatten().tolist() ,\
                        integration_homo_non_1_1.flatten().tolist() ,\
                        integration_homo_non_1_2.flatten().tolist() ,\
                        integration_homo_non_1_3.flatten().tolist() ,\
                        integration_homo_non_1_4.flatten().tolist() ,\
                        integration_homo_non_1_5.flatten().tolist() ,\
                        integration_homo_non_1_6.flatten().tolist() ,\
                        integration_homo_non_1_7.flatten().tolist() ,\
                        integration_homo_non_1_8.flatten().tolist() ,\
                        integration_homo_non_1_9.flatten().tolist() ,\
                        integration_homo_non_2_0.flatten().tolist() ,\
                        integration_homo_non_2_2.flatten().tolist() ,\
                        integration_homo_non_2_4.flatten().tolist() ,\
                        integration_homo_non_2_6.flatten().tolist() ,\
                        integration_homo_non_2_8.flatten().tolist() ,\
                        integration_homo_non_3_0.flatten().tolist() ,\
                        integration_homo_non_3_2.flatten().tolist() ,\
                        integration_homo_non_3_4.flatten().tolist() ,\
                        integration_homo_non_3_6.flatten().tolist() ,\
                        integration_homo_non_3_8.flatten().tolist() ,\
                        integration_homo_non_4_0.flatten().tolist() ,\
                        integration_homo_non_4_2.flatten().tolist() ,\
                        integration_homo_non_4_4.flatten().tolist() ,\
                        integration_homo_non_4_6.flatten().tolist() ,\
                        integration_homo_non_4_8.flatten().tolist() ,\
                        integration_homo_non_5_0.flatten().tolist() ,\
                        integration_homo_non_5_2.flatten().tolist() ,\
                        integration_homo_non_5_4.flatten().tolist() ,\
                        integration_homo_non_5_6.flatten().tolist() ,\
                        integration_homo_non_5_8.flatten().tolist() ,\
                        integration_homo_non_6_0.flatten().tolist() ,\
                        integration_homo_non_0_05_ntave.flatten().tolist() ,\
                        integration_homo_non_0_1_ntave.flatten().tolist() ,\
                        integration_homo_non_0_15_ntave.flatten().tolist() ,\
                        integration_homo_non_0_2_ntave.flatten().tolist() ,\
                        integration_homo_non_0_25_ntave.flatten().tolist() ,\
                        integration_homo_non_0_3_ntave.flatten().tolist() ,\
                        integration_homo_non_0_35_ntave.flatten().tolist() ,\
                        integration_homo_non_0_4_ntave.flatten().tolist() ,\
                        integration_homo_non_0_45_ntave.flatten().tolist() ,\
                        integration_homo_non_0_5_ntave.flatten().tolist() ,\
                        integration_homo_non_0_6_ntave.flatten().tolist() ,\
                        integration_homo_non_0_7_ntave.flatten().tolist() ,\
                        integration_homo_non_0_8_ntave.flatten().tolist() ,\
                        integration_homo_non_0_9_ntave.flatten().tolist() ,\
                        integration_homo_non_1_0_ntave.flatten().tolist() ,\
                        integration_homo_non_1_1_ntave.flatten().tolist() ,\
                        integration_homo_non_1_2_ntave.flatten().tolist() ,\
                        integration_homo_non_1_3_ntave.flatten().tolist() ,\
                        integration_homo_non_1_4_ntave.flatten().tolist() ,\
                        integration_homo_non_1_5_ntave.flatten().tolist() ,\
                        integration_homo_non_1_6_ntave.flatten().tolist() ,\
                        integration_homo_non_1_7_ntave.flatten().tolist() ,\
                        integration_homo_non_1_8_ntave.flatten().tolist() ,\
                        integration_homo_non_1_9_ntave.flatten().tolist() ,\
                        integration_homo_non_2_0_ntave.flatten().tolist() ,\
                        integration_homo_non_2_2_ntave.flatten().tolist() ,\
                        integration_homo_non_2_4_ntave.flatten().tolist() ,\
                        integration_homo_non_2_6_ntave.flatten().tolist() ,\
                        integration_homo_non_2_8_ntave.flatten().tolist() ,\
                        integration_homo_non_3_0_ntave.flatten().tolist() ,\
                        integration_homo_non_3_2_ntave.flatten().tolist() ,\
                        integration_homo_non_3_4_ntave.flatten().tolist() ,\
                        integration_homo_non_3_6_ntave.flatten().tolist() ,\
                        integration_homo_non_3_8_ntave.flatten().tolist() ,\
                        integration_homo_non_4_0_ntave.flatten().tolist() ,\
                        integration_homo_non_4_2_ntave.flatten().tolist() ,\
                        integration_homo_non_4_4_ntave.flatten().tolist() ,\
                        integration_homo_non_4_6_ntave.flatten().tolist() ,\
                        integration_homo_non_4_8_ntave.flatten().tolist() ,\
                        integration_homo_non_5_0_ntave.flatten().tolist() ,\
                        integration_homo_non_5_2_ntave.flatten().tolist() ,\
                        integration_homo_non_5_4_ntave.flatten().tolist() ,\
                        integration_homo_non_5_6_ntave.flatten().tolist() ,\
                        integration_homo_non_5_8_ntave.flatten().tolist() ,\
                        integration_homo_non_6_0_ntave.flatten().tolist())
    else:
        padx, pady, padz = get_pads(pad_list)
  
        result = zip(   x[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        y[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        z[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        dimx[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        dimy[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        dimz[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        hx_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        hy_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        hz_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        It_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        num_e_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        nt_plot[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        n_plot[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_0_05[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_0_1[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_0_15[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_0_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_0_25[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_0_3[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_0_35[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_0_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_0_45[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_0_5[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_0_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_0_7[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_0_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_0_9[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_1_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_1_1[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_1_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_1_3[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_1_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_1_5[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_1_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_1_7[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_1_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_1_9[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_2_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_2_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_2_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_2_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_2_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_3_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_3_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_3_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_3_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_3_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_4_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_4_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_4_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_4_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_4_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_5_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_5_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_5_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_5_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_5_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_6_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_05[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_1[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_15[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_25[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_3[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_35[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_45[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_5[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_7[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_9[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_1_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_1_1[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_1_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_1_3[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_1_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_1_5[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_1_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_1_7[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_1_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_1_9[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_2_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_2_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_2_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_2_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_2_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_3_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_3_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_3_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_3_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_3_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_4_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_4_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_4_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_4_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_4_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_5_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_5_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_5_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_5_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_5_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_6_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_05_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_1_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_15_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_2_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_25_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_3_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_35_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_4_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_45_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_5_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_6_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_7_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_8_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_9_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_1_0_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_1_1_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_1_2_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_1_3_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_1_4_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_1_5_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_1_6_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_1_7_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_1_8_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_1_9_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_2_0_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_2_2_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_2_4_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_2_6_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_2_8_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_3_0_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_3_2_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_3_4_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_3_6_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_3_8_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_4_0_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_4_2_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_4_4_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_4_6_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_4_8_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_5_0_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_5_2_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_5_4_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_5_6_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_5_8_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_6_0_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist())
#                    integration_h1s_non_0_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_0_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_0_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_0_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_1_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_1_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_1_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_1_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_1_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_2_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_2_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_2_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_2_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_2_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_3_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_3_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_3_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_3_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_3_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_4_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_4_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_4_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_4_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_4_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_5_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist())
    print("after zip: " + str(time.time()-start))
#    print('\n\n\n######\nafter convolution')
#    print(type(result))
#    print(len(result))
#    print(result[0])
    return result

def get_discriptors_from_density_integral(nt,hx,hy,hz, n, It, num_e, periodic = False, integral_accuracy = 4):
    '''
    get the first-fourth derivative from the density matrix using convolution
    get the integration convolution at 0.5, 1.0, 1.5, 2.0, 2.5
    '''
    def get_xyz_descriptors(nt, hx, hy, hz, It, num_e):

        dimx = np.ones(nt.shape)*nt.shape[0]
        dimy = np.ones(nt.shape)*nt.shape[1]
        dimz = np.ones(nt.shape)*nt.shape[2]
      
        nt_ave = It / float(hx*hy*hz) 
                
        hx_ = np.ones_like(nt)*hx
        hy_ = np.ones_like(nt)*hy
        hz_ = np.ones_like(nt)*hz
        num_e_ = np.ones_like(nt)*float(num_e)
        It_ = np.ones_like(nt)*float(It)
             
        x = np.ones(nt.shape)
        y = np.ones(nt.shape)
        z = np.ones(nt.shape)
        
        for index, density in np.ndenumerate(nt):
            x[index[0]][index[1]][index[2]] = index[0]
            y[index[0]][index[1]][index[2]] = index[1]
            z[index[0]][index[1]][index[2]] = index[2]
        
        return x, y, z, dimx, dimy, dimz, hx_, hy_, hz_, num_e_, It_, nt_ave
    
#    def get_auto_accuracy(hx,hy,hz, r):
#        h = max([hx,hy,hz])
#        temp = 5 - int(math.floor((r/h)/3.))
#        if temp < 1:
#            return 1
#        else:
#            return temp
    
    n_plot = n.copy()
    nt_plot = nt.copy()    
    
 
    pad_list = []  
    result = []
    x, y, z, dimx, dimy, dimz, hx_, hy_, hz_, num_e_, It_, nt_ave = get_xyz_descriptors(n, hx, hy, hz, It, num_e)
    
    
    start = time.time()    
    
    print('\n\ngetting integration convolutions...')


    integration_0_2, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.2, accuracy = get_auto_accuracy(hx,hy,hz, 0.2))
    pad_list.append(temp_pad)
    print('done 0.2')

    
    integration_0_4, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.4, accuracy = get_auto_accuracy(hx,hy,hz, 0.4))
    pad_list.append(temp_pad)
    print('done 0.4')

  
    integration_0_6, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.6, accuracy = get_auto_accuracy(hx,hy,hz, 0.6))
    pad_list.append(temp_pad)
    print('done 0.6')


    
    integration_0_8, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.8, accuracy = get_auto_accuracy(hx,hy,hz, 0.8))
    pad_list.append(temp_pad)
    print('done 0.8')

    
    integration_1_0, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.0, accuracy = get_auto_accuracy(hx,hy,hz, 1.0))
    pad_list.append(temp_pad)
    print('done 1.0')


    
    integration_1_2, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.2, accuracy = get_auto_accuracy(hx,hy,hz, 1.2))
    pad_list.append(temp_pad)
    print('done 1.2')


    
    integration_1_4, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.4, accuracy = get_auto_accuracy(hx,hy,hz, 1.4))
    pad_list.append(temp_pad)
    print('done 1.4')

    
    integration_1_6, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.6, accuracy = get_auto_accuracy(hx,hy,hz, 1.6))
    pad_list.append(temp_pad)
    print('done 1.6')

    
    integration_1_8, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.8, accuracy = get_auto_accuracy(hx,hy,hz, 1.8))
    pad_list.append(temp_pad)
    print('done 1.8')

    
    integration_2_0, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 2.0, accuracy = get_auto_accuracy(hx,hy,hz, 2.0))
    pad_list.append(temp_pad)
    print('done 2.0')

    integration_2_2, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 2.2, accuracy = get_auto_accuracy(hx,hy,hz, 2.2))
    pad_list.append(temp_pad)
    print('done 2.2')
  
    integration_2_4, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 2.4, accuracy = get_auto_accuracy(hx,hy,hz, 2.4))
    pad_list.append(temp_pad)
    print('done 2.4')
    
    integration_2_6, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 2.6, accuracy = get_auto_accuracy(hx,hy,hz, 2.6))
    pad_list.append(temp_pad)
    print('done 2.6')
    
    integration_2_8, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 2.8, accuracy = get_auto_accuracy(hx,hy,hz, 2.8))
    pad_list.append(temp_pad)
    print('done 2.8')
    
    integration_3_0, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 3.0, accuracy = get_auto_accuracy(hx,hy,hz, 3.0))
    pad_list.append(temp_pad)
    print('done 3.0'                           )

    integration_3_2, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 3.2, accuracy = get_auto_accuracy(hx,hy,hz, 3.2))
    pad_list.append(temp_pad)
    print('done 1.2')
  
    integration_3_4, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 3.4, accuracy = get_auto_accuracy(hx,hy,hz, 3.4))
    pad_list.append(temp_pad)
    print('done 1.4')
    
    integration_3_6, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 3.6, accuracy = get_auto_accuracy(hx,hy,hz, 3.6))
    pad_list.append(temp_pad)
    print('done 1.6')
    
    integration_3_8, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 3.8, accuracy = get_auto_accuracy(hx,hy,hz, 3.8))
    pad_list.append(temp_pad)
    print('done 1.8')
    
    integration_4_0, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 4.0, accuracy = get_auto_accuracy(hx,hy,hz, 4.0))
    pad_list.append(temp_pad)
    print('done 2.0')

    integration_4_2, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 4.2, accuracy = get_auto_accuracy(hx,hy,hz, 4.2))
    pad_list.append(temp_pad)
    print('done 2.2')
  
    integration_4_4, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 4.4, accuracy = get_auto_accuracy(hx,hy,hz, 4.4))
    pad_list.append(temp_pad)
    print('done 2.4')
    
    integration_4_6, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 4.6, accuracy = get_auto_accuracy(hx,hy,hz, 4.6))
    pad_list.append(temp_pad)
    print('done 2.6')
    
    integration_4_8, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 4.8, accuracy = get_auto_accuracy(hx,hy,hz, 4.8))
    pad_list.append(temp_pad)
    print('done 2.8')
    
    integration_5_0, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 5.0, accuracy = get_auto_accuracy(hx,hy,hz, 5.0))
    pad_list.append(temp_pad)
    print('done 3.0'   )

    integration_5_2, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 5.2, accuracy = get_auto_accuracy(hx,hy,hz, 5.2))
    pad_list.append(temp_pad)
    print('done 2.2')
  
    integration_5_4, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 5.4, accuracy = get_auto_accuracy(hx,hy,hz, 5.4))
    pad_list.append(temp_pad)
    print('done 2.4')
    
    integration_5_6, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 5.6, accuracy = get_auto_accuracy(hx,hy,hz, 5.6))
    pad_list.append(temp_pad)
    print('done 2.6')
    
    integration_5_8, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 5.8, accuracy = get_auto_accuracy(hx,hy,hz, 5.8))
    pad_list.append(temp_pad)
    print('done 2.8')
    
    integration_6_0, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 6.0, accuracy = get_auto_accuracy(hx,hy,hz, 6.0))
    pad_list.append(temp_pad)
    print('done 3.0' )

    print("after first conv: " + str(time.time()-start))

    integration_homo_non_0_2 = get_homo_nondimensional(integration_0_2.copy(), nt.copy(), 0.2)
    integration_homo_non_0_4 = get_homo_nondimensional(integration_0_4.copy(), nt.copy(), 0.4)
    integration_homo_non_0_6 = get_homo_nondimensional(integration_0_6.copy(), nt.copy(), 0.6)
    integration_homo_non_0_8 = get_homo_nondimensional(integration_0_8.copy(), nt.copy(), 0.8)
    integration_homo_non_1_0 = get_homo_nondimensional(integration_1_0.copy(), nt.copy(), 1.0)
    integration_homo_non_1_2 = get_homo_nondimensional(integration_1_2.copy(), nt.copy(), 1.2)
    integration_homo_non_1_4 = get_homo_nondimensional(integration_1_4.copy(), nt.copy(), 1.4)
    integration_homo_non_1_6 = get_homo_nondimensional(integration_1_6.copy(), nt.copy(), 1.6)
    integration_homo_non_1_8 = get_homo_nondimensional(integration_1_8.copy(), nt.copy(), 1.8)
    integration_homo_non_2_0 = get_homo_nondimensional(integration_2_0.copy(), nt.copy(), 2.0)
    integration_homo_non_2_2 = get_homo_nondimensional(integration_2_2.copy(), nt.copy(), 2.2)
    integration_homo_non_2_4 = get_homo_nondimensional(integration_2_4.copy(), nt.copy(), 2.4)
    integration_homo_non_2_6 = get_homo_nondimensional(integration_2_6.copy(), nt.copy(), 2.6)
    integration_homo_non_2_8 = get_homo_nondimensional(integration_2_8.copy(), nt.copy(), 2.8)
    integration_homo_non_3_0 = get_homo_nondimensional(integration_3_0.copy(), nt.copy(), 3.0)
    integration_homo_non_3_2 = get_homo_nondimensional(integration_3_2.copy(), nt.copy(), 3.2)
    integration_homo_non_3_4 = get_homo_nondimensional(integration_3_4.copy(), nt.copy(), 3.4)
    integration_homo_non_3_6 = get_homo_nondimensional(integration_3_6.copy(), nt.copy(), 3.6)
    integration_homo_non_3_8 = get_homo_nondimensional(integration_3_8.copy(), nt.copy(), 3.8)
    integration_homo_non_4_0 = get_homo_nondimensional(integration_4_0.copy(), nt.copy(), 4.0)
    integration_homo_non_4_2 = get_homo_nondimensional(integration_4_2.copy(), nt.copy(), 4.2)
    integration_homo_non_4_4 = get_homo_nondimensional(integration_4_4.copy(), nt.copy(), 4.4)
    integration_homo_non_4_6 = get_homo_nondimensional(integration_4_6.copy(), nt.copy(), 4.6)
    integration_homo_non_4_8 = get_homo_nondimensional(integration_4_8.copy(), nt.copy(), 4.8)
    integration_homo_non_5_0 = get_homo_nondimensional(integration_5_0.copy(), nt.copy(), 5.0)
    integration_homo_non_5_2 = get_homo_nondimensional(integration_5_2.copy(), nt.copy(), 5.2)
    integration_homo_non_5_4 = get_homo_nondimensional(integration_5_4.copy(), nt.copy(), 5.4)
    integration_homo_non_5_6 = get_homo_nondimensional(integration_5_6.copy(), nt.copy(), 5.6)
    integration_homo_non_5_8 = get_homo_nondimensional(integration_5_8.copy(), nt.copy(), 5.8)
    integration_homo_non_6_0 = get_homo_nondimensional(integration_6_0.copy(), nt.copy(), 6.0)    

    
    integration_homo_non_0_2_ntave = get_homo_nondimensional_nave(integration_0_2.copy(), nt_ave, 0.2)
    integration_homo_non_0_4_ntave = get_homo_nondimensional_nave(integration_0_4.copy(), nt_ave, 0.4)
    integration_homo_non_0_6_ntave = get_homo_nondimensional_nave(integration_0_6.copy(), nt_ave, 0.6)
    integration_homo_non_0_8_ntave = get_homo_nondimensional_nave(integration_0_8.copy(), nt_ave, 0.8)
    integration_homo_non_1_0_ntave = get_homo_nondimensional_nave(integration_1_0.copy(), nt_ave, 1.0)
    integration_homo_non_1_2_ntave = get_homo_nondimensional_nave(integration_1_2.copy(), nt_ave, 1.2)
    integration_homo_non_1_4_ntave = get_homo_nondimensional_nave(integration_1_4.copy(), nt_ave, 1.4)
    integration_homo_non_1_6_ntave = get_homo_nondimensional_nave(integration_1_6.copy(), nt_ave, 1.6)
    integration_homo_non_1_8_ntave = get_homo_nondimensional_nave(integration_1_8.copy(), nt_ave, 1.8)
    integration_homo_non_2_0_ntave = get_homo_nondimensional_nave(integration_2_0.copy(), nt_ave, 2.0)
    integration_homo_non_2_2_ntave = get_homo_nondimensional_nave(integration_2_2.copy(), nt_ave, 2.2)
    integration_homo_non_2_4_ntave = get_homo_nondimensional_nave(integration_2_4.copy(), nt_ave, 2.4)
    integration_homo_non_2_6_ntave = get_homo_nondimensional_nave(integration_2_6.copy(), nt_ave, 2.6)
    integration_homo_non_2_8_ntave = get_homo_nondimensional_nave(integration_2_8.copy(), nt_ave, 2.8)
    integration_homo_non_3_0_ntave = get_homo_nondimensional_nave(integration_3_0.copy(), nt_ave, 3.0)
    integration_homo_non_3_2_ntave = get_homo_nondimensional_nave(integration_3_2.copy(), nt_ave, 3.2)
    integration_homo_non_3_4_ntave = get_homo_nondimensional_nave(integration_3_4.copy(), nt_ave, 3.4)
    integration_homo_non_3_6_ntave = get_homo_nondimensional_nave(integration_3_6.copy(), nt_ave, 3.6)
    integration_homo_non_3_8_ntave = get_homo_nondimensional_nave(integration_3_8.copy(), nt_ave, 3.8)
    integration_homo_non_4_0_ntave = get_homo_nondimensional_nave(integration_4_0.copy(), nt_ave, 4.0)
    integration_homo_non_4_2_ntave = get_homo_nondimensional_nave(integration_4_2.copy(), nt_ave, 4.2)
    integration_homo_non_4_4_ntave = get_homo_nondimensional_nave(integration_4_4.copy(), nt_ave, 4.4)
    integration_homo_non_4_6_ntave = get_homo_nondimensional_nave(integration_4_6.copy(), nt_ave, 4.6)
    integration_homo_non_4_8_ntave = get_homo_nondimensional_nave(integration_4_8.copy(), nt_ave, 4.8)
    integration_homo_non_5_0_ntave = get_homo_nondimensional_nave(integration_5_0.copy(), nt_ave, 5.0)
    integration_homo_non_5_2_ntave = get_homo_nondimensional_nave(integration_5_2.copy(), nt_ave, 5.2)
    integration_homo_non_5_4_ntave = get_homo_nondimensional_nave(integration_5_4.copy(), nt_ave, 5.4)
    integration_homo_non_5_6_ntave = get_homo_nondimensional_nave(integration_5_6.copy(), nt_ave, 5.6)
    integration_homo_non_5_8_ntave = get_homo_nondimensional_nave(integration_5_8.copy(), nt_ave, 5.8)
    integration_homo_non_6_0_ntave = get_homo_nondimensional_nave(integration_6_0.copy(), nt_ave, 6.0)

#    integration_h1s_non_0_2 = integration_0_2.copy() / get_h_1s_desity_integral(0.2)
#    integration_h1s_non_0_4 = integration_0_4.copy() / get_h_1s_desity_integral(0.4)
#    integration_h1s_non_0_6 = integration_0_6.copy() / get_h_1s_desity_integral(0.6)
#    integration_h1s_non_0_8 = integration_0_8.copy() / get_h_1s_desity_integral(0.8)
#    integration_h1s_non_1_0 = integration_1_0.copy() / get_h_1s_desity_integral(1.0)
#    
#    integration_h1s_non_1_2 = integration_1_2.copy() / get_h_1s_desity_integral(1.2)
#    integration_h1s_non_1_4 = integration_1_4.copy() / get_h_1s_desity_integral(1.4)
#    integration_h1s_non_1_6 = integration_1_6.copy() / get_h_1s_desity_integral(1.6)
#    integration_h1s_non_1_8 = integration_1_8.copy() / get_h_1s_desity_integral(1.8)
#    integration_h1s_non_2_0 = integration_2_0.copy() / get_h_1s_desity_integral(2.0)
#    
#    integration_h1s_non_2_2 = integration_2_2.copy() / get_h_1s_desity_integral(2.2)
#    integration_h1s_non_2_4 = integration_2_4.copy() / get_h_1s_desity_integral(2.4)
#    integration_h1s_non_2_6 = integration_2_6.copy() / get_h_1s_desity_integral(2.6)
#    integration_h1s_non_2_8 = integration_2_8.copy() / get_h_1s_desity_integral(2.8)
#    integration_h1s_non_3_0 = integration_3_0.copy() / get_h_1s_desity_integral(3.0)
#
#    integration_h1s_non_3_2 = integration_3_2.copy() / get_h_1s_desity_integral(3.2)
#    integration_h1s_non_3_4 = integration_3_4.copy() / get_h_1s_desity_integral(3.4)
#    integration_h1s_non_3_6 = integration_3_6.copy() / get_h_1s_desity_integral(3.6)
#    integration_h1s_non_3_8 = integration_3_8.copy() / get_h_1s_desity_integral(3.8)
#    integration_h1s_non_4_0 = integration_4_0.copy() / get_h_1s_desity_integral(4.0)
#    
#    integration_h1s_non_4_2 = integration_4_2.copy() / get_h_1s_desity_integral(4.2)
#    integration_h1s_non_4_4 = integration_4_4.copy() / get_h_1s_desity_integral(4.4)
#    integration_h1s_non_4_6 = integration_4_6.copy() / get_h_1s_desity_integral(4.6)
#    integration_h1s_non_4_8 = integration_4_8.copy() / get_h_1s_desity_integral(4.8)
#    integration_h1s_non_5_0 = integration_5_0.copy() / get_h_1s_desity_integral(5.0)
    print("after everything: " + str(time.time()-start)    )
    
    if periodic:
        result = zip(   x.flatten().tolist() ,\
                        y.flatten().tolist() ,\
                        z.flatten().tolist() ,\
                        dimx.flatten().tolist() ,\
                        dimy.flatten().tolist() ,\
                        dimz.flatten().tolist() ,\
                        hx_.flatten().tolist() ,\
                        hy_.flatten().tolist() ,\
                        hz_.flatten().tolist() ,\
                        It_.flatten().tolist() ,\
                        num_e_.flatten().tolist() ,\
                        nt_plot.flatten().tolist() ,\
                        n_plot.flatten().tolist() ,\
                        integration_0_2.flatten().tolist() ,\
                        integration_0_4.flatten().tolist() ,\
                        integration_0_6.flatten().tolist() ,\
                        integration_0_8.flatten().tolist() ,\
                        integration_1_0.flatten().tolist() ,\
                        integration_1_2.flatten().tolist() ,\
                        integration_1_4.flatten().tolist() ,\
                        integration_1_6.flatten().tolist() ,\
                        integration_1_8.flatten().tolist() ,\
                        integration_2_0.flatten().tolist() ,\
                        integration_2_2.flatten().tolist() ,\
                        integration_2_4.flatten().tolist() ,\
                        integration_2_6.flatten().tolist() ,\
                        integration_2_8.flatten().tolist() ,\
                        integration_3_0.flatten().tolist() ,\
                        integration_3_2.flatten().tolist() ,\
                        integration_3_4.flatten().tolist() ,\
                        integration_3_6.flatten().tolist() ,\
                        integration_3_8.flatten().tolist() ,\
                        integration_4_0.flatten().tolist() ,\
                        integration_4_2.flatten().tolist() ,\
                        integration_4_4.flatten().tolist() ,\
                        integration_4_6.flatten().tolist() ,\
                        integration_4_8.flatten().tolist() ,\
                        integration_5_0.flatten().tolist() ,\
                        integration_5_2.flatten().tolist() ,\
                        integration_5_4.flatten().tolist() ,\
                        integration_5_6.flatten().tolist() ,\
                        integration_5_8.flatten().tolist() ,\
                        integration_6_0.flatten().tolist() ,\
                        integration_homo_non_0_2.flatten().tolist() ,\
                        integration_homo_non_0_4.flatten().tolist() ,\
                        integration_homo_non_0_6.flatten().tolist() ,\
                        integration_homo_non_0_8.flatten().tolist() ,\
                        integration_homo_non_1_0.flatten().tolist() ,\
                        integration_homo_non_1_2.flatten().tolist() ,\
                        integration_homo_non_1_4.flatten().tolist() ,\
                        integration_homo_non_1_6.flatten().tolist() ,\
                        integration_homo_non_1_8.flatten().tolist() ,\
                        integration_homo_non_2_0.flatten().tolist() ,\
                        integration_homo_non_2_2.flatten().tolist() ,\
                        integration_homo_non_2_4.flatten().tolist() ,\
                        integration_homo_non_2_6.flatten().tolist() ,\
                        integration_homo_non_2_8.flatten().tolist() ,\
                        integration_homo_non_3_0.flatten().tolist() ,\
                        integration_homo_non_3_2.flatten().tolist() ,\
                        integration_homo_non_3_4.flatten().tolist() ,\
                        integration_homo_non_3_6.flatten().tolist() ,\
                        integration_homo_non_3_8.flatten().tolist() ,\
                        integration_homo_non_4_0.flatten().tolist() ,\
                        integration_homo_non_4_2.flatten().tolist() ,\
                        integration_homo_non_4_4.flatten().tolist() ,\
                        integration_homo_non_4_6.flatten().tolist() ,\
                        integration_homo_non_4_8.flatten().tolist() ,\
                        integration_homo_non_5_0.flatten().tolist() ,\
                        integration_homo_non_5_2.flatten().tolist() ,\
                        integration_homo_non_5_4.flatten().tolist() ,\
                        integration_homo_non_5_6.flatten().tolist() ,\
                        integration_homo_non_5_8.flatten().tolist() ,\
                        integration_homo_non_6_0.flatten().tolist() ,\
                        integration_homo_non_0_2_ntave.flatten().tolist() ,\
                        integration_homo_non_0_4_ntave.flatten().tolist() ,\
                        integration_homo_non_0_6_ntave.flatten().tolist() ,\
                        integration_homo_non_0_8_ntave.flatten().tolist() ,\
                        integration_homo_non_1_0_ntave.flatten().tolist() ,\
                        integration_homo_non_1_2_ntave.flatten().tolist() ,\
                        integration_homo_non_1_4_ntave.flatten().tolist() ,\
                        integration_homo_non_1_6_ntave.flatten().tolist() ,\
                        integration_homo_non_1_8_ntave.flatten().tolist() ,\
                        integration_homo_non_2_0_ntave.flatten().tolist() ,\
                        integration_homo_non_2_2_ntave.flatten().tolist() ,\
                        integration_homo_non_2_4_ntave.flatten().tolist() ,\
                        integration_homo_non_2_6_ntave.flatten().tolist() ,\
                        integration_homo_non_2_8_ntave.flatten().tolist() ,\
                        integration_homo_non_3_0_ntave.flatten().tolist() ,\
                        integration_homo_non_3_2_ntave.flatten().tolist() ,\
                        integration_homo_non_3_4_ntave.flatten().tolist() ,\
                        integration_homo_non_3_6_ntave.flatten().tolist() ,\
                        integration_homo_non_3_8_ntave.flatten().tolist() ,\
                        integration_homo_non_4_0_ntave.flatten().tolist() ,\
                        integration_homo_non_4_2_ntave.flatten().tolist() ,\
                        integration_homo_non_4_4_ntave.flatten().tolist() ,\
                        integration_homo_non_4_6_ntave.flatten().tolist() ,\
                        integration_homo_non_4_8_ntave.flatten().tolist() ,\
                        integration_homo_non_5_0_ntave.flatten().tolist() ,\
                        integration_homo_non_5_2_ntave.flatten().tolist() ,\
                        integration_homo_non_5_4_ntave.flatten().tolist() ,\
                        integration_homo_non_5_6_ntave.flatten().tolist() ,\
                        integration_homo_non_5_8_ntave.flatten().tolist() ,\
                        integration_homo_non_6_0_ntave.flatten().tolist())
    else:
        padx, pady, padz = get_pads(pad_list)
  
        result = zip(   x[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        y[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        z[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        dimx[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        dimy[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        dimz[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        hx_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        hy_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        hz_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        It_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        num_e_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        nt_plot[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        n_plot[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_0_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_0_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_0_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_0_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_1_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_1_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_1_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_1_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_1_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_2_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_2_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_2_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_2_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_2_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_3_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_3_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_3_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_3_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_3_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_4_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_4_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_4_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_4_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_4_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_5_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_5_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_5_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_5_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_5_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_6_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_1_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_1_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_1_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_1_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_1_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_2_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_2_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_2_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_2_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_2_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_3_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_3_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_3_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_3_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_3_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_4_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_4_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_4_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_4_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_4_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_5_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_5_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_5_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_5_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_5_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_6_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_2_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_4_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_6_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_0_8_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_1_0_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_1_2_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_1_4_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_1_6_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_1_8_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_2_0_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_2_2_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_2_4_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_2_6_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_2_8_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_3_0_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_3_2_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_3_4_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_3_6_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_3_8_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_4_0_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_4_2_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_4_4_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_4_6_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_4_8_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_5_0_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_5_2_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_5_4_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_5_6_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_5_8_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_homo_non_6_0_ntave[padx:-padx,pady:-pady,padz:-padz].flatten().tolist())
#                    integration_h1s_non_0_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_0_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_0_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_0_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_1_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_1_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_1_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_1_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_1_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_2_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_2_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_2_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_2_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_2_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_3_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_3_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_3_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_3_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_3_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_4_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_4_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_4_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_4_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_4_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                    integration_h1s_non_5_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist())
    print("after zip: " + str(time.time()-start))
#    print('\n\n\n######\nafter convolution')
#    print(type(result))
#    print(len(result))
#    print(result[0])
    return result


def get_discriptors_from_density(n,list_stencils,pad = (0,0,0)):
    
    padx = pad[0]
    pady = pad[1]
    padz = pad[2]
    result = []
    result.append(n[padx:-padx,pady:-pady,padz:-padz])
        
    for stencil in list_stencils:
        temp = convolve(n,stencil)
        result.append(temp[padx:-padx,pady:-pady,padz:-padz])
    
    return result


def delete_vacuum_descriptors_generic(list_descriptors,cutoff = 1e-3):
    print('\n\n\nlength before')
    print(len(list_descriptors))
 
    result = []    
    for discriptors in list_descriptors:
        if max(discriptors[10:]) > cutoff:
            result.append(discriptors)
    
    print('\n\n\nlength after')
    print(len(result))

    return result

def gpaw_to_density(filename):
    '''
    read gpaw file and output the density, and the grid spacing at each dimensions
    '''

#filename = sys.argv[1]

    atoms,calc=restart(filename)
    
    ham =calc.hamiltonian
    h = ham.finegd.get_grid_spacings() 
    hx = h[0]
    hy = h[1]
    hz = h[2]
    wfs = calc.wfs
    dens = calc.density
    
    occ = calc.occupations
    gd = ham.finegd
    
    atoms.set_calculator(calc)
    diff = calc.get_xc_difference('pyPBE')
        
    n_sg = dens.nt_sg
    nt = n_sg[0]
    
    gridref = 2
    n = calc.get_all_electron_density(gridrefinement=gridref)
    dv = atoms.get_volume() / calc.get_number_of_grid_points().prod()
    dvt = hx*hy*hz
#    It = nt.sum() * dv
    It = np.sum(nt) * dvt
    num_e = n.sum() * dv / gridref**3
    print('\n\n\n\n!!!!!size of density')
    print(n.shape)
    
    print(num_e)
    
    return nt, hx, hy, hz, n, It, num_e
    
def gpaw_to_descriptors_generic(filename, Periodic =False ):
    
    nt, hx, hy, hz, n, It, num_e = gpaw_to_density(filename)
    list_of_discriptors = get_discriptors_from_density_generic(nt, hx, hy, hz, n, It, num_e, periodic = Periodic)
    
    return list_of_discriptors

def gpaw_to_descriptors_integral(filename, Periodic =False ):
    
    nt, hx, hy, hz, n, It, num_e = gpaw_to_density(filename)
    list_of_discriptors = get_discriptors_from_density_integral(nt, hx, hy, hz, n, It, num_e, periodic = Periodic)
    
    return list_of_discriptors
    

















def get_density_derivative_central(density1, density3, dr):
    temp1 = np.multiply(density1, -0.5)
    temp2 = np.multiply(density3, 0.5)
    return np.divide(np.add(temp1, temp2) , dr)

def get_density_derivative_forward(density2, density3, dr):
    temp1 = np.multiply(density2, -0.5)
    temp2 = np.multiply(density3, 0.5)
    return np.divide(np.add(temp1, temp2) , dr)

def get_density_derivative_back(density1, density2, dr):
    temp1 = np.multiply(density1, -0.5)
    temp2 = np.multiply(density2, 0.5)
    return np.divide(np.add(temp1, temp2) , dr)

    

def get_discriptors_from_density_integral_derivative(nt,hx,hy,hz, n, It, num_e, periodic = False, integral_accuracy = 4):
    '''
    get the first-fourth derivative from the density matrix using convolution
    get the integration convolution at 0.5, 1.0, 1.5, 2.0, 2.5
    '''
    def get_xyz_descriptors(nt, hx, hy, hz, It, num_e):

        dimx = np.ones(nt.shape)*nt.shape[0]
        dimy = np.ones(nt.shape)*nt.shape[1]
        dimz = np.ones(nt.shape)*nt.shape[2]
      
        nt_ave = It / float(hx*hy*hz) 
                
        hx_ = np.ones_like(nt)*hx
        hy_ = np.ones_like(nt)*hy
        hz_ = np.ones_like(nt)*hz
        num_e_ = np.ones_like(nt)*float(num_e)
        It_ = np.ones_like(nt)*float(It)
             
        x = np.ones(nt.shape)
        y = np.ones(nt.shape)
        z = np.ones(nt.shape)
        
        for index, density in np.ndenumerate(nt):
            x[index[0]][index[1]][index[2]] = index[0]
            y[index[0]][index[1]][index[2]] = index[1]
            z[index[0]][index[1]][index[2]] = index[2]
        
        return x, y, z, dimx, dimy, dimz, hx_, hy_, hz_, num_e_, It_, nt_ave
#    
#    def get_auto_accuracy(hx,hy,hz, r):
#        h = max([hx,hy,hz])
#        temp = 5 - int(math.floor((r/h)/3.))
#        if temp < 1:
#            return 1
#        else:
#            return temp
    
    n_plot = n.copy()
    nt_plot = nt.copy()    
    
 
    pad_list = []  
    result = []
    x, y, z, dimx, dimy, dimz, hx_, hy_, hz_, num_e_, It_, nt_ave = get_xyz_descriptors(n, hx, hy, hz, It, num_e)
    
    
    start = time.time()    
    
    print('\n\ngetting integration convolutions...')
    try:
        if max(hx, hy, hz) - min(hx, hy, hz) >= 0.001:
            raise NotImplementedError
        stencil_data = read_integration_stencil_file(hx, hy, hz)
        integration_0_05, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 0.05, stencil_data[str(0.05).replace(".","_")], stencil_data[str(0.05).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_0_1, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 0.1, stencil_data[str(0.1).replace(".","_")], stencil_data[str(0.1).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_0_15, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 0.15, stencil_data[str(0.15).replace(".","_")], stencil_data[str(0.15).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_0_2, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 0.2, stencil_data[str(0.2).replace(".","_")], stencil_data[str(0.2).replace(".","_") + "pad"])
        pad_list.append(temp_pad)

        integration_0_25, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 0.25, stencil_data[str(0.25).replace(".","_")], stencil_data[str(0.25).replace(".","_") + "pad"])
        pad_list.append(temp_pad)

        integration_0_3, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 0.3, stencil_data[str(0.3).replace(".","_")], stencil_data[str(0.3).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_0_35, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 0.35, stencil_data[str(0.35).replace(".","_")], stencil_data[str(0.35).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_0_4, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 0.4, stencil_data[str(0.4).replace(".","_")], stencil_data[str(0.4).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_0_45, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 0.45, stencil_data[str(0.45).replace(".","_")], stencil_data[str(0.45).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_0_5, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 0.5, stencil_data[str(0.5).replace(".","_")], stencil_data[str(0.5).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_0_6, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 0.6, stencil_data[str(0.6).replace(".","_")], stencil_data[str(0.6).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_0_7, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 0.7, stencil_data[str(0.7).replace(".","_")], stencil_data[str(0.7).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_0_8, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 0.8, stencil_data[str(0.8).replace(".","_")], stencil_data[str(0.8).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_0_9, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 0.9, stencil_data[str(0.9).replace(".","_")], stencil_data[str(0.9).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_1_0, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 1.0, stencil_data[str(1.0).replace(".","_")], stencil_data[str(1.0).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_1_1, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 1.1, stencil_data[str(1.1).replace(".","_")], stencil_data[str(1.1).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_1_2, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 1.2, stencil_data[str(1.2).replace(".","_")], stencil_data[str(1.2).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_1_3, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 1.3, stencil_data[str(1.3).replace(".","_")], stencil_data[str(1.3).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_1_4, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 1.4, stencil_data[str(1.4).replace(".","_")], stencil_data[str(1.4).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_1_5, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 1.5, stencil_data[str(1.5).replace(".","_")], stencil_data[str(1.5).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_1_6, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 1.6, stencil_data[str(1.6).replace(".","_")], stencil_data[str(1.6).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_1_7, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 1.7, stencil_data[str(1.7).replace(".","_")], stencil_data[str(1.7).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_1_8, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 1.8, stencil_data[str(1.8).replace(".","_")], stencil_data[str(1.8).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_1_9, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 1.9, stencil_data[str(1.9).replace(".","_")], stencil_data[str(1.9).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_2_0, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 2.0, stencil_data[str(2.0).replace(".","_")], stencil_data[str(2.0).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_2_2, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 2.2, stencil_data[str(2.2).replace(".","_")], stencil_data[str(2.2).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_2_4, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 2.4, stencil_data[str(2.4).replace(".","_")], stencil_data[str(2.4).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_2_6, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 2.6, stencil_data[str(2.6).replace(".","_")], stencil_data[str(2.6).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_2_8, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 2.8, stencil_data[str(2.8).replace(".","_")], stencil_data[str(2.8).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_3_0, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 3.0, stencil_data[str(3.0).replace(".","_")], stencil_data[str(3.0).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_3_2, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 3.2, stencil_data[str(3.2).replace(".","_")], stencil_data[str(3.2).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_3_4, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 3.4, stencil_data[str(3.4).replace(".","_")], stencil_data[str(3.4).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_3_6, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 3.6, stencil_data[str(3.6).replace(".","_")], stencil_data[str(3.6).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_3_8, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 3.8, stencil_data[str(3.8).replace(".","_")], stencil_data[str(3.8).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_4_0, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 4.0, stencil_data[str(4.0).replace(".","_")], stencil_data[str(4.0).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_4_2, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 4.2, stencil_data[str(4.2).replace(".","_")], stencil_data[str(4.2).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_4_4, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 4.4, stencil_data[str(4.4).replace(".","_")], stencil_data[str(4.4).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_4_6, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 4.6, stencil_data[str(4.6).replace(".","_")], stencil_data[str(4.6).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_4_8, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 4.8, stencil_data[str(4.8).replace(".","_")], stencil_data[str(4.8).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_5_0, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 5.0, stencil_data[str(5.0).replace(".","_")], stencil_data[str(5.0).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_5_2, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 5.2, stencil_data[str(5.2).replace(".","_")], stencil_data[str(5.2).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_5_4, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 5.4, stencil_data[str(5.4).replace(".","_")], stencil_data[str(5.4).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_5_6, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 5.6, stencil_data[str(5.6).replace(".","_")], stencil_data[str(5.6).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_5_8, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 5.8, stencil_data[str(5.8).replace(".","_")], stencil_data[str(5.8).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_6_0, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 6.0, stencil_data[str(6.0).replace(".","_")], stencil_data[str(6.0).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        
        
    except:
        integration_0_05, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.05, accuracy = get_auto_accuracy(hx,hy,hz, 0.05))
        pad_list.append(temp_pad)
        print('done 0.2')
    
        integration_0_1, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.1, accuracy = get_auto_accuracy(hx,hy,hz, 0.1))
        pad_list.append(temp_pad)
        print('done 0.2')
    
        integration_0_15, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.15, accuracy = get_auto_accuracy(hx,hy,hz, 0.15))
        pad_list.append(temp_pad)
        print('done 0.2')
    
        integration_0_2, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.2, accuracy = get_auto_accuracy(hx,hy,hz, 0.2))
        pad_list.append(temp_pad)
        print('done 0.2')
    
        integration_0_25, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.25, accuracy = get_auto_accuracy(hx,hy,hz, 0.25))
        pad_list.append(temp_pad)
        print('done 0.2')
    
        integration_0_3, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.3, accuracy = get_auto_accuracy(hx,hy,hz, 0.3))
        pad_list.append(temp_pad)
        print('done 0.2')
    
        integration_0_35, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.35, accuracy = get_auto_accuracy(hx,hy,hz, 0.35))
        pad_list.append(temp_pad)
        print('done 0.2')
        
        integration_0_4, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.4, accuracy = get_auto_accuracy(hx,hy,hz, 0.4))
        pad_list.append(temp_pad)
        print('done 0.4')
    
        integration_0_45, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.45, accuracy = get_auto_accuracy(hx,hy,hz, 0.45))
        pad_list.append(temp_pad)
        print('done 0.2')
    
        integration_0_5, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.5, accuracy = get_auto_accuracy(hx,hy,hz, 0.5))
        pad_list.append(temp_pad)
        print('done 0.2')
      
        integration_0_6, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.6, accuracy = get_auto_accuracy(hx,hy,hz, 0.6))
        pad_list.append(temp_pad)
        print('done 0.6')
    
        integration_0_7, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.7, accuracy = get_auto_accuracy(hx,hy,hz, 0.7))
        pad_list.append(temp_pad)
        print('done 0.8')
        
        integration_0_8, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.8, accuracy = get_auto_accuracy(hx,hy,hz, 0.8))
        pad_list.append(temp_pad)
        print('done 0.8')
    
        integration_0_9, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.9, accuracy = get_auto_accuracy(hx,hy,hz, 0.9))
        pad_list.append(temp_pad)
        print('done 0.8')
        
        integration_1_0, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.0, accuracy = get_auto_accuracy(hx,hy,hz, 1.0))
        pad_list.append(temp_pad)
        print('done 1.0')
    
        integration_1_1, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.1, accuracy = get_auto_accuracy(hx,hy,hz, 1.1))
        pad_list.append(temp_pad)
        print('done 0.8')
        
        integration_1_2, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.2, accuracy = get_auto_accuracy(hx,hy,hz, 1.2))
        pad_list.append(temp_pad)
        print('done 1.2')
    
        integration_1_3, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.3, accuracy = get_auto_accuracy(hx,hy,hz, 1.3))
        pad_list.append(temp_pad)
        print('done 0.8')
        
        integration_1_4, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.4, accuracy = get_auto_accuracy(hx,hy,hz, 1.4))
        pad_list.append(temp_pad)
        print('done 1.4')
    
        integration_1_5, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.5, accuracy = get_auto_accuracy(hx,hy,hz, 1.5))
        pad_list.append(temp_pad)
        print('done 0.8')
        
        integration_1_6, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.6, accuracy = get_auto_accuracy(hx,hy,hz, 1.6))
        pad_list.append(temp_pad)
        print('done 1.6')
    
        integration_1_7, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.7, accuracy = get_auto_accuracy(hx,hy,hz, 1.7))
        pad_list.append(temp_pad)
        print('done 1.8')
        
        integration_1_8, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.8, accuracy = get_auto_accuracy(hx,hy,hz, 1.8))
        pad_list.append(temp_pad)
        print('done 1.8')
    
        integration_1_9, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.9, accuracy = get_auto_accuracy(hx,hy,hz, 1.9))
        pad_list.append(temp_pad)
        print('done 1.8')
        
        integration_2_0, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 2.0, accuracy = get_auto_accuracy(hx,hy,hz, 2.0))
        pad_list.append(temp_pad)
        print('done 2.0')
    
        integration_2_2, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 2.2, accuracy = get_auto_accuracy(hx,hy,hz, 2.2))
        pad_list.append(temp_pad)
        print('done 2.2')
      
        integration_2_4, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 2.4, accuracy = get_auto_accuracy(hx,hy,hz, 2.4))
        pad_list.append(temp_pad)
        print('done 2.4')
        
        integration_2_6, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 2.6, accuracy = get_auto_accuracy(hx,hy,hz, 2.6))
        pad_list.append(temp_pad)
        print('done 2.6')
        
        integration_2_8, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 2.8, accuracy = get_auto_accuracy(hx,hy,hz, 2.8))
        pad_list.append(temp_pad)
        print('done 2.8')
        
        integration_3_0, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 3.0, accuracy = get_auto_accuracy(hx,hy,hz, 3.0))
        pad_list.append(temp_pad)
        print('done 3.0'                           )
    
        integration_3_2, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 3.2, accuracy = get_auto_accuracy(hx,hy,hz, 3.2))
        pad_list.append(temp_pad)
        print('done 1.2')
      
        integration_3_4, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 3.4, accuracy = get_auto_accuracy(hx,hy,hz, 3.4))
        pad_list.append(temp_pad)
        print('done 1.4')
        
        integration_3_6, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 3.6, accuracy = get_auto_accuracy(hx,hy,hz, 3.6))
        pad_list.append(temp_pad)
        print('done 1.6')
        
        integration_3_8, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 3.8, accuracy = get_auto_accuracy(hx,hy,hz, 3.8))
        pad_list.append(temp_pad)
        print('done 1.8')
        
        integration_4_0, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 4.0, accuracy = get_auto_accuracy(hx,hy,hz, 4.0))
        pad_list.append(temp_pad)
        print('done 2.0')
    
        integration_4_2, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 4.2, accuracy = get_auto_accuracy(hx,hy,hz, 4.2))
        pad_list.append(temp_pad)
        print('done 2.2')
      
        integration_4_4, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 4.4, accuracy = get_auto_accuracy(hx,hy,hz, 4.4))
        pad_list.append(temp_pad)
        print('done 2.4')
        
        integration_4_6, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 4.6, accuracy = get_auto_accuracy(hx,hy,hz, 4.6))
        pad_list.append(temp_pad)
        print('done 2.6')
        
        integration_4_8, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 4.8, accuracy = get_auto_accuracy(hx,hy,hz, 4.8))
        pad_list.append(temp_pad)
        print('done 2.8')
        
        integration_5_0, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 5.0, accuracy = get_auto_accuracy(hx,hy,hz, 5.0))
        pad_list.append(temp_pad)
        print('done 3.0'   )
    
        integration_5_2, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 5.2, accuracy = get_auto_accuracy(hx,hy,hz, 5.2))
        pad_list.append(temp_pad)
        print('done 2.2')
      
        integration_5_4, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 5.4, accuracy = get_auto_accuracy(hx,hy,hz, 5.4))
        pad_list.append(temp_pad)
        print('done 2.4')
        
        integration_5_6, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 5.6, accuracy = get_auto_accuracy(hx,hy,hz, 5.6))
        pad_list.append(temp_pad)
        print('done 2.6')
        
        integration_5_8, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 5.8, accuracy = get_auto_accuracy(hx,hy,hz, 5.8))
        pad_list.append(temp_pad)
        print('done 2.8')
        
        integration_6_0, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 6.0, accuracy = get_auto_accuracy(hx,hy,hz, 6.0))
        pad_list.append(temp_pad)
        print('done 3.0' )

    print("after first conv: " + str(time.time()-start))
    
    ave_density_0_00_norm = nt.copy() 
    ave_density_0_05_norm = get_homo_nondimensional_nave(integration_0_05, 1.0, 0.05)
    ave_density_0_10_norm = get_homo_nondimensional_nave(integration_0_1, 1.0, 0.10)
    ave_density_0_15_norm = get_homo_nondimensional_nave(integration_0_15, 1.0, 0.15)
    ave_density_0_20_norm = get_homo_nondimensional_nave(integration_0_2, 1.0, 0.20)
    ave_density_0_25_norm = get_homo_nondimensional_nave(integration_0_25, 1.0, 0.25)
    ave_density_0_30_norm = get_homo_nondimensional_nave(integration_0_3, 1.0, 0.30)
    ave_density_0_35_norm = get_homo_nondimensional_nave(integration_0_35, 1.0, 0.35)
    ave_density_0_40_norm = get_homo_nondimensional_nave(integration_0_4, 1.0, 0.40)
    ave_density_0_45_norm = get_homo_nondimensional_nave(integration_0_45, 1.0, 0.45)
    ave_density_0_5_norm = get_homo_nondimensional_nave(integration_0_5, 1.0, 0.50)
    ave_density_0_6_norm = get_homo_nondimensional_nave(integration_0_6, 1.0, 0.60)
    ave_density_0_7_norm = get_homo_nondimensional_nave(integration_0_7, 1.0, 0.70)
    ave_density_0_8_norm = get_homo_nondimensional_nave(integration_0_8, 1.0, 0.80)
    ave_density_0_9_norm = get_homo_nondimensional_nave(integration_0_9, 1.0, 0.90)
    ave_density_1_0_norm = get_homo_nondimensional_nave(integration_1_0, 1.0, 1.0)
    ave_density_1_1_norm = get_homo_nondimensional_nave(integration_1_1, 1.0, 1.1)
    ave_density_1_2_norm = get_homo_nondimensional_nave(integration_1_2, 1.0, 1.2)
    ave_density_1_3_norm = get_homo_nondimensional_nave(integration_1_3, 1.0, 1.3)
    ave_density_1_4_norm = get_homo_nondimensional_nave(integration_1_4, 1.0, 1.4)
    ave_density_1_5_norm = get_homo_nondimensional_nave(integration_1_5, 1.0, 1.5)
    ave_density_1_6_norm = get_homo_nondimensional_nave(integration_1_6, 1.0, 1.6)
    ave_density_1_7_norm = get_homo_nondimensional_nave(integration_1_7, 1.0, 1.7)
    ave_density_1_8_norm = get_homo_nondimensional_nave(integration_1_8, 1.0, 1.8)
    ave_density_1_9_norm = get_homo_nondimensional_nave(integration_1_9, 1.0, 1.9)
    ave_density_2_0_norm = get_homo_nondimensional_nave(integration_2_0, 1.0, 2.0)
    ave_density_2_2_norm = get_homo_nondimensional_nave(integration_2_2, 1.0, 2.2)
    ave_density_2_4_norm = get_homo_nondimensional_nave(integration_2_4, 1.0, 2.4)
    ave_density_2_6_norm = get_homo_nondimensional_nave(integration_2_6, 1.0, 2.6)
    ave_density_2_8_norm = get_homo_nondimensional_nave(integration_2_8, 1.0, 2.8)
    
    ave_density_3_0_norm = get_homo_nondimensional_nave(integration_3_0, 1.0, 3.0)
    ave_density_3_2_norm = get_homo_nondimensional_nave(integration_3_2, 1.0, 3.2)
    ave_density_3_4_norm = get_homo_nondimensional_nave(integration_3_4, 1.0, 3.4)
    ave_density_3_6_norm = get_homo_nondimensional_nave(integration_3_6, 1.0, 3.6)
    ave_density_3_8_norm = get_homo_nondimensional_nave(integration_3_8, 1.0, 3.8)
    
    ave_density_4_0_norm = get_homo_nondimensional_nave(integration_4_0, 1.0, 4.0)
    ave_density_4_2_norm = get_homo_nondimensional_nave(integration_4_2, 1.0, 4.2)
    ave_density_4_4_norm = get_homo_nondimensional_nave(integration_4_4, 1.0, 4.4)
    ave_density_4_6_norm = get_homo_nondimensional_nave(integration_4_6, 1.0, 4.6)
    ave_density_4_8_norm = get_homo_nondimensional_nave(integration_4_8, 1.0, 4.8)
    
    ave_density_5_0_norm = get_homo_nondimensional_nave(integration_5_0, 1.0, 5.0)
    ave_density_5_2_norm = get_homo_nondimensional_nave(integration_5_2, 1.0, 5.2)
    ave_density_5_4_norm = get_homo_nondimensional_nave(integration_5_4, 1.0, 5.4)
    ave_density_5_6_norm = get_homo_nondimensional_nave(integration_5_6, 1.0, 5.6)
    ave_density_5_8_norm = get_homo_nondimensional_nave(integration_5_8, 1.0, 5.8)
    ave_density_6_0_norm = get_homo_nondimensional_nave(integration_6_0, 1.0, 6.0)

    
    
    integration_derivative_0_0 = get_density_derivative_forward(ave_density_0_00_norm, ave_density_0_05_norm, 0.05)
    integration_derivative_0_05 = get_density_derivative_central(ave_density_0_00_norm, ave_density_0_10_norm, 0.05)
    integration_derivative_0_1 = get_density_derivative_central(ave_density_0_05_norm, ave_density_0_15_norm, 0.05)
    integration_derivative_0_15 = get_density_derivative_central(ave_density_0_10_norm, ave_density_0_20_norm, 0.05)
    integration_derivative_0_2 = get_density_derivative_central(ave_density_0_15_norm, ave_density_0_25_norm, 0.05)
    integration_derivative_0_25 = get_density_derivative_central(ave_density_0_20_norm, ave_density_0_30_norm, 0.05)
    integration_derivative_0_3 = get_density_derivative_central(ave_density_0_25_norm, ave_density_0_35_norm, 0.05)
    integration_derivative_0_35 = get_density_derivative_central(ave_density_0_30_norm, ave_density_0_40_norm, 0.05)
    integration_derivative_0_4 = get_density_derivative_central(ave_density_0_35_norm, ave_density_0_45_norm, 0.05)
    integration_derivative_0_45 = get_density_derivative_central(ave_density_0_40_norm, ave_density_0_5_norm, 0.05)
    integration_derivative_0_5 = get_density_derivative_central(ave_density_0_40_norm, ave_density_0_6_norm, 0.1)
    integration_derivative_0_6 = get_density_derivative_central(ave_density_0_5_norm, ave_density_0_7_norm, 0.1)
    integration_derivative_0_7 = get_density_derivative_central(ave_density_0_6_norm, ave_density_0_8_norm, 0.1)
    integration_derivative_0_8 = get_density_derivative_central(ave_density_0_7_norm, ave_density_0_9_norm, 0.1)
    integration_derivative_0_9 = get_density_derivative_central(ave_density_0_8_norm, ave_density_1_0_norm, 0.1)
    integration_derivative_1_0 = get_density_derivative_central(ave_density_0_9_norm, ave_density_1_1_norm, 0.1)
    integration_derivative_1_1 = get_density_derivative_central(ave_density_1_0_norm, ave_density_1_2_norm, 0.1)
    integration_derivative_1_2 = get_density_derivative_central(ave_density_1_1_norm, ave_density_1_3_norm, 0.1)
    integration_derivative_1_3 = get_density_derivative_central(ave_density_1_2_norm, ave_density_1_4_norm, 0.1)
    integration_derivative_1_4 = get_density_derivative_central(ave_density_1_3_norm, ave_density_1_5_norm, 0.1)
    integration_derivative_1_5 = get_density_derivative_central(ave_density_1_4_norm, ave_density_1_6_norm, 0.1)
    integration_derivative_1_6 = get_density_derivative_central(ave_density_1_5_norm, ave_density_1_7_norm, 0.1)
    integration_derivative_1_7 = get_density_derivative_central(ave_density_1_6_norm, ave_density_1_8_norm, 0.1)
    integration_derivative_1_8 = get_density_derivative_central(ave_density_1_7_norm, ave_density_1_9_norm, 0.1)
    integration_derivative_1_9 = get_density_derivative_central(ave_density_1_8_norm, ave_density_2_0_norm, 0.1)
    integration_derivative_2_0 = get_density_derivative_central(ave_density_1_8_norm, ave_density_2_2_norm, 0.2)
    integration_derivative_2_2 = get_density_derivative_central(ave_density_2_0_norm, ave_density_2_4_norm, 0.2)
    integration_derivative_2_4 = get_density_derivative_central(ave_density_2_2_norm, ave_density_2_6_norm, 0.2)
    integration_derivative_2_6 = get_density_derivative_central(ave_density_2_4_norm, ave_density_2_8_norm, 0.2)
    integration_derivative_2_8 = get_density_derivative_central(ave_density_2_6_norm, ave_density_3_0_norm, 0.2)
    integration_derivative_3_0 = get_density_derivative_central(ave_density_2_8_norm, ave_density_3_2_norm, 0.2)
    integration_derivative_3_2 = get_density_derivative_central(ave_density_3_0_norm, ave_density_3_4_norm, 0.2)
    integration_derivative_3_4 = get_density_derivative_central(ave_density_3_2_norm, ave_density_3_6_norm, 0.2)
    integration_derivative_3_6 = get_density_derivative_central(ave_density_3_4_norm, ave_density_3_8_norm, 0.2)
    integration_derivative_3_8 = get_density_derivative_central(ave_density_3_6_norm, ave_density_4_0_norm, 0.2)
    integration_derivative_4_0 = get_density_derivative_central(ave_density_3_8_norm, ave_density_4_2_norm, 0.2)
    integration_derivative_4_2 = get_density_derivative_central(ave_density_4_0_norm, ave_density_4_4_norm, 0.2)
    integration_derivative_4_4 = get_density_derivative_central(ave_density_4_2_norm, ave_density_4_6_norm, 0.2)
    integration_derivative_4_6 = get_density_derivative_central(ave_density_4_4_norm, ave_density_4_8_norm, 0.2)
    integration_derivative_4_8 = get_density_derivative_central(ave_density_4_6_norm, ave_density_5_0_norm, 0.2)
    integration_derivative_5_0 = get_density_derivative_central(ave_density_4_8_norm, ave_density_5_2_norm, 0.2)
    integration_derivative_5_2 = get_density_derivative_central(ave_density_5_0_norm, ave_density_5_4_norm, 0.2)
    integration_derivative_5_4 = get_density_derivative_central(ave_density_5_2_norm, ave_density_5_6_norm, 0.2)
    integration_derivative_5_6 = get_density_derivative_central(ave_density_5_4_norm, ave_density_5_8_norm, 0.2)
    integration_derivative_5_8 = get_density_derivative_central(ave_density_5_6_norm, ave_density_6_0_norm, 0.2)
    integration_derivative_6_0 = get_density_derivative_back(ave_density_5_8_norm, ave_density_6_0_norm, 0.2)


    print("after everything: " + str(time.time()-start)    )
    
    if periodic:
        result = zip(   x.flatten().tolist() ,\
                        y.flatten().tolist() ,\
                        z.flatten().tolist() ,\
                        dimx.flatten().tolist() ,\
                        dimy.flatten().tolist() ,\
                        dimz.flatten().tolist() ,\
                        hx_.flatten().tolist() ,\
                        hy_.flatten().tolist() ,\
                        hz_.flatten().tolist() ,\
                        It_.flatten().tolist() ,\
                        num_e_.flatten().tolist() ,\
                        nt_plot.flatten().tolist() ,\
                        n_plot.flatten().tolist() ,\
                        integration_derivative_0_0.flatten().tolist() ,\
                        integration_derivative_0_05.flatten().tolist() ,\
                        integration_derivative_0_1.flatten().tolist() ,\
                        integration_derivative_0_15.flatten().tolist() ,\
                        integration_derivative_0_2.flatten().tolist() ,\
                        integration_derivative_0_25.flatten().tolist() ,\
                        integration_derivative_0_3.flatten().tolist() ,\
                        integration_derivative_0_35.flatten().tolist() ,\
                        integration_derivative_0_4.flatten().tolist() ,\
                        integration_derivative_0_45.flatten().tolist() ,\
                        integration_derivative_0_5.flatten().tolist() ,\
                        integration_derivative_0_6.flatten().tolist() ,\
                        integration_derivative_0_7.flatten().tolist() ,\
                        integration_derivative_0_8.flatten().tolist() ,\
                        integration_derivative_0_9.flatten().tolist() ,\
                        integration_derivative_1_0.flatten().tolist() ,\
                        integration_derivative_1_1.flatten().tolist() ,\
                        integration_derivative_1_2.flatten().tolist() ,\
                        integration_derivative_1_3.flatten().tolist() ,\
                        integration_derivative_1_4.flatten().tolist() ,\
                        integration_derivative_1_5.flatten().tolist() ,\
                        integration_derivative_1_6.flatten().tolist() ,\
                        integration_derivative_1_7.flatten().tolist() ,\
                        integration_derivative_1_8.flatten().tolist() ,\
                        integration_derivative_1_9.flatten().tolist() ,\
                        integration_derivative_2_0.flatten().tolist() ,\
                        integration_derivative_2_2.flatten().tolist() ,\
                        integration_derivative_2_4.flatten().tolist() ,\
                        integration_derivative_2_6.flatten().tolist() ,\
                        integration_derivative_2_8.flatten().tolist() ,\
                        integration_derivative_3_0.flatten().tolist() ,\
                        integration_derivative_3_2.flatten().tolist() ,\
                        integration_derivative_3_4.flatten().tolist() ,\
                        integration_derivative_3_6.flatten().tolist() ,\
                        integration_derivative_3_8.flatten().tolist() ,\
                        integration_derivative_4_0.flatten().tolist() ,\
                        integration_derivative_4_2.flatten().tolist() ,\
                        integration_derivative_4_4.flatten().tolist() ,\
                        integration_derivative_4_6.flatten().tolist() ,\
                        integration_derivative_4_8.flatten().tolist() ,\
                        integration_derivative_5_0.flatten().tolist() ,\
                        integration_derivative_5_2.flatten().tolist() ,\
                        integration_derivative_5_4.flatten().tolist() ,\
                        integration_derivative_5_6.flatten().tolist() ,\
                        integration_derivative_5_8.flatten().tolist() ,\
                        integration_derivative_6_0.flatten().tolist())
                        
                        

    else:
        padx, pady, padz = get_pads(pad_list)
  
        result = zip(   x[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        y[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        z[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        dimx[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        dimy[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        dimz[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        hx_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        hy_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        hz_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        It_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        num_e_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        nt_plot[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        n_plot[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_0_05[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_0_1[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_0_15[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_0_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_0_25[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_0_3[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_0_35[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_0_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_0_45[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_0_5[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_0_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_0_7[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_0_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_0_9[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_1_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_1_1[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_1_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_1_3[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_1_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_1_5[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_1_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_1_7[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_1_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_1_9[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_2_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_2_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_2_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_2_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_2_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_3_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_3_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_3_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_3_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_3_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_4_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_4_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_4_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_4_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_4_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_5_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_5_2[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_5_4[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_5_6[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_5_8[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        integration_derivative_6_0[padx:-padx,pady:-pady,padz:-padz].flatten().tolist())

    print("after zip: " + str(time.time()-start))

    return result



















def get_discriptors_from_density_integral_simple_norm(nt,hx,hy,hz, n, It, num_e, periodic = False, integral_accuracy = 4):
    '''
    get the first-fourth derivative from the density matrix using convolution
    get the integration convolution at 0.5, 1.0, 1.5, 2.0, 2.5
    '''
    def get_xyz_descriptors(nt, hx, hy, hz, It, num_e):

        dimx = np.ones(nt.shape)*nt.shape[0]
        dimy = np.ones(nt.shape)*nt.shape[1]
        dimz = np.ones(nt.shape)*nt.shape[2]
      
        nt_ave = It / float(hx*hy*hz) 
                
        hx_ = np.ones_like(nt)*hx
        hy_ = np.ones_like(nt)*hy
        hz_ = np.ones_like(nt)*hz
        num_e_ = np.ones_like(nt)*float(num_e)
        It_ = np.ones_like(nt)*float(It)
             
        x = np.ones(nt.shape)
        y = np.ones(nt.shape)
        z = np.ones(nt.shape)
        
        for index, density in np.ndenumerate(nt):
            x[index[0]][index[1]][index[2]] = index[0]
            y[index[0]][index[1]][index[2]] = index[1]
            z[index[0]][index[1]][index[2]] = index[2]
        
        return x, y, z, dimx, dimy, dimz, hx_, hy_, hz_, num_e_, It_, nt_ave
#    
#    def get_auto_accuracy(hx,hy,hz, r):
#        h = max([hx,hy,hz])
#        temp = 5 - int(math.floor((r/h)/3.))
#        if temp < 1:
#            return 1
#        else:
#            return temp
    
    n_plot = n.copy()
    nt_plot = nt.copy()    
    
 
    pad_list = []  
    result = []
    x, y, z, dimx, dimy, dimz, hx_, hy_, hz_, num_e_, It_, nt_ave = get_xyz_descriptors(n, hx, hy, hz, It, num_e)
    
    
    start = time.time()    
	
    
    print('\n\ngetting integration convolutions...')
    print(hx)
    print(hy)
    print(hz)
    print(max(hx, hy, hz) - min(hx, hy, hz))
    print(round(((hx + hy + hz) / 3.), 3))
    try:
        if max(hx, hy, hz) - min(hx, hy, hz) >= 0.01:
            raise NotImplementedError
        stencil_data = read_integration_stencil_file(hx, hy, hz)
        print("loaded stencil data from file")
        integration_0_05, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 0.05, stencil_data[str(0.05).replace(".","_")], stencil_data[str(0.05).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_0_1, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 0.1, stencil_data[str(0.1).replace(".","_")], stencil_data[str(0.1).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_0_15, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 0.15, stencil_data[str(0.15).replace(".","_")], stencil_data[str(0.15).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_0_2, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 0.2, stencil_data[str(0.2).replace(".","_")], stencil_data[str(0.2).replace(".","_") + "pad"])
        pad_list.append(temp_pad)

        integration_0_25, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 0.25, stencil_data[str(0.25).replace(".","_")], stencil_data[str(0.25).replace(".","_") + "pad"])
        pad_list.append(temp_pad)

        integration_0_3, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 0.3, stencil_data[str(0.3).replace(".","_")], stencil_data[str(0.3).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_0_35, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 0.35, stencil_data[str(0.35).replace(".","_")], stencil_data[str(0.35).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_0_4, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 0.4, stencil_data[str(0.4).replace(".","_")], stencil_data[str(0.4).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_0_45, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 0.45, stencil_data[str(0.45).replace(".","_")], stencil_data[str(0.45).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_0_5, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 0.5, stencil_data[str(0.5).replace(".","_")], stencil_data[str(0.5).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_0_6, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 0.6, stencil_data[str(0.6).replace(".","_")], stencil_data[str(0.6).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_0_7, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 0.7, stencil_data[str(0.7).replace(".","_")], stencil_data[str(0.7).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_0_8, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 0.8, stencil_data[str(0.8).replace(".","_")], stencil_data[str(0.8).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_0_9, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 0.9, stencil_data[str(0.9).replace(".","_")], stencil_data[str(0.9).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_1_0, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 1.0, stencil_data[str(1.0).replace(".","_")], stencil_data[str(1.0).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_1_1, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 1.1, stencil_data[str(1.1).replace(".","_")], stencil_data[str(1.1).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_1_2, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 1.2, stencil_data[str(1.2).replace(".","_")], stencil_data[str(1.2).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_1_3, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 1.3, stencil_data[str(1.3).replace(".","_")], stencil_data[str(1.3).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_1_4, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 1.4, stencil_data[str(1.4).replace(".","_")], stencil_data[str(1.4).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_1_5, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 1.5, stencil_data[str(1.5).replace(".","_")], stencil_data[str(1.5).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_1_6, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 1.6, stencil_data[str(1.6).replace(".","_")], stencil_data[str(1.6).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_1_7, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 1.7, stencil_data[str(1.7).replace(".","_")], stencil_data[str(1.7).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_1_8, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 1.8, stencil_data[str(1.8).replace(".","_")], stencil_data[str(1.8).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_1_9, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 1.9, stencil_data[str(1.9).replace(".","_")], stencil_data[str(1.9).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_2_0, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 2.0, stencil_data[str(2.0).replace(".","_")], stencil_data[str(2.0).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_2_2, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 2.2, stencil_data[str(2.2).replace(".","_")], stencil_data[str(2.2).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_2_4, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 2.4, stencil_data[str(2.4).replace(".","_")], stencil_data[str(2.4).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_2_6, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 2.6, stencil_data[str(2.6).replace(".","_")], stencil_data[str(2.6).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_2_8, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 2.8, stencil_data[str(2.8).replace(".","_")], stencil_data[str(2.8).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_3_0, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 3.0, stencil_data[str(3.0).replace(".","_")], stencil_data[str(3.0).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_3_2, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 3.2, stencil_data[str(3.2).replace(".","_")], stencil_data[str(3.2).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_3_4, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 3.4, stencil_data[str(3.4).replace(".","_")], stencil_data[str(3.4).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_3_6, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 3.6, stencil_data[str(3.6).replace(".","_")], stencil_data[str(3.6).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_3_8, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 3.8, stencil_data[str(3.8).replace(".","_")], stencil_data[str(3.8).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_4_0, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 4.0, stencil_data[str(4.0).replace(".","_")], stencil_data[str(4.0).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_4_2, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 4.2, stencil_data[str(4.2).replace(".","_")], stencil_data[str(4.2).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_4_4, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 4.4, stencil_data[str(4.4).replace(".","_")], stencil_data[str(4.4).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_4_6, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 4.6, stencil_data[str(4.6).replace(".","_")], stencil_data[str(4.6).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_4_8, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 4.8, stencil_data[str(4.8).replace(".","_")], stencil_data[str(4.8).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_5_0, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 5.0, stencil_data[str(5.0).replace(".","_")], stencil_data[str(5.0).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_5_2, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 5.2, stencil_data[str(5.2).replace(".","_")], stencil_data[str(5.2).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_5_4, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 5.4, stencil_data[str(5.4).replace(".","_")], stencil_data[str(5.4).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_5_6, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 5.6, stencil_data[str(5.6).replace(".","_")], stencil_data[str(5.6).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_5_8, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 5.8, stencil_data[str(5.8).replace(".","_")], stencil_data[str(5.8).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        integration_6_0, temp_pad = get_integral_fftconv_with_known_stencil(nt.copy(), hx, hy, hz, 6.0, stencil_data[str(6.0).replace(".","_")], stencil_data[str(6.0).replace(".","_") + "pad"])
        pad_list.append(temp_pad)
        
        
        
    except:
        integration_0_05, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.05, accuracy = get_auto_accuracy(hx,hy,hz, 0.05))
        pad_list.append(temp_pad)
        print('done 0.2')
    
        integration_0_1, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.1, accuracy = get_auto_accuracy(hx,hy,hz, 0.1))
        pad_list.append(temp_pad)
        print('done 0.2')
    
        integration_0_15, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.15, accuracy = get_auto_accuracy(hx,hy,hz, 0.15))
        pad_list.append(temp_pad)
        print('done 0.2')
    
        integration_0_2, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.2, accuracy = get_auto_accuracy(hx,hy,hz, 0.2))
        pad_list.append(temp_pad)
        print('done 0.2')
    
        integration_0_25, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.25, accuracy = get_auto_accuracy(hx,hy,hz, 0.25))
        pad_list.append(temp_pad)
        print('done 0.2')
    
        integration_0_3, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.3, accuracy = get_auto_accuracy(hx,hy,hz, 0.3))
        pad_list.append(temp_pad)
        print('done 0.2')
    
        integration_0_35, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.35, accuracy = get_auto_accuracy(hx,hy,hz, 0.35))
        pad_list.append(temp_pad)
        print('done 0.2')
        
        integration_0_4, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.4, accuracy = get_auto_accuracy(hx,hy,hz, 0.4))
        pad_list.append(temp_pad)
        print('done 0.4')
    
        integration_0_45, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.45, accuracy = get_auto_accuracy(hx,hy,hz, 0.45))
        pad_list.append(temp_pad)
        print('done 0.2')
    
        integration_0_5, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.5, accuracy = get_auto_accuracy(hx,hy,hz, 0.5))
        pad_list.append(temp_pad)
        print('done 0.2')
      
        integration_0_6, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.6, accuracy = get_auto_accuracy(hx,hy,hz, 0.6))
        pad_list.append(temp_pad)
        print('done 0.6')
    
        integration_0_7, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.7, accuracy = get_auto_accuracy(hx,hy,hz, 0.7))
        pad_list.append(temp_pad)
        print('done 0.8')
        
        integration_0_8, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.8, accuracy = get_auto_accuracy(hx,hy,hz, 0.8))
        pad_list.append(temp_pad)
        print('done 0.8')
    
        integration_0_9, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 0.9, accuracy = get_auto_accuracy(hx,hy,hz, 0.9))
        pad_list.append(temp_pad)
        print('done 0.8')
        
        integration_1_0, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.0, accuracy = get_auto_accuracy(hx,hy,hz, 1.0))
        pad_list.append(temp_pad)
        print('done 1.0')
    
        integration_1_1, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.1, accuracy = get_auto_accuracy(hx,hy,hz, 1.1))
        pad_list.append(temp_pad)
        print('done 0.8')
        
        integration_1_2, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.2, accuracy = get_auto_accuracy(hx,hy,hz, 1.2))
        pad_list.append(temp_pad)
        print('done 1.2')
    
        integration_1_3, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.3, accuracy = get_auto_accuracy(hx,hy,hz, 1.3))
        pad_list.append(temp_pad)
        print('done 0.8')
        
        integration_1_4, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.4, accuracy = get_auto_accuracy(hx,hy,hz, 1.4))
        pad_list.append(temp_pad)
        print('done 1.4')
    
        integration_1_5, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.5, accuracy = get_auto_accuracy(hx,hy,hz, 1.5))
        pad_list.append(temp_pad)
        print('done 0.8')
        
        integration_1_6, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.6, accuracy = get_auto_accuracy(hx,hy,hz, 1.6))
        pad_list.append(temp_pad)
        print('done 1.6')
    
        integration_1_7, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.7, accuracy = get_auto_accuracy(hx,hy,hz, 1.7))
        pad_list.append(temp_pad)
        print('done 1.8')
        
        integration_1_8, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.8, accuracy = get_auto_accuracy(hx,hy,hz, 1.8))
        pad_list.append(temp_pad)
        print('done 1.8')
    
        integration_1_9, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 1.9, accuracy = get_auto_accuracy(hx,hy,hz, 1.9))
        pad_list.append(temp_pad)
        print('done 1.8')
        
        integration_2_0, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 2.0, accuracy = get_auto_accuracy(hx,hy,hz, 2.0))
        pad_list.append(temp_pad)
        print('done 2.0')
    
        integration_2_2, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 2.2, accuracy = get_auto_accuracy(hx,hy,hz, 2.2))
        pad_list.append(temp_pad)
        print('done 2.2')
      
        integration_2_4, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 2.4, accuracy = get_auto_accuracy(hx,hy,hz, 2.4))
        pad_list.append(temp_pad)
        print('done 2.4')
        
        integration_2_6, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 2.6, accuracy = get_auto_accuracy(hx,hy,hz, 2.6))
        pad_list.append(temp_pad)
        print('done 2.6')
        
        integration_2_8, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 2.8, accuracy = get_auto_accuracy(hx,hy,hz, 2.8))
        pad_list.append(temp_pad)
        print('done 2.8')
        
        integration_3_0, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 3.0, accuracy = get_auto_accuracy(hx,hy,hz, 3.0))
        pad_list.append(temp_pad)
        print('done 3.0'                           )
    
        integration_3_2, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 3.2, accuracy = get_auto_accuracy(hx,hy,hz, 3.2))
        pad_list.append(temp_pad)
        print('done 1.2')
      
        integration_3_4, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 3.4, accuracy = get_auto_accuracy(hx,hy,hz, 3.4))
        pad_list.append(temp_pad)
        print('done 1.4')
        
        integration_3_6, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 3.6, accuracy = get_auto_accuracy(hx,hy,hz, 3.6))
        pad_list.append(temp_pad)
        print('done 1.6')
        
        integration_3_8, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 3.8, accuracy = get_auto_accuracy(hx,hy,hz, 3.8))
        pad_list.append(temp_pad)
        print('done 1.8')
        
        integration_4_0, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 4.0, accuracy = get_auto_accuracy(hx,hy,hz, 4.0))
        pad_list.append(temp_pad)
        print('done 2.0')
    
        integration_4_2, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 4.2, accuracy = get_auto_accuracy(hx,hy,hz, 4.2))
        pad_list.append(temp_pad)
        print('done 2.2')
      
        integration_4_4, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 4.4, accuracy = get_auto_accuracy(hx,hy,hz, 4.4))
        pad_list.append(temp_pad)
        print('done 2.4')
        
        integration_4_6, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 4.6, accuracy = get_auto_accuracy(hx,hy,hz, 4.6))
        pad_list.append(temp_pad)
        print('done 2.6')
        
        integration_4_8, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 4.8, accuracy = get_auto_accuracy(hx,hy,hz, 4.8))
        pad_list.append(temp_pad)
        print('done 2.8')
        
        integration_5_0, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 5.0, accuracy = get_auto_accuracy(hx,hy,hz, 5.0))
        pad_list.append(temp_pad)
        print('done 3.0'   )
    
        integration_5_2, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 5.2, accuracy = get_auto_accuracy(hx,hy,hz, 5.2))
        pad_list.append(temp_pad)
        print('done 2.2')
      
        integration_5_4, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 5.4, accuracy = get_auto_accuracy(hx,hy,hz, 5.4))
        pad_list.append(temp_pad)
        print('done 2.4')
        
        integration_5_6, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 5.6, accuracy = get_auto_accuracy(hx,hy,hz, 5.6))
        pad_list.append(temp_pad)
        print('done 2.6')
        
        integration_5_8, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 5.8, accuracy = get_auto_accuracy(hx,hy,hz, 5.8))
        pad_list.append(temp_pad)
        print('done 2.8')
        
        integration_6_0, temp_pad  = get_integration_fftconv(nt.copy(), hx, hy, hz, 6.0, accuracy = get_auto_accuracy(hx,hy,hz, 6.0))
        pad_list.append(temp_pad)
        print('done 3.0' )

    print("after first conv: " + str(time.time()-start))
    
    ave_density_0_00_norm = nt.copy() 
    ave_density_0_05_norm = get_homo_nondimensional_nave(integration_0_05, 1.0, 0.05)
    ave_density_0_10_norm = get_homo_nondimensional_nave(integration_0_1, 1.0, 0.10)
    ave_density_0_15_norm = get_homo_nondimensional_nave(integration_0_15, 1.0, 0.15)
    ave_density_0_20_norm = get_homo_nondimensional_nave(integration_0_2, 1.0, 0.20)
    ave_density_0_25_norm = get_homo_nondimensional_nave(integration_0_25, 1.0, 0.25)
    ave_density_0_30_norm = get_homo_nondimensional_nave(integration_0_3, 1.0, 0.30)
    ave_density_0_35_norm = get_homo_nondimensional_nave(integration_0_35, 1.0, 0.35)
    ave_density_0_40_norm = get_homo_nondimensional_nave(integration_0_4, 1.0, 0.40)
    ave_density_0_45_norm = get_homo_nondimensional_nave(integration_0_45, 1.0, 0.45)
    ave_density_0_5_norm = get_homo_nondimensional_nave(integration_0_5, 1.0, 0.50)
    ave_density_0_6_norm = get_homo_nondimensional_nave(integration_0_6, 1.0, 0.60)
    ave_density_0_7_norm = get_homo_nondimensional_nave(integration_0_7, 1.0, 0.70)
    ave_density_0_8_norm = get_homo_nondimensional_nave(integration_0_8, 1.0, 0.80)
    ave_density_0_9_norm = get_homo_nondimensional_nave(integration_0_9, 1.0, 0.90)
    ave_density_1_0_norm = get_homo_nondimensional_nave(integration_1_0, 1.0, 1.0)
    ave_density_1_1_norm = get_homo_nondimensional_nave(integration_1_1, 1.0, 1.1)
    ave_density_1_2_norm = get_homo_nondimensional_nave(integration_1_2, 1.0, 1.2)
    ave_density_1_3_norm = get_homo_nondimensional_nave(integration_1_3, 1.0, 1.3)
    ave_density_1_4_norm = get_homo_nondimensional_nave(integration_1_4, 1.0, 1.4)
    ave_density_1_5_norm = get_homo_nondimensional_nave(integration_1_5, 1.0, 1.5)
    ave_density_1_6_norm = get_homo_nondimensional_nave(integration_1_6, 1.0, 1.6)
    ave_density_1_7_norm = get_homo_nondimensional_nave(integration_1_7, 1.0, 1.7)
    ave_density_1_8_norm = get_homo_nondimensional_nave(integration_1_8, 1.0, 1.8)
    ave_density_1_9_norm = get_homo_nondimensional_nave(integration_1_9, 1.0, 1.9)
    ave_density_2_0_norm = get_homo_nondimensional_nave(integration_2_0, 1.0, 2.0)
    ave_density_2_2_norm = get_homo_nondimensional_nave(integration_2_2, 1.0, 2.2)
    ave_density_2_4_norm = get_homo_nondimensional_nave(integration_2_4, 1.0, 2.4)
    ave_density_2_6_norm = get_homo_nondimensional_nave(integration_2_6, 1.0, 2.6)
    ave_density_2_8_norm = get_homo_nondimensional_nave(integration_2_8, 1.0, 2.8)
    
    ave_density_3_0_norm = get_homo_nondimensional_nave(integration_3_0, 1.0, 3.0)
    ave_density_3_2_norm = get_homo_nondimensional_nave(integration_3_2, 1.0, 3.2)
    ave_density_3_4_norm = get_homo_nondimensional_nave(integration_3_4, 1.0, 3.4)
    ave_density_3_6_norm = get_homo_nondimensional_nave(integration_3_6, 1.0, 3.6)
    ave_density_3_8_norm = get_homo_nondimensional_nave(integration_3_8, 1.0, 3.8)
    
    ave_density_4_0_norm = get_homo_nondimensional_nave(integration_4_0, 1.0, 4.0)
    ave_density_4_2_norm = get_homo_nondimensional_nave(integration_4_2, 1.0, 4.2)
    ave_density_4_4_norm = get_homo_nondimensional_nave(integration_4_4, 1.0, 4.4)
    ave_density_4_6_norm = get_homo_nondimensional_nave(integration_4_6, 1.0, 4.6)
    ave_density_4_8_norm = get_homo_nondimensional_nave(integration_4_8, 1.0, 4.8)
    
    ave_density_5_0_norm = get_homo_nondimensional_nave(integration_5_0, 1.0, 5.0)
    ave_density_5_2_norm = get_homo_nondimensional_nave(integration_5_2, 1.0, 5.2)
    ave_density_5_4_norm = get_homo_nondimensional_nave(integration_5_4, 1.0, 5.4)
    ave_density_5_6_norm = get_homo_nondimensional_nave(integration_5_6, 1.0, 5.6)
    ave_density_5_8_norm = get_homo_nondimensional_nave(integration_5_8, 1.0, 5.8)
    ave_density_6_0_norm = get_homo_nondimensional_nave(integration_6_0, 1.0, 6.0)



    print("after everything: " + str(time.time()-start)    )


    if periodic:
        result = zip(   x.flatten().tolist() ,\
                        y.flatten().tolist() ,\
                        z.flatten().tolist() ,\
                        dimx.flatten().tolist() ,\
                        dimy.flatten().tolist() ,\
                        dimz.flatten().tolist() ,\
                        hx_.flatten().tolist() ,\
                        hy_.flatten().tolist() ,\
                        hz_.flatten().tolist() ,\
                        It_.flatten().tolist() ,\
                        num_e_.flatten().tolist() ,\
                        nt_plot.flatten().tolist() ,\
                        n_plot.flatten().tolist() ,\
                        ave_density_0_00_norm.flatten().tolist() ,\
                        ave_density_0_05_norm.flatten().tolist() ,\
                        ave_density_0_10_norm.flatten().tolist() ,\
                        ave_density_0_15_norm.flatten().tolist() ,\
                        ave_density_0_20_norm.flatten().tolist() ,\
                        ave_density_0_25_norm.flatten().tolist() ,\
                        ave_density_0_30_norm.flatten().tolist() ,\
                        ave_density_0_35_norm.flatten().tolist() ,\
                        ave_density_0_40_norm.flatten().tolist() ,\
                        ave_density_0_45_norm.flatten().tolist() ,\
                        ave_density_0_5_norm.flatten().tolist() ,\
                        ave_density_0_6_norm.flatten().tolist() ,\
                        ave_density_0_7_norm.flatten().tolist() ,\
                        ave_density_0_8_norm.flatten().tolist() ,\
                        ave_density_0_9_norm.flatten().tolist() ,\
                        ave_density_1_0_norm.flatten().tolist() ,\
                        ave_density_1_1_norm.flatten().tolist() ,\
                        ave_density_1_2_norm.flatten().tolist() ,\
                        ave_density_1_3_norm.flatten().tolist() ,\
                        ave_density_1_4_norm.flatten().tolist() ,\
                        ave_density_1_5_norm.flatten().tolist() ,\
                        ave_density_1_6_norm.flatten().tolist() ,\
                        ave_density_1_7_norm.flatten().tolist() ,\
                        ave_density_1_8_norm.flatten().tolist() ,\
                        ave_density_1_9_norm.flatten().tolist() ,\
                        ave_density_2_0_norm.flatten().tolist() ,\
                        ave_density_2_2_norm.flatten().tolist() ,\
                        ave_density_2_4_norm.flatten().tolist() ,\
                        ave_density_2_6_norm.flatten().tolist() ,\
                        ave_density_2_8_norm.flatten().tolist() ,\
                        ave_density_3_0_norm.flatten().tolist() ,\
                        ave_density_3_2_norm.flatten().tolist() ,\
                        ave_density_3_4_norm.flatten().tolist() ,\
                        ave_density_3_6_norm.flatten().tolist() ,\
                        ave_density_3_8_norm.flatten().tolist() ,\
                        ave_density_4_0_norm.flatten().tolist() ,\
                        ave_density_4_2_norm.flatten().tolist() ,\
                        ave_density_4_4_norm.flatten().tolist() ,\
                        ave_density_4_6_norm.flatten().tolist() ,\
                        ave_density_4_8_norm.flatten().tolist() ,\
                        ave_density_5_0_norm.flatten().tolist() ,\
                        ave_density_5_2_norm.flatten().tolist() ,\
                        ave_density_5_4_norm.flatten().tolist() ,\
                        ave_density_5_6_norm.flatten().tolist() ,\
                        ave_density_5_8_norm.flatten().tolist() ,\
                        ave_density_6_0_norm.flatten().tolist())
                        
                        

    else:
        padx, pady, padz = get_pads(pad_list)
  
        result = zip(   x[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        y[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        z[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        dimx[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        dimy[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        dimz[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        hx_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        hy_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        hz_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        It_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        num_e_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        nt_plot[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        n_plot[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_05_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_10_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_15_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_20_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_25_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_30_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_35_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_40_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_45_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_5_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_6_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_7_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_8_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_9_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_0_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_1_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_2_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_3_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_4_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_5_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_6_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_7_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_8_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_9_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_2_0_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_2_2_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_2_4_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_2_6_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_2_8_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_3_0_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_3_2_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_3_4_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_3_6_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_3_8_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_4_0_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_4_2_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_4_4_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_4_6_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_4_8_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_5_0_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_5_2_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_5_4_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_5_6_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_5_8_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_6_0_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist())

  

    print("after zip: " + str(time.time()-start))

    return result
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def get_discriptors_from_density_integral_simple_norm_psi4_test_short(hx,hy,hz, n, num_e, V_xc, ep_xc, tau, gamma, periodic = False, integral_accuracy = 4):
    '''
    get the first-fourth derivative from the density matrix using convolution
    get the integration convolution at 0.5, 1.0, 1.5, 2.0, 2.5
    '''
    def get_xyz_descriptors(n, hx, hy, hz, num_e):

        dimx = np.ones(n.shape)*n.shape[0]
        dimy = np.ones(n.shape)*n.shape[1]
        dimz = np.ones(n.shape)*n.shape[2]
      
#        nt_ave = It / float(hx*hy*hz) 
                
        hx_ = np.ones_like(n)*hx
        hy_ = np.ones_like(n)*hy
        hz_ = np.ones_like(n)*hz
        num_e_ = np.ones_like(n)*float(num_e)
#        It_ = np.ones_like(n)*float(It)
             
        x = np.ones(n.shape)
        y = np.ones(n.shape)
        z = np.ones(n.shape)
        
        for index, density in np.ndenumerate(n):
            x[index[0]][index[1]][index[2]] = index[0]
            y[index[0]][index[1]][index[2]] = index[1]
            z[index[0]][index[1]][index[2]] = index[2]
        
        return x, y, z, dimx, dimy, dimz, hx_, hy_, hz_, num_e_

    n_plot = n.copy()

    pad_list = []  
    result = []
    x, y, z, dimx, dimy, dimz, hx_, hy_, hz_, num_e_= get_xyz_descriptors(n, hx, hy, hz, num_e)
    
    
    start = time.time()    
	
    
    print('\n\ngetting integration convolutions...')
    print(hx)
    print(hy)
    print(hz)
    print(max(hx, hy, hz) - min(hx, hy, hz))
    print(round(((hx + hy + hz) / 3.), 3))
#    try:
#        if max(hx, hy, hz) - min(hx, hy, hz) >= 0.01:
#            raise NotImplementedError
#        stencil_data = read_integration_stencil_file(hx, hy, hz)
#        print("loaded stencil data from file")
#        
#        integration_0_05, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.05, stencil_data[str(0.05).replace(".","_")], stencil_data[str(0.05).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_1, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.1, stencil_data[str(0.1).replace(".","_")], stencil_data[str(0.1).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_15, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.15, stencil_data[str(0.15).replace(".","_")], stencil_data[str(0.15).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_2, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.2, stencil_data[str(0.2).replace(".","_")], stencil_data[str(0.2).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#
#        integration_0_25, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.25, stencil_data[str(0.25).replace(".","_")], stencil_data[str(0.25).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#
#        integration_0_3, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.3, stencil_data[str(0.3).replace(".","_")], stencil_data[str(0.3).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_35, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.35, stencil_data[str(0.35).replace(".","_")], stencil_data[str(0.35).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_4, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.4, stencil_data[str(0.4).replace(".","_")], stencil_data[str(0.4).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_45, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.45, stencil_data[str(0.45).replace(".","_")], stencil_data[str(0.45).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_5, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.5, stencil_data[str(0.5).replace(".","_")], stencil_data[str(0.5).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_55, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.55, stencil_data[str(0.55).replace(".","_")], stencil_data[str(0.55).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        
#        integration_0_6, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.6, stencil_data[str(0.6).replace(".","_")], stencil_data[str(0.6).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_65, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.65, stencil_data[str(0.65).replace(".","_")], stencil_data[str(0.65).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        
#        integration_0_7, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.7, stencil_data[str(0.7).replace(".","_")], stencil_data[str(0.7).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_75, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.75, stencil_data[str(0.75).replace(".","_")], stencil_data[str(0.75).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        
#        integration_0_8, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.8, stencil_data[str(0.8).replace(".","_")], stencil_data[str(0.8).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_85, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.85, stencil_data[str(0.85).replace(".","_")], stencil_data[str(0.85).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        
#        integration_0_9, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.9, stencil_data[str(0.9).replace(".","_")], stencil_data[str(0.9).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_95, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.95, stencil_data[str(0.95).replace(".","_")], stencil_data[str(0.95).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        
#        integration_1_0, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 1.0, stencil_data[str(1.0).replace(".","_")], stencil_data[str(1.0).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#
#    except:
#    integration_0_05, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.05, accuracy = get_auto_accuracy(hx,hy,hz, 0.05))
#    pad_list.append(temp_pad)
#    print('done 0.2')

    integration_0_1, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.1, accuracy = get_auto_accuracy(hx,hy,hz, 0.1))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_0_15, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.15, accuracy = get_auto_accuracy(hx,hy,hz, 0.15))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_0_2, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.2, accuracy = get_auto_accuracy(hx,hy,hz, 0.2))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_0_25, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.25, accuracy = get_auto_accuracy(hx,hy,hz, 0.25))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_0_3, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.3, accuracy = get_auto_accuracy(hx,hy,hz, 0.3))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_0_35, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.35, accuracy = get_auto_accuracy(hx,hy,hz, 0.35))
    pad_list.append(temp_pad)
    print('done 0.2')
    
    integration_0_4, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.4, accuracy = get_auto_accuracy(hx,hy,hz, 0.4))
    pad_list.append(temp_pad)
    print('done 0.4')

    integration_0_45, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.45, accuracy = get_auto_accuracy(hx,hy,hz, 0.45))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_0_5, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.5, accuracy = get_auto_accuracy(hx,hy,hz, 0.5))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_0_55, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.55, accuracy = get_auto_accuracy(hx,hy,hz, 0.55))
    pad_list.append(temp_pad)
    print('done 0.2')

  
    integration_0_6, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.6, accuracy = get_auto_accuracy(hx,hy,hz, 0.6))
    pad_list.append(temp_pad)
    print('done 0.6')

    integration_0_65, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.65, accuracy = get_auto_accuracy(hx,hy,hz, 0.65))
    pad_list.append(temp_pad)
    print('done 0.2')


    integration_0_7, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.7, accuracy = get_auto_accuracy(hx,hy,hz, 0.7))
    pad_list.append(temp_pad)
    print('done 0.8')

    integration_0_75, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.75, accuracy = get_auto_accuracy(hx,hy,hz, 0.75))
    pad_list.append(temp_pad)
    print('done 0.2')

    
    integration_0_8, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.8, accuracy = get_auto_accuracy(hx,hy,hz, 0.8))
    pad_list.append(temp_pad)
    print('done 0.8')

    integration_0_85, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.85, accuracy = get_auto_accuracy(hx,hy,hz, 0.85))
    pad_list.append(temp_pad)
    print('done 0.2')


    integration_0_9, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.9, accuracy = get_auto_accuracy(hx,hy,hz, 0.9))
    pad_list.append(temp_pad)
    print('done 0.8')

    integration_0_95, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.95, accuracy = get_auto_accuracy(hx,hy,hz, 0.95))
    pad_list.append(temp_pad)
    print('done 0.2')

    
    integration_1_0, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 1.0, accuracy = get_auto_accuracy(hx,hy,hz, 1.0))
    pad_list.append(temp_pad)
    print('done 1.0')
    


    print("after first conv: " + str(time.time()-start))
    
    ave_density_0_00_norm = n.copy()
#    ave_density_0_05_norm = get_homo_nondimensional_nave(integration_0_05, 1.0, 0.05)
    ave_density_0_10_norm = get_homo_nondimensional_nave(integration_0_1, 1.0, 0.10)
    ave_density_0_15_norm = get_homo_nondimensional_nave(integration_0_15, 1.0, 0.15)
    ave_density_0_20_norm = get_homo_nondimensional_nave(integration_0_2, 1.0, 0.20)
    ave_density_0_25_norm = get_homo_nondimensional_nave(integration_0_25, 1.0, 0.25)
    ave_density_0_30_norm = get_homo_nondimensional_nave(integration_0_3, 1.0, 0.30)
    ave_density_0_35_norm = get_homo_nondimensional_nave(integration_0_35, 1.0, 0.35)
    ave_density_0_40_norm = get_homo_nondimensional_nave(integration_0_4, 1.0, 0.40)
    ave_density_0_45_norm = get_homo_nondimensional_nave(integration_0_45, 1.0, 0.45)
    ave_density_0_50_norm = get_homo_nondimensional_nave(integration_0_5, 1.0, 0.50)
    ave_density_0_55_norm = get_homo_nondimensional_nave(integration_0_55, 1.0, 0.55)
    ave_density_0_60_norm = get_homo_nondimensional_nave(integration_0_6, 1.0, 0.60)
    ave_density_0_65_norm = get_homo_nondimensional_nave(integration_0_65, 1.0, 0.65)
    ave_density_0_70_norm = get_homo_nondimensional_nave(integration_0_7, 1.0, 0.70)
    ave_density_0_75_norm = get_homo_nondimensional_nave(integration_0_75, 1.0, 0.75)
    ave_density_0_80_norm = get_homo_nondimensional_nave(integration_0_8, 1.0, 0.80)
    ave_density_0_85_norm = get_homo_nondimensional_nave(integration_0_85, 1.0, 0.85)
    ave_density_0_90_norm = get_homo_nondimensional_nave(integration_0_9, 1.0, 0.90)
    ave_density_0_95_norm = get_homo_nondimensional_nave(integration_0_95, 1.0, 0.95)
    ave_density_1_0_norm = get_homo_nondimensional_nave(integration_1_0, 1.0, 1.0)




    print("after everything: " + str(time.time()-start)    )


    if periodic:
        result = zip(   x.flatten().tolist() ,\
                        y.flatten().tolist() ,\
                        z.flatten().tolist() ,\
                        dimx.flatten().tolist() ,\
                        dimy.flatten().tolist() ,\
                        dimz.flatten().tolist() ,\
                        hx_.flatten().tolist() ,\
                        hy_.flatten().tolist() ,\
                        hz_.flatten().tolist() ,\
                        num_e_.flatten().tolist() ,\
                        n_plot.flatten().tolist() ,\
                        V_xc.flatten().tolist() ,\
                        ep_xc.flatten().tolist() ,\
                        tau.flatten().tolist() ,\
                        gamma.flatten().tolist() ,\
                        ave_density_0_00_norm.flatten().tolist() ,\
#                        ave_density_0_05_norm.flatten().tolist() ,\
                        ave_density_0_10_norm.flatten().tolist() ,\
                        ave_density_0_15_norm.flatten().tolist() ,\
                        ave_density_0_20_norm.flatten().tolist() ,\
                        ave_density_0_25_norm.flatten().tolist() ,\
                        ave_density_0_30_norm.flatten().tolist() ,\
                        ave_density_0_35_norm.flatten().tolist() ,\
                        ave_density_0_40_norm.flatten().tolist() ,\
                        ave_density_0_45_norm.flatten().tolist() ,\
                        ave_density_0_50_norm.flatten().tolist() ,\
                        ave_density_0_55_norm.flatten().tolist() ,\
                        ave_density_0_60_norm.flatten().tolist() ,\
                        ave_density_0_65_norm.flatten().tolist() ,\
                        ave_density_0_70_norm.flatten().tolist() ,\
                        ave_density_0_75_norm.flatten().tolist() ,\
                        ave_density_0_80_norm.flatten().tolist() ,\
                        ave_density_0_85_norm.flatten().tolist() ,\
                        ave_density_0_90_norm.flatten().tolist() ,\
                        ave_density_0_95_norm.flatten().tolist() ,\
                        ave_density_1_0_norm.flatten().tolist())
                        
                        

    else:
        padx, pady, padz = get_pads(pad_list)
  
        result = zip(   x[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        y[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        z[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        dimx[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        dimy[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        dimz[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        hx_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        hy_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        hz_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        num_e_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        n_plot[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        V_xc[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ep_xc[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        tau[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        gamma[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_00_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
#                        ave_density_0_05_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_10_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_15_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_20_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_25_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_30_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_35_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_40_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_45_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_50_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_55_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_60_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_65_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_70_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_75_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_80_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_85_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_90_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_95_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_0_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() )

  

    print("after zip: " + str(time.time()-start))

    return result
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def get_discriptors_from_density_integral_simple_norm_psi4_test(hx,hy,hz, n, num_e, V_xc, ep_xc, tau, gamma, periodic = False, integral_accuracy = 4):
    '''
    get the first-fourth derivative from the density matrix using convolution
    get the integration convolution at 0.5, 1.0, 1.5, 2.0, 2.5
    '''
    def get_xyz_descriptors(n, hx, hy, hz, num_e):

        dimx = np.ones(n.shape)*n.shape[0]
        dimy = np.ones(n.shape)*n.shape[1]
        dimz = np.ones(n.shape)*n.shape[2]
      
#        nt_ave = It / float(hx*hy*hz) 
                
        hx_ = np.ones_like(n)*hx
        hy_ = np.ones_like(n)*hy
        hz_ = np.ones_like(n)*hz
        num_e_ = np.ones_like(n)*float(num_e)
#        It_ = np.ones_like(n)*float(It)
             
        x = np.ones(n.shape)
        y = np.ones(n.shape)
        z = np.ones(n.shape)
        
        for index, density in np.ndenumerate(n):
            x[index[0]][index[1]][index[2]] = index[0]
            y[index[0]][index[1]][index[2]] = index[1]
            z[index[0]][index[1]][index[2]] = index[2]
        
        return x, y, z, dimx, dimy, dimz, hx_, hy_, hz_, num_e_

    n_plot = n.copy()

    pad_list = []  
    result = []
    x, y, z, dimx, dimy, dimz, hx_, hy_, hz_, num_e_= get_xyz_descriptors(n, hx, hy, hz, num_e)
    
    
    start = time.time()    
	
    
    print('\n\ngetting integration convolutions...')
    print(hx)
    print(hy)
    print(hz)
    print(max(hx, hy, hz) - min(hx, hy, hz))
    print(round(((hx + hy + hz) / 3.), 3))
#    try:
#        if max(hx, hy, hz) - min(hx, hy, hz) >= 0.01:
#            raise NotImplementedError
#        stencil_data = read_integration_stencil_file(hx, hy, hz)
#        print("loaded stencil data from file")
#        
#        integration_0_05, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.05, stencil_data[str(0.05).replace(".","_")], stencil_data[str(0.05).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_1, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.1, stencil_data[str(0.1).replace(".","_")], stencil_data[str(0.1).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_15, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.15, stencil_data[str(0.15).replace(".","_")], stencil_data[str(0.15).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_2, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.2, stencil_data[str(0.2).replace(".","_")], stencil_data[str(0.2).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#
#        integration_0_25, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.25, stencil_data[str(0.25).replace(".","_")], stencil_data[str(0.25).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#
#        integration_0_3, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.3, stencil_data[str(0.3).replace(".","_")], stencil_data[str(0.3).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_35, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.35, stencil_data[str(0.35).replace(".","_")], stencil_data[str(0.35).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_4, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.4, stencil_data[str(0.4).replace(".","_")], stencil_data[str(0.4).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_45, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.45, stencil_data[str(0.45).replace(".","_")], stencil_data[str(0.45).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_5, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.5, stencil_data[str(0.5).replace(".","_")], stencil_data[str(0.5).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_55, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.55, stencil_data[str(0.55).replace(".","_")], stencil_data[str(0.55).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        
#        integration_0_6, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.6, stencil_data[str(0.6).replace(".","_")], stencil_data[str(0.6).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_65, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.65, stencil_data[str(0.65).replace(".","_")], stencil_data[str(0.65).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        
#        integration_0_7, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.7, stencil_data[str(0.7).replace(".","_")], stencil_data[str(0.7).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_75, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.75, stencil_data[str(0.75).replace(".","_")], stencil_data[str(0.75).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        
#        integration_0_8, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.8, stencil_data[str(0.8).replace(".","_")], stencil_data[str(0.8).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_85, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.85, stencil_data[str(0.85).replace(".","_")], stencil_data[str(0.85).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        
#        integration_0_9, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.9, stencil_data[str(0.9).replace(".","_")], stencil_data[str(0.9).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_95, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.95, stencil_data[str(0.95).replace(".","_")], stencil_data[str(0.95).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        
#        integration_1_0, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 1.0, stencil_data[str(1.0).replace(".","_")], stencil_data[str(1.0).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#
#    except:
    integration_0_05, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.05, accuracy = get_auto_accuracy(hx,hy,hz, 0.05))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_0_1, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.1, accuracy = get_auto_accuracy(hx,hy,hz, 0.1))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_0_15, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.15, accuracy = get_auto_accuracy(hx,hy,hz, 0.15))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_0_2, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.2, accuracy = get_auto_accuracy(hx,hy,hz, 0.2))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_0_25, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.25, accuracy = get_auto_accuracy(hx,hy,hz, 0.25))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_0_3, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.3, accuracy = get_auto_accuracy(hx,hy,hz, 0.3))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_0_35, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.35, accuracy = get_auto_accuracy(hx,hy,hz, 0.35))
    pad_list.append(temp_pad)
    print('done 0.2')
    
    integration_0_4, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.4, accuracy = get_auto_accuracy(hx,hy,hz, 0.4))
    pad_list.append(temp_pad)
    print('done 0.4')

    integration_0_45, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.45, accuracy = get_auto_accuracy(hx,hy,hz, 0.45))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_0_5, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.5, accuracy = get_auto_accuracy(hx,hy,hz, 0.5))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_0_55, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.55, accuracy = get_auto_accuracy(hx,hy,hz, 0.55))
    pad_list.append(temp_pad)
    print('done 0.2')

  
    integration_0_6, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.6, accuracy = get_auto_accuracy(hx,hy,hz, 0.6))
    pad_list.append(temp_pad)
    print('done 0.6')

    integration_0_65, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.65, accuracy = get_auto_accuracy(hx,hy,hz, 0.65))
    pad_list.append(temp_pad)
    print('done 0.2')


    integration_0_7, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.7, accuracy = get_auto_accuracy(hx,hy,hz, 0.7))
    pad_list.append(temp_pad)
    print('done 0.8')

    integration_0_75, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.75, accuracy = get_auto_accuracy(hx,hy,hz, 0.75))
    pad_list.append(temp_pad)
    print('done 0.2')

    
    integration_0_8, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.8, accuracy = get_auto_accuracy(hx,hy,hz, 0.8))
    pad_list.append(temp_pad)
    print('done 0.8')

    integration_0_85, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.85, accuracy = get_auto_accuracy(hx,hy,hz, 0.85))
    pad_list.append(temp_pad)
    print('done 0.2')


    integration_0_9, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.9, accuracy = get_auto_accuracy(hx,hy,hz, 0.9))
    pad_list.append(temp_pad)
    print('done 0.8')

    integration_0_95, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.95, accuracy = get_auto_accuracy(hx,hy,hz, 0.95))
    pad_list.append(temp_pad)
    print('done 0.2')

    
    integration_1_0, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 1.0, accuracy = get_auto_accuracy(hx,hy,hz, 1.0))
    pad_list.append(temp_pad)
    print('done 1.0')
    
    
    integration_1_05, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 1.05, accuracy = get_auto_accuracy(hx,hy,hz, 1.05))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_1_1, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 1.1, accuracy = get_auto_accuracy(hx,hy,hz, 1.1))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_1_15, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 1.15, accuracy = get_auto_accuracy(hx,hy,hz, 1.15))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_1_2, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 1.2, accuracy = get_auto_accuracy(hx,hy,hz, 1.2))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_1_25, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 1.25, accuracy = get_auto_accuracy(hx,hy,hz, 1.25))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_1_3, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 1.3, accuracy = get_auto_accuracy(hx,hy,hz, 1.3))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_1_35, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 1.35, accuracy = get_auto_accuracy(hx,hy,hz, 1.35))
    pad_list.append(temp_pad)
    print('done 0.2')
    
    integration_1_4, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 1.4, accuracy = get_auto_accuracy(hx,hy,hz, 1.4))
    pad_list.append(temp_pad)
    print('done 0.4')

    integration_1_45, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 1.45, accuracy = get_auto_accuracy(hx,hy,hz, 1.45))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_1_5, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 1.5, accuracy = get_auto_accuracy(hx,hy,hz, 1.5))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_1_55, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 1.55, accuracy = get_auto_accuracy(hx,hy,hz, 1.55))
    pad_list.append(temp_pad)
    print('done 0.2')

  
    integration_1_6, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 1.6, accuracy = get_auto_accuracy(hx,hy,hz, 1.6))
    pad_list.append(temp_pad)
    print('done 0.6')

    integration_1_65, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 1.65, accuracy = get_auto_accuracy(hx,hy,hz, 1.65))
    pad_list.append(temp_pad)
    print('done 0.2')


    integration_1_7, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 1.7, accuracy = get_auto_accuracy(hx,hy,hz, 1.7))
    pad_list.append(temp_pad)
    print('done 0.8')

    integration_1_75, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 1.75, accuracy = get_auto_accuracy(hx,hy,hz, 1.75))
    pad_list.append(temp_pad)
    print('done 0.2')

    
    integration_1_8, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 1.8, accuracy = get_auto_accuracy(hx,hy,hz, 1.8))
    pad_list.append(temp_pad)
    print('done 0.8')

    integration_1_85, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 1.85, accuracy = get_auto_accuracy(hx,hy,hz, 1.85))
    pad_list.append(temp_pad)
    print('done 0.2')


    integration_1_9, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 1.9, accuracy = get_auto_accuracy(hx,hy,hz, 1.9))
    pad_list.append(temp_pad)
    print('done 0.8')

    integration_1_95, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 1.95, accuracy = get_auto_accuracy(hx,hy,hz, 1.95))
    pad_list.append(temp_pad)
    print('done 0.2')

    
    integration_2_0, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 2.0, accuracy = get_auto_accuracy(hx,hy,hz, 2.0))
    pad_list.append(temp_pad)
    print('done 1.0')
    


    print("after first conv: " + str(time.time()-start))
    
    ave_density_0_00_norm = n.copy()
    ave_density_0_05_norm = get_homo_nondimensional_nave(integration_0_05, 1.0, 0.05)
    ave_density_0_10_norm = get_homo_nondimensional_nave(integration_0_1, 1.0, 0.10)
    ave_density_0_15_norm = get_homo_nondimensional_nave(integration_0_15, 1.0, 0.15)
    ave_density_0_20_norm = get_homo_nondimensional_nave(integration_0_2, 1.0, 0.20)
    ave_density_0_25_norm = get_homo_nondimensional_nave(integration_0_25, 1.0, 0.25)
    ave_density_0_30_norm = get_homo_nondimensional_nave(integration_0_3, 1.0, 0.30)
    ave_density_0_35_norm = get_homo_nondimensional_nave(integration_0_35, 1.0, 0.35)
    ave_density_0_40_norm = get_homo_nondimensional_nave(integration_0_4, 1.0, 0.40)
    ave_density_0_45_norm = get_homo_nondimensional_nave(integration_0_45, 1.0, 0.45)
    ave_density_0_50_norm = get_homo_nondimensional_nave(integration_0_5, 1.0, 0.50)
    ave_density_0_55_norm = get_homo_nondimensional_nave(integration_0_55, 1.0, 0.55)
    ave_density_0_60_norm = get_homo_nondimensional_nave(integration_0_6, 1.0, 0.60)
    ave_density_0_65_norm = get_homo_nondimensional_nave(integration_0_65, 1.0, 0.65)
    ave_density_0_70_norm = get_homo_nondimensional_nave(integration_0_7, 1.0, 0.70)
    ave_density_0_75_norm = get_homo_nondimensional_nave(integration_0_75, 1.0, 0.75)
    ave_density_0_80_norm = get_homo_nondimensional_nave(integration_0_8, 1.0, 0.80)
    ave_density_0_85_norm = get_homo_nondimensional_nave(integration_0_85, 1.0, 0.85)
    ave_density_0_90_norm = get_homo_nondimensional_nave(integration_0_9, 1.0, 0.90)
    ave_density_0_95_norm = get_homo_nondimensional_nave(integration_0_95, 1.0, 0.95)
    ave_density_1_0_norm = get_homo_nondimensional_nave(integration_1_0, 1.0, 1.0)
    ave_density_1_05_norm = get_homo_nondimensional_nave(integration_1_05, 1.0, 1.05)
    ave_density_1_10_norm = get_homo_nondimensional_nave(integration_1_1, 1.0, 1.10)
    ave_density_1_15_norm = get_homo_nondimensional_nave(integration_1_15, 1.0, 1.15)
    ave_density_1_20_norm = get_homo_nondimensional_nave(integration_1_2, 1.0, 1.20)
    ave_density_1_25_norm = get_homo_nondimensional_nave(integration_1_25, 1.0, 1.25)
    ave_density_1_30_norm = get_homo_nondimensional_nave(integration_1_3, 1.0, 1.30)
    ave_density_1_35_norm = get_homo_nondimensional_nave(integration_1_35, 1.0, 1.35)
    ave_density_1_40_norm = get_homo_nondimensional_nave(integration_1_4, 1.0, 1.40)
    ave_density_1_45_norm = get_homo_nondimensional_nave(integration_1_45, 1.0, 1.45)
    ave_density_1_50_norm = get_homo_nondimensional_nave(integration_1_5, 1.0, 1.50)
    ave_density_1_55_norm = get_homo_nondimensional_nave(integration_1_55, 1.0, 1.55)
    ave_density_1_60_norm = get_homo_nondimensional_nave(integration_1_6, 1.0, 1.60)
    ave_density_1_65_norm = get_homo_nondimensional_nave(integration_1_65, 1.0, 1.65)
    ave_density_1_70_norm = get_homo_nondimensional_nave(integration_1_7, 1.0, 1.70)
    ave_density_1_75_norm = get_homo_nondimensional_nave(integration_1_75, 1.0, 1.75)
    ave_density_1_80_norm = get_homo_nondimensional_nave(integration_1_8, 1.0, 1.80)
    ave_density_1_85_norm = get_homo_nondimensional_nave(integration_1_85, 1.0, 1.85)
    ave_density_1_90_norm = get_homo_nondimensional_nave(integration_1_9, 1.0, 1.90)
    ave_density_1_95_norm = get_homo_nondimensional_nave(integration_1_95, 1.0, 1.95)
    ave_density_2_0_norm = get_homo_nondimensional_nave(integration_2_0, 1.0, 2.0)




    print("after everything: " + str(time.time()-start)    )


    if periodic:
        result = zip(   x.flatten().tolist() ,\
                        y.flatten().tolist() ,\
                        z.flatten().tolist() ,\
                        dimx.flatten().tolist() ,\
                        dimy.flatten().tolist() ,\
                        dimz.flatten().tolist() ,\
                        hx_.flatten().tolist() ,\
                        hy_.flatten().tolist() ,\
                        hz_.flatten().tolist() ,\
                        num_e_.flatten().tolist() ,\
                        n_plot.flatten().tolist() ,\
                        V_xc.flatten().tolist() ,\
                        ep_xc.flatten().tolist() ,\
                        tau.flatten().tolist() ,\
                        gamma.flatten().tolist() ,\
                        ave_density_0_00_norm.flatten().tolist() ,\
                        ave_density_0_05_norm.flatten().tolist() ,\
                        ave_density_0_10_norm.flatten().tolist() ,\
                        ave_density_0_15_norm.flatten().tolist() ,\
                        ave_density_0_20_norm.flatten().tolist() ,\
                        ave_density_0_25_norm.flatten().tolist() ,\
                        ave_density_0_30_norm.flatten().tolist() ,\
                        ave_density_0_35_norm.flatten().tolist() ,\
                        ave_density_0_40_norm.flatten().tolist() ,\
                        ave_density_0_45_norm.flatten().tolist() ,\
                        ave_density_0_50_norm.flatten().tolist() ,\
                        ave_density_0_55_norm.flatten().tolist() ,\
                        ave_density_0_60_norm.flatten().tolist() ,\
                        ave_density_0_65_norm.flatten().tolist() ,\
                        ave_density_0_70_norm.flatten().tolist() ,\
                        ave_density_0_75_norm.flatten().tolist() ,\
                        ave_density_0_80_norm.flatten().tolist() ,\
                        ave_density_0_85_norm.flatten().tolist() ,\
                        ave_density_0_90_norm.flatten().tolist() ,\
                        ave_density_0_95_norm.flatten().tolist() ,\
                        ave_density_1_0_norm.flatten().tolist() ,\
                        ave_density_1_05_norm.flatten().tolist() ,\
                        ave_density_1_10_norm.flatten().tolist() ,\
                        ave_density_1_15_norm.flatten().tolist() ,\
                        ave_density_1_20_norm.flatten().tolist() ,\
                        ave_density_1_25_norm.flatten().tolist() ,\
                        ave_density_1_30_norm.flatten().tolist() ,\
                        ave_density_1_35_norm.flatten().tolist() ,\
                        ave_density_1_40_norm.flatten().tolist() ,\
                        ave_density_1_45_norm.flatten().tolist() ,\
                        ave_density_1_50_norm.flatten().tolist() ,\
                        ave_density_1_55_norm.flatten().tolist() ,\
                        ave_density_1_60_norm.flatten().tolist() ,\
                        ave_density_1_65_norm.flatten().tolist() ,\
                        ave_density_1_70_norm.flatten().tolist() ,\
                        ave_density_1_75_norm.flatten().tolist() ,\
                        ave_density_1_80_norm.flatten().tolist() ,\
                        ave_density_1_85_norm.flatten().tolist() ,\
                        ave_density_1_90_norm.flatten().tolist() ,\
                        ave_density_1_95_norm.flatten().tolist() ,\
                        ave_density_2_0_norm.flatten().tolist())
                        
                        

    else:
        padx, pady, padz = get_pads(pad_list)
  
        result = zip(   x[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        y[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        z[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        dimx[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        dimy[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        dimz[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        hx_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        hy_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        hz_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        num_e_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        n_plot[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        V_xc[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ep_xc[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        tau[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        gamma[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_00_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_05_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_10_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_15_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_20_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_25_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_30_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_35_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_40_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_45_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_50_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_55_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_60_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_65_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_70_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_75_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_80_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_85_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_90_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_95_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_0_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist()  ,\
                        ave_density_1_05_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_10_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_15_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_20_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_25_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_30_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_35_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_40_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_45_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_50_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_55_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_60_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_65_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_70_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_75_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_80_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_85_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_90_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_95_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_2_0_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() )

  

    print("after zip: " + str(time.time()-start))

    return result
    
    
    
    
    
    
    
    
    
    
    
def get_discriptors_from_density_integral_simple_norm_psi4_test_extra(hx,hy,hz, n, num_e, V_xc, ep_xc, tau, gamma, periodic = False, integral_accuracy = 4):
    '''
    get the first-fourth derivative from the density matrix using convolution
    get the integration convolution at 0.5, 1.0, 1.5, 2.0, 2.5
    '''
    def get_xyz_descriptors(n, hx, hy, hz, num_e):

        dimx = np.ones(n.shape)*n.shape[0]
        dimy = np.ones(n.shape)*n.shape[1]
        dimz = np.ones(n.shape)*n.shape[2]
      
#        nt_ave = It / float(hx*hy*hz) 
                
        hx_ = np.ones_like(n)*hx
        hy_ = np.ones_like(n)*hy
        hz_ = np.ones_like(n)*hz
        num_e_ = np.ones_like(n)*float(num_e)
#        It_ = np.ones_like(n)*float(It)
             
        x = np.ones(n.shape)
        y = np.ones(n.shape)
        z = np.ones(n.shape)
        
        for index, density in np.ndenumerate(n):
            x[index[0]][index[1]][index[2]] = index[0]
            y[index[0]][index[1]][index[2]] = index[1]
            z[index[0]][index[1]][index[2]] = index[2]
        
        return x, y, z, dimx, dimy, dimz, hx_, hy_, hz_, num_e_

    n_plot = n.copy()

    pad_list = []  
    result = []
    x, y, z, dimx, dimy, dimz, hx_, hy_, hz_, num_e_= get_xyz_descriptors(n, hx, hy, hz, num_e)
    
    
    start = time.time()    
	
    
    print('\n\ngetting integration convolutions...')
    print(hx)
    print(hy)
    print(hz)
    print(max(hx, hy, hz) - min(hx, hy, hz))
    print(round(((hx + hy + hz) / 3.), 3))
#    try:
#        if max(hx, hy, hz) - min(hx, hy, hz) >= 0.01:
#            raise NotImplementedError
#        stencil_data = read_integration_stencil_file(hx, hy, hz)
#        print("loaded stencil data from file")
#        
#        integration_0_05, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.05, stencil_data[str(0.05).replace(".","_")], stencil_data[str(0.05).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_1, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.1, stencil_data[str(0.1).replace(".","_")], stencil_data[str(0.1).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_15, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.15, stencil_data[str(0.15).replace(".","_")], stencil_data[str(0.15).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_2, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.2, stencil_data[str(0.2).replace(".","_")], stencil_data[str(0.2).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#
#        integration_0_25, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.25, stencil_data[str(0.25).replace(".","_")], stencil_data[str(0.25).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#
#        integration_0_3, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.3, stencil_data[str(0.3).replace(".","_")], stencil_data[str(0.3).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_35, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.35, stencil_data[str(0.35).replace(".","_")], stencil_data[str(0.35).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_4, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.4, stencil_data[str(0.4).replace(".","_")], stencil_data[str(0.4).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_45, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.45, stencil_data[str(0.45).replace(".","_")], stencil_data[str(0.45).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_5, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.5, stencil_data[str(0.5).replace(".","_")], stencil_data[str(0.5).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_55, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.55, stencil_data[str(0.55).replace(".","_")], stencil_data[str(0.55).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        
#        integration_0_6, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.6, stencil_data[str(0.6).replace(".","_")], stencil_data[str(0.6).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_65, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.65, stencil_data[str(0.65).replace(".","_")], stencil_data[str(0.65).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        
#        integration_0_7, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.7, stencil_data[str(0.7).replace(".","_")], stencil_data[str(0.7).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_75, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.75, stencil_data[str(0.75).replace(".","_")], stencil_data[str(0.75).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        
#        integration_0_8, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.8, stencil_data[str(0.8).replace(".","_")], stencil_data[str(0.8).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_85, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.85, stencil_data[str(0.85).replace(".","_")], stencil_data[str(0.85).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        
#        integration_0_9, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.9, stencil_data[str(0.9).replace(".","_")], stencil_data[str(0.9).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        integration_0_95, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 0.95, stencil_data[str(0.95).replace(".","_")], stencil_data[str(0.95).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#        
#        
#        integration_1_0, temp_pad = get_integral_fftconv_with_known_stencil(n.copy(), hx, hy, hz, 1.0, stencil_data[str(1.0).replace(".","_")], stencil_data[str(1.0).replace(".","_") + "pad"])
#        pad_list.append(temp_pad)
#
#    except:
    integration_0_05, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.05, accuracy = get_auto_accuracy(hx,hy,hz, 0.05))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_0_1, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.1, accuracy = get_auto_accuracy(hx,hy,hz, 0.1))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_0_15, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.15, accuracy = get_auto_accuracy(hx,hy,hz, 0.15))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_0_2, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.2, accuracy = get_auto_accuracy(hx,hy,hz, 0.2))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_0_25, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.25, accuracy = get_auto_accuracy(hx,hy,hz, 0.25))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_0_3, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.3, accuracy = get_auto_accuracy(hx,hy,hz, 0.3))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_0_35, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.35, accuracy = get_auto_accuracy(hx,hy,hz, 0.35))
    pad_list.append(temp_pad)
    print('done 0.2')
    
    integration_0_4, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.4, accuracy = get_auto_accuracy(hx,hy,hz, 0.4))
    pad_list.append(temp_pad)
    print('done 0.4')

    integration_0_45, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.45, accuracy = get_auto_accuracy(hx,hy,hz, 0.45))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_0_5, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.5, accuracy = get_auto_accuracy(hx,hy,hz, 0.5))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_0_55, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.55, accuracy = get_auto_accuracy(hx,hy,hz, 0.55))
    pad_list.append(temp_pad)
    print('done 0.2')

  
    integration_0_6, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.6, accuracy = get_auto_accuracy(hx,hy,hz, 0.6))
    pad_list.append(temp_pad)
    print('done 0.6')

    integration_0_65, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.65, accuracy = get_auto_accuracy(hx,hy,hz, 0.65))
    pad_list.append(temp_pad)
    print('done 0.2')


    integration_0_7, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.7, accuracy = get_auto_accuracy(hx,hy,hz, 0.7))
    pad_list.append(temp_pad)
    print('done 0.8')

    integration_0_75, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.75, accuracy = get_auto_accuracy(hx,hy,hz, 0.75))
    pad_list.append(temp_pad)
    print('done 0.2')

    
    integration_0_8, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.8, accuracy = get_auto_accuracy(hx,hy,hz, 0.8))
    pad_list.append(temp_pad)
    print('done 0.8')

    integration_0_85, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.85, accuracy = get_auto_accuracy(hx,hy,hz, 0.85))
    pad_list.append(temp_pad)
    print('done 0.2')


    integration_0_9, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.9, accuracy = get_auto_accuracy(hx,hy,hz, 0.9))
    pad_list.append(temp_pad)
    print('done 0.8')

    integration_0_95, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 0.95, accuracy = get_auto_accuracy(hx,hy,hz, 0.95))
    pad_list.append(temp_pad)
    print('done 0.2')

    
    integration_1_0, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 1.0, accuracy = get_auto_accuracy(hx,hy,hz, 1.0))
    pad_list.append(temp_pad)
    print('done 1.0')

    integration_1_1, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 1.1, accuracy = get_auto_accuracy(hx,hy,hz, 1.1))
    pad_list.append(temp_pad)
    print('done 0.2')


    integration_1_2, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 1.2, accuracy = get_auto_accuracy(hx,hy,hz, 1.2))
    pad_list.append(temp_pad)
    print('done 0.2')

    integration_1_3, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 1.3, accuracy = get_auto_accuracy(hx,hy,hz, 1.3))
    pad_list.append(temp_pad)
    print('done 0.2')

    
    integration_1_4, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 1.4, accuracy = get_auto_accuracy(hx,hy,hz, 1.4))
    pad_list.append(temp_pad)
    print('done 0.4')


    integration_1_5, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 1.5, accuracy = get_auto_accuracy(hx,hy,hz, 1.5))
    pad_list.append(temp_pad)
    print('done 0.2')


  
    integration_1_6, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 1.6, accuracy = get_auto_accuracy(hx,hy,hz, 1.6))
    pad_list.append(temp_pad)
    print('done 0.6')



    integration_1_7, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 1.7, accuracy = get_auto_accuracy(hx,hy,hz, 1.7))
    pad_list.append(temp_pad)
    print('done 0.8')


    
    integration_1_8, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 1.8, accuracy = get_auto_accuracy(hx,hy,hz, 1.8))
    pad_list.append(temp_pad)
    print('done 0.8')



    integration_1_9, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 1.9, accuracy = get_auto_accuracy(hx,hy,hz, 1.9))
    pad_list.append(temp_pad)
    print('done 0.8')

    
    integration_2_0, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 2.0, accuracy = get_auto_accuracy(hx,hy,hz, 2.0))
    pad_list.append(temp_pad)
    print('done 1.0')


    integration_2_1, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 2.1, accuracy = get_auto_accuracy(hx,hy,hz, 2.1))
    pad_list.append(temp_pad)
    print('done 0.2')


    integration_2_2, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 2.2, accuracy = get_auto_accuracy(hx,hy,hz, 2.2))
    pad_list.append(temp_pad)
    print('done 0.2')


    integration_2_3, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 2.3, accuracy = get_auto_accuracy(hx,hy,hz, 2.3))
    pad_list.append(temp_pad)
    print('done 0.2')

    
    integration_2_4, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 2.4, accuracy = get_auto_accuracy(hx,hy,hz, 2.4))
    pad_list.append(temp_pad)
    print('done 0.4')


    integration_2_5, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 2.5, accuracy = get_auto_accuracy(hx,hy,hz, 2.5))
    pad_list.append(temp_pad)
    print('done 0.2')


  
    integration_2_6, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 2.6, accuracy = get_auto_accuracy(hx,hy,hz, 2.6))
    pad_list.append(temp_pad)
    print('done 0.6')



    integration_2_7, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 2.7, accuracy = get_auto_accuracy(hx,hy,hz, 2.7))
    pad_list.append(temp_pad)
    print('done 0.8')


    
    integration_2_8, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 2.8, accuracy = get_auto_accuracy(hx,hy,hz, 2.8))
    pad_list.append(temp_pad)
    print('done 0.8')


    integration_2_9, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 2.9, accuracy = get_auto_accuracy(hx,hy,hz, 2.9))
    pad_list.append(temp_pad)
    print('done 0.8')
    
    integration_3_0, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 3.0, accuracy = get_auto_accuracy(hx,hy,hz, 3.0))
    pad_list.append(temp_pad)
    print('done 1.0')


    integration_3_1, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 3.1, accuracy = get_auto_accuracy(hx,hy,hz, 3.1))
    pad_list.append(temp_pad)
    print('done 0.2')


    integration_3_2, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 3.2, accuracy = get_auto_accuracy(hx,hy,hz, 3.2))
    pad_list.append(temp_pad)
    print('done 0.2')


    integration_3_3, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 3.3, accuracy = get_auto_accuracy(hx,hy,hz, 3.3))
    pad_list.append(temp_pad)
    print('done 0.2')

    
    integration_3_4, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 3.4, accuracy = get_auto_accuracy(hx,hy,hz, 3.4))
    pad_list.append(temp_pad)
    print('done 0.4')

    integration_3_5, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 3.5, accuracy = get_auto_accuracy(hx,hy,hz, 3.5))
    pad_list.append(temp_pad)
    print('done 0.2')

  
    integration_3_6, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 3.6, accuracy = get_auto_accuracy(hx,hy,hz, 3.6))
    pad_list.append(temp_pad)
    print('done 0.6')



    integration_3_7, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 3.7, accuracy = get_auto_accuracy(hx,hy,hz, 3.7))
    pad_list.append(temp_pad)
    print('done 0.8')


    
    integration_3_8, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 3.8, accuracy = get_auto_accuracy(hx,hy,hz, 3.8))
    pad_list.append(temp_pad)
    print('done 0.8')



    integration_3_9, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 3.9, accuracy = get_auto_accuracy(hx,hy,hz, 3.9))
    pad_list.append(temp_pad)
    print('done 0.8')


    
    integration_4_0, temp_pad  = get_integration_fftconv(n.copy(), hx, hy, hz, 4.0, accuracy = get_auto_accuracy(hx,hy,hz, 4.0))
    pad_list.append(temp_pad)
    print('done 1.0')

    print("after first conv: " + str(time.time()-start))
    
    ave_density_0_00_norm = n.copy()
    ave_density_0_05_norm = get_homo_nondimensional_nave(integration_0_05, 1.0, 0.05)
    ave_density_0_10_norm = get_homo_nondimensional_nave(integration_0_1, 1.0, 0.10)
    ave_density_0_15_norm = get_homo_nondimensional_nave(integration_0_15, 1.0, 0.15)
    ave_density_0_20_norm = get_homo_nondimensional_nave(integration_0_2, 1.0, 0.20)
    ave_density_0_25_norm = get_homo_nondimensional_nave(integration_0_25, 1.0, 0.25)
    ave_density_0_30_norm = get_homo_nondimensional_nave(integration_0_3, 1.0, 0.30)
    ave_density_0_35_norm = get_homo_nondimensional_nave(integration_0_35, 1.0, 0.35)
    ave_density_0_40_norm = get_homo_nondimensional_nave(integration_0_4, 1.0, 0.40)
    ave_density_0_45_norm = get_homo_nondimensional_nave(integration_0_45, 1.0, 0.45)
    ave_density_0_50_norm = get_homo_nondimensional_nave(integration_0_5, 1.0, 0.50)
    ave_density_0_55_norm = get_homo_nondimensional_nave(integration_0_55, 1.0, 0.55)
    ave_density_0_60_norm = get_homo_nondimensional_nave(integration_0_6, 1.0, 0.60)
    ave_density_0_65_norm = get_homo_nondimensional_nave(integration_0_65, 1.0, 0.65)
    ave_density_0_70_norm = get_homo_nondimensional_nave(integration_0_7, 1.0, 0.70)
    ave_density_0_75_norm = get_homo_nondimensional_nave(integration_0_75, 1.0, 0.75)
    ave_density_0_80_norm = get_homo_nondimensional_nave(integration_0_8, 1.0, 0.80)
    ave_density_0_85_norm = get_homo_nondimensional_nave(integration_0_85, 1.0, 0.85)
    ave_density_0_90_norm = get_homo_nondimensional_nave(integration_0_9, 1.0, 0.90)
    ave_density_0_95_norm = get_homo_nondimensional_nave(integration_0_95, 1.0, 0.95)
    ave_density_1_0_norm = get_homo_nondimensional_nave(integration_1_0, 1.0, 1.0)
    ave_density_1_10_norm = get_homo_nondimensional_nave(integration_1_1, 1.0, 1.10)
    ave_density_1_20_norm = get_homo_nondimensional_nave(integration_1_2, 1.0, 1.20)
    ave_density_1_30_norm = get_homo_nondimensional_nave(integration_1_3, 1.0, 1.30)
    ave_density_1_40_norm = get_homo_nondimensional_nave(integration_1_4, 1.0, 1.40)
    ave_density_1_50_norm = get_homo_nondimensional_nave(integration_1_5, 1.0, 1.50)
    ave_density_1_60_norm = get_homo_nondimensional_nave(integration_1_6, 1.0, 1.60)
    ave_density_1_70_norm = get_homo_nondimensional_nave(integration_1_7, 1.0, 1.70)
    ave_density_1_80_norm = get_homo_nondimensional_nave(integration_1_8, 1.0, 1.80)
    ave_density_1_90_norm = get_homo_nondimensional_nave(integration_1_9, 1.0, 1.90)
    ave_density_2_0_norm = get_homo_nondimensional_nave(integration_2_0, 1.0, 2.0)
    ave_density_2_10_norm = get_homo_nondimensional_nave(integration_2_1, 1.0, 2.10)
    ave_density_2_20_norm = get_homo_nondimensional_nave(integration_2_2, 1.0, 2.20)
    ave_density_2_30_norm = get_homo_nondimensional_nave(integration_2_3, 1.0, 2.30)
    ave_density_2_40_norm = get_homo_nondimensional_nave(integration_2_4, 1.0, 2.40)
    ave_density_2_50_norm = get_homo_nondimensional_nave(integration_2_5, 1.0, 2.50)
    ave_density_2_60_norm = get_homo_nondimensional_nave(integration_2_6, 1.0, 2.60)
    ave_density_2_70_norm = get_homo_nondimensional_nave(integration_2_7, 1.0, 2.70)
    ave_density_2_80_norm = get_homo_nondimensional_nave(integration_2_8, 1.0, 2.80)
    ave_density_2_90_norm = get_homo_nondimensional_nave(integration_2_9, 1.0, 2.90)
    ave_density_3_0_norm = get_homo_nondimensional_nave(integration_3_0, 1.0, 3.0)
    ave_density_3_10_norm = get_homo_nondimensional_nave(integration_3_1, 1.0, 3.10)
    ave_density_3_20_norm = get_homo_nondimensional_nave(integration_3_2, 1.0, 3.20)
    ave_density_3_30_norm = get_homo_nondimensional_nave(integration_3_3, 1.0, 3.30)
    ave_density_3_40_norm = get_homo_nondimensional_nave(integration_3_4, 1.0, 3.40)
    ave_density_3_50_norm = get_homo_nondimensional_nave(integration_3_5, 1.0, 3.50)
    ave_density_3_60_norm = get_homo_nondimensional_nave(integration_3_6, 1.0, 3.60)
    ave_density_3_70_norm = get_homo_nondimensional_nave(integration_3_7, 1.0, 3.70)
    ave_density_3_80_norm = get_homo_nondimensional_nave(integration_3_8, 1.0, 3.80)
    ave_density_3_90_norm = get_homo_nondimensional_nave(integration_3_9, 1.0, 3.90)
    ave_density_4_0_norm = get_homo_nondimensional_nave(integration_4_0, 1.0, 4.0)




    print("after everything: " + str(time.time()-start)    )


    if periodic:
        result = zip(   x.flatten().tolist() ,\
                        y.flatten().tolist() ,\
                        z.flatten().tolist() ,\
                        dimx.flatten().tolist() ,\
                        dimy.flatten().tolist() ,\
                        dimz.flatten().tolist() ,\
                        hx_.flatten().tolist() ,\
                        hy_.flatten().tolist() ,\
                        hz_.flatten().tolist() ,\
                        num_e_.flatten().tolist() ,\
                        n_plot.flatten().tolist() ,\
                        V_xc.flatten().tolist() ,\
                        ep_xc.flatten().tolist() ,\
                        tau.flatten().tolist() ,\
                        gamma.flatten().tolist() ,\
                        ave_density_0_00_norm.flatten().tolist() ,\
                        ave_density_0_05_norm.flatten().tolist() ,\
                        ave_density_0_10_norm.flatten().tolist() ,\
                        ave_density_0_15_norm.flatten().tolist() ,\
                        ave_density_0_20_norm.flatten().tolist() ,\
                        ave_density_0_25_norm.flatten().tolist() ,\
                        ave_density_0_30_norm.flatten().tolist() ,\
                        ave_density_0_35_norm.flatten().tolist() ,\
                        ave_density_0_40_norm.flatten().tolist() ,\
                        ave_density_0_45_norm.flatten().tolist() ,\
                        ave_density_0_50_norm.flatten().tolist() ,\
                        ave_density_0_55_norm.flatten().tolist() ,\
                        ave_density_0_60_norm.flatten().tolist() ,\
                        ave_density_0_65_norm.flatten().tolist() ,\
                        ave_density_0_70_norm.flatten().tolist() ,\
                        ave_density_0_75_norm.flatten().tolist() ,\
                        ave_density_0_80_norm.flatten().tolist() ,\
                        ave_density_0_85_norm.flatten().tolist() ,\
                        ave_density_0_90_norm.flatten().tolist() ,\
                        ave_density_0_95_norm.flatten().tolist() ,\
                        ave_density_1_0_norm.flatten().tolist() ,\
                        ave_density_1_10_norm.flatten().tolist() ,\
                        ave_density_1_20_norm.flatten().tolist() ,\
                        ave_density_1_30_norm.flatten().tolist() ,\
                        ave_density_1_40_norm.flatten().tolist() ,\
                        ave_density_1_50_norm.flatten().tolist() ,\
                        ave_density_1_60_norm.flatten().tolist() ,\
                        ave_density_1_70_norm.flatten().tolist() ,\
                        ave_density_1_80_norm.flatten().tolist() ,\
                        ave_density_1_90_norm.flatten().tolist() ,\
                        ave_density_2_0_norm.flatten().tolist() ,\
                        ave_density_2_10_norm.flatten().tolist() ,\
                        ave_density_2_20_norm.flatten().tolist() ,\
                        ave_density_2_30_norm.flatten().tolist() ,\
                        ave_density_2_40_norm.flatten().tolist() ,\
                        ave_density_2_50_norm.flatten().tolist() ,\
                        ave_density_2_60_norm.flatten().tolist() ,\
                        ave_density_2_70_norm.flatten().tolist() ,\
                        ave_density_2_80_norm.flatten().tolist() ,\
                        ave_density_2_90_norm.flatten().tolist() ,\
                        ave_density_3_0_norm.flatten().tolist() ,\
                        ave_density_3_10_norm.flatten().tolist() ,\
                        ave_density_3_20_norm.flatten().tolist() ,\
                        ave_density_3_30_norm.flatten().tolist() ,\
                        ave_density_3_40_norm.flatten().tolist() ,\
                        ave_density_3_50_norm.flatten().tolist() ,\
                        ave_density_3_60_norm.flatten().tolist() ,\
                        ave_density_3_70_norm.flatten().tolist() ,\
                        ave_density_3_80_norm.flatten().tolist() ,\
                        ave_density_3_90_norm.flatten().tolist() ,\
                        ave_density_4_0_norm.flatten().tolist())
                        
                        

    else:
        padx, pady, padz = get_pads(pad_list)
  
        result = zip(   x[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        y[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        z[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        dimx[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        dimy[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        dimz[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        hx_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        hy_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        hz_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        num_e_[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        n_plot[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        V_xc[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ep_xc[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        tau[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        gamma[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_00_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_05_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_10_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_15_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_20_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_25_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_30_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_35_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_40_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_45_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_50_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_55_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_60_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_65_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_70_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_75_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_80_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_85_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_90_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_0_95_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_0_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist()  ,\
                        ave_density_1_10_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_20_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_30_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_40_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_50_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_60_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_70_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_80_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_1_90_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_2_0_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_2_10_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_2_20_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_2_30_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_2_40_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_2_50_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_2_60_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_2_70_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_2_80_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_2_90_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_3_0_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist()  ,\
                        ave_density_3_10_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_3_20_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_3_30_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_3_40_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_3_50_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_3_60_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_3_70_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_3_80_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_3_90_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist() ,\
                        ave_density_4_0_norm[padx:-padx,pady:-pady,padz:-padz].flatten().tolist()  )

  

    print("after zip: " + str(time.time()-start))

    return result

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:48:37 2017

@author: ray
"""

import numpy as np
from scipy.ndimage.filters import convolve
from scipy.signal import fftconvolve
from math import pi
import math
import os
try: import cPickle as pickle
except: import pickle




'''
Differenciations

'''


def get_first_grad_stencil(n, hx, hy, hz, stencil_type = 'mid', accuracy = '2'):
    '''
    n: electron density, or whatever we need to convolve
    hx, hy, hz: grid spacing at each direction
    stencil type:   mid: only mid-row has elements
                    uniform: all rows have the same elements in each dimension
                    times2:  emphasize on the middle rows
    accuracy: degree of accuracy of the finite difference method uses
    '''
                       
    fd_coefficients = {}
                       
    fd_coefficients['2'] = {'coeff':    np.asarray([-1., 0., 1.])* -1., 
                            'mid':      np.asarray([0., 1., 0.]), 
                            'uniform':  np.asarray([1., 1., 1.]) * (1./3.),
                            'times2':   np.asarray([1., 2., 1.]) * (1./4.),
                            'pad':      1,
                            'norm_fac': 2.}
                
    fd_coefficients['4'] = {'coeff':    np.asarray([1., -8., 0., 8., -1.])* -1., 
                            'mid':      np.asarray([0., 0., 1., 0., 0.]), 
                            'uniform':  np.asarray([1., 1., 1., 1., 1.]) * (1./5.),
                            'times2':   np.asarray([1., 2., 4., 2., 1.]) * (1./10.),
                            'pad':      2,
                            'norm_fac': 12.}
                            
    fd_coefficients['6'] = {'coeff':    np.asarray([-1., 9., -45., 0., 45., -9., 1.])* -1., 
                            'mid':      np.asarray([0., 0., 0., 1., 0., 0., 0.]), 
                            'uniform':  np.asarray([1., 1., 1., 1., 1., 1., 1.]) * (1./7.),
                            'times2':   np.asarray([1., 2., 4., 8., 4., 2., 1.]) * (1./22.),
                            'pad':      3,
                            'norm_fac': 60.}
                            
    fd_coefficients['8'] = {'coeff':    np.asarray([3., -32., 168., -672., 0., 672., -168., 32., 3.])* -1., 
                            'mid':      np.asarray([0., 0., 0., 0., 1., 0., 0., 0., 0.]), 
                            'uniform':  np.asarray([1., 1., 1., 1., 1., 1., 1., 1., 1.]) * (1./9.),
                            'times2':   np.asarray([1., 2., 4., 8.,16., 8., 4., 2., 1.]) * (1./46.),
                            'pad':      4,
                            'norm_fac': 840.}
                            
    if accuracy not in list(fd_coefficients.keys()):
        raise NotImplementedError
    
    if stencil_type not in list(fd_coefficients[accuracy].keys()):
        raise NotImplementedError
    
    
    coefficient = np.asarray(fd_coefficients[accuracy]['coeff'])
    extend_mat  = np.asarray(fd_coefficients[accuracy][stencil_type])
    pad =  (fd_coefficients[accuracy]['pad'],)*3
    normalization_factor = fd_coefficients[accuracy]['norm_fac']
    temp_num = len(extend_mat)
    
    # get the stencils based on finite difference coefficients
    # transpose the stencil on x direction to get the other ones
    Gx_temp = np.reshape(extend_mat,(temp_num,1,1)) * ( np.reshape(extend_mat,(1,temp_num,1)) * coefficient)
    Gy_temp = Gx_temp.copy().transpose(0,2,1)
    Gz_temp = Gx_temp.copy().transpose(2,1,0)
    
    Gx = (1./(normalization_factor*hx)) * Gx_temp.copy()
    Gy = (1./(normalization_factor*hy)) * Gy_temp.copy()
    Gz = (1./(normalization_factor*hz)) * Gz_temp.copy()
    
    G = Gx + Gy + Gz
    
    return G, Gx, Gy, Gz, pad
    

def get_second_grad_stencil(n, hx, hy, hz, stencil_type = 'mid', accuracy = '2'):
                       
    fd_coefficients = {}
                       
    fd_coefficients['2'] = {'coeff':    np.asarray([1.,-2., 1.]), 
                            'mid':      np.asarray([0., 1., 0.]), 
                            'uniform':  np.asarray([1., 1., 1.]) * (1./3.),
                            'times2':   np.asarray([1., 2., 1.]) * (1./4.),
                            'pad':      1,
                            'norm_fac': 1.}
                
    fd_coefficients['4'] = {'coeff':    np.asarray([-1., 16., -30., 16., -1.]), 
                            'mid':      np.asarray([0., 0., 1., 0., 0.]), 
                            'uniform':  np.asarray([1., 1., 1., 1., 1.]) * (1./5.),
                            'times2':   np.asarray([1., 2., 4., 2., 1.]) * (1./10.),
                            'pad':      2,
                            'norm_fac': 12.}
                            
    fd_coefficients['6'] = {'coeff':    np.asarray([2., -27.,  270., -490., 270., -27., 2.]), 
                            'mid':      np.asarray([0., 0., 0., 1., 0., 0., 0.]), 
                            'uniform':  np.asarray([1., 1., 1., 1., 1., 1., 1.]) * (1./7.),
                            'times2':   np.asarray([1., 2., 4., 8., 4., 2., 1.]) * (1./22.),
                            'pad':      3,
                            'norm_fac': 180.}
                            
    fd_coefficients['8'] = {'coeff':    np.asarray([-9., 128., -1008., 8064., -14350., 8064., -1008., 128., -9.]), 
                            'mid':      np.asarray([0., 0., 0., 0., 1., 0., 0., 0., 0.]), 
                            'uniform':  np.asarray([1., 1., 1., 1., 1., 1., 1., 1., 1.]) * (1./9.),
                            'times2':   np.asarray([1., 2., 4., 8.,16., 8., 4., 2., 1.]) * (1./46.),
                            'pad':      4,
                            'norm_fac': 5040.}
                            
    if accuracy not in list(fd_coefficients.keys()):
        raise NotImplementedError
    
    if stencil_type not in list(fd_coefficients[accuracy].keys()):
        raise NotImplementedError
    
    
    coefficient = np.asarray(fd_coefficients[accuracy]['coeff'])
    extend_mat  = np.asarray(fd_coefficients[accuracy][stencil_type])
    pad =  (fd_coefficients[accuracy]['pad'],)*3
    normalization_factor = fd_coefficients[accuracy]['norm_fac']
    temp_num = len(extend_mat)
    
    Gx_temp = np.reshape(extend_mat,(temp_num,1,1)) * ( np.reshape(extend_mat,(1,temp_num,1)) * coefficient)
    Gy_temp = Gx_temp.copy().transpose(0,2,1)
    Gz_temp = Gx_temp.copy().transpose(2,1,0)
    
    Gx = (1./(normalization_factor*hx*hx)) * Gx_temp.copy()
    Gy = (1./(normalization_factor*hy*hy)) * Gy_temp.copy()
    Gz = (1./(normalization_factor*hz*hz)) * Gz_temp.copy()
#    print 'start conv'
    G = Gx + Gy + Gz
    
    return G, Gx, Gy, Gz, pad
    


def get_third_grad_stencil(n, hx, hy, hz, stencil_type = 'mid', accuracy = '2'):
                       
    fd_coefficients = {}
               
    fd_coefficients['2'] = {'coeff':    np.asarray([-1., 2., 0., -2., 1.])* -1., 
                            'mid':      np.asarray([0., 0., 1., 0., 0.]), 
                            'uniform':  np.asarray([1., 1., 1., 1., 1.]) * (1./5.),
                            'times2':   np.asarray([1., 2., 4., 2., 1.]) * (1./10.),
                            'pad':      2,
                            'norm_fac': 2.}
                            
    fd_coefficients['4'] = {'coeff':    np.asarray([1., -8., 13., 0., -13., 8., -1.])* -1., 
                            'mid':      np.asarray([0., 0., 0., 1., 0., 0., 0.]), 
                            'uniform':  np.asarray([1., 1., 1., 1., 1., 1., 1.]) * (1./7.),
                            'times2':   np.asarray([1., 2., 4., 8., 4., 2., 1.]) * (1./22.),
                            'pad':      3,
                            'norm_fac': 8.}
                            
    fd_coefficients['6'] = {'coeff':    np.asarray([-7., 72., -338., 488., 0., -488., 338., -72., 7.])* -1., 
                            'mid':      np.asarray([0., 0., 0., 0., 1., 0., 0., 0., 0.]), 
                            'uniform':  np.asarray([1., 1., 1., 1., 1., 1., 1., 1., 1.]) * (1./9.),
                            'times2':   np.asarray([1., 2., 4., 8.,16., 8., 4., 2., 1.]) * (1./46.),
                            'pad':      4,
                            'norm_fac': 240.}#434400.
                            
    if accuracy not in list(fd_coefficients.keys()):
        raise NotImplementedError
    
    if stencil_type not in list(fd_coefficients[accuracy].keys()):
        raise NotImplementedError
    
    
    coefficient = np.asarray(fd_coefficients[accuracy]['coeff'])
    extend_mat  = np.asarray(fd_coefficients[accuracy][stencil_type])
    pad =  (fd_coefficients[accuracy]['pad'],)*3
    normalization_factor = fd_coefficients[accuracy]['norm_fac']
    temp_num = len(extend_mat)
    
    Gx_temp = np.reshape(extend_mat,(temp_num,1,1)) * ( np.reshape(extend_mat,(1,temp_num,1)) * coefficient)
    Gy_temp = Gx_temp.copy().transpose(0,2,1)
    Gz_temp = Gx_temp.copy().transpose(2,1,0)
    
    Gx = (1./(normalization_factor*hx*hx*hx)) * Gx_temp.copy()
    Gy = (1./(normalization_factor*hy*hy*hy)) * Gy_temp.copy()
    Gz = (1./(normalization_factor*hz*hz*hz)) * Gz_temp.copy()
    
    G = Gx + Gy + Gz
    
    return G, Gx, Gy, Gz, pad   




def get_fourth_grad_stencil(n, hx, hy, hz, stencil_type = 'mid', accuracy = '2'):
                       
    fd_coefficients = {}
               
    fd_coefficients['2'] = {'coeff':    np.asarray([1., -4., 6., -4., 1.]), 
                            'mid':      np.asarray([0., 0., 1., 0., 0.]), 
                            'uniform':  np.asarray([1., 1., 1., 1., 1.]) * (1./5.),
                            'times2':   np.asarray([1., 2., 4., 2., 1.]) * (1./10.),
                            'pad':      2,
                            'norm_fac': 1.}
                            
    fd_coefficients['4'] = {'coeff':    np.asarray([-1., 12., -78., 112., -78., 12., -1.]), 
                            'mid':      np.asarray([0., 0., 0., 1., 0., 0., 0.]), 
                            'uniform':  np.asarray([1., 1., 1., 1., 1., 1., 1.]) * (1./7.),
                            'times2':   np.asarray([1., 2., 4., 8., 4., 2., 1.]) * (1./22.),
                            'pad':      3,
                            'norm_fac': 1.}#352
                            
    fd_coefficients['6'] = {'coeff':    np.asarray([7., -76., 676., -1952., 2730., -1952., 676., -76., 7.]), 
                            'mid':      np.asarray([0., 0., 0., 0., 1., 0., 0., 0., 0.]), 
                            'uniform':  np.asarray([1., 1., 1., 1., 1., 1., 1., 1., 1.]) * (1./9.),
                            'times2':   np.asarray([1., 2., 4., 8.,16., 8., 4., 2., 1.]) * (1./46.),
                            'pad':      4,
                            'norm_fac': 1.}#434400.
                            
    if accuracy not in list(fd_coefficients.keys()):
        raise NotImplementedError
    
    if stencil_type not in list(fd_coefficients[accuracy].keys()):
        raise NotImplementedError
    
    
    coefficient = np.asarray(fd_coefficients[accuracy]['coeff'])
    extend_mat  = np.asarray(fd_coefficients[accuracy][stencil_type])
    pad =  (fd_coefficients[accuracy]['pad'],)*3
    normalization_factor = fd_coefficients[accuracy]['norm_fac']
    temp_num = len(extend_mat)
    
    Gx_temp = np.reshape(extend_mat,(temp_num,1,1)) * ( np.reshape(extend_mat,(1,temp_num,1)) * coefficient)
    Gy_temp = Gx_temp.copy().transpose(0,2,1)
    Gz_temp = Gx_temp.copy().transpose(2,1,0)
    
    Gx = (1./(normalization_factor*(hx*hx*hx*hx))) * Gx_temp.copy()
    Gy = (1./(normalization_factor*(hy*hy*hy*hy))) * Gy_temp.copy()
    Gz = (1./(normalization_factor*(hz*hz*hz*hz))) * Gz_temp.copy()
    
    G = Gx + Gy + Gz
    
    return G, Gx, Gy, Gz, pad    
    
    
    
def get_fifth_grad_stencil(n, hx, hy, hz, stencil_type = 'mid', accuracy = '2'):
                       
    fd_coefficients = {}
                                          
    fd_coefficients['2'] = {'coeff':    np.asarray([-1., 4., -10., 0., 10., -4., 1.]), 
                            'mid':      np.asarray([0., 0., 0., 1., 0., 0., 0.]), 
                            'uniform':  np.asarray([1., 1., 1., 1., 1., 1., 1.]) * (1./7.),
                            'times2':   np.asarray([1., 2., 4., 8., 4., 2., 1.]) * (1./22.),
                            'pad':      3,
                            'norm_fac': 30.}

                            
    if accuracy not in list(fd_coefficients.keys()):
        raise NotImplementedError
    
    if stencil_type not in list(fd_coefficients[accuracy].keys()):
        raise NotImplementedError
    
    
    coefficient = np.asarray(fd_coefficients[accuracy]['coeff'])
    extend_mat  = np.asarray(fd_coefficients[accuracy][stencil_type])
    pad = (fd_coefficients[accuracy]['pad'],)*3
    normalization_factor = fd_coefficients[accuracy]['norm_fac']
    temp_num = len(extend_mat)
    
    Gx_temp = np.reshape(extend_mat,(temp_num,1,1)) * ( np.reshape(extend_mat,(1,temp_num,1)) * coefficient)
    Gy_temp = Gx_temp.copy().transpose(0,2,1)
    Gz_temp = Gx_temp.copy().transpose(2,1,0)
    
    Gx = (1./(normalization_factor*(hx*hx*hx*hx*hx))) * Gx_temp.copy()
    Gy = (1./(normalization_factor*(hy*hy*hy*hy*hy))) * Gy_temp.copy()
    Gz = (1./(normalization_factor*(hz*hz*hz*hz*hz))) * Gz_temp.copy()

    G = Gx + Gy + Gz
    
    return G, Gx, Gy, Gz, pad    
    
    
    
    
def get_sixth_grad_stencil(n, hx, hy, hz, stencil_type = 'mid', accuracy = '2'):
                       
    fd_coefficients = {}
                                          
    fd_coefficients['2'] = {'coeff':    np.asarray([1., -6., 15., -20., 15., -6., 1.]), 
                            'mid':      np.asarray([0., 0., 0., 1., 0., 0., 0.]), 
                            'uniform':  np.asarray([1., 1., 1., 1., 1., 1., 1.]) * (1./7.),
                            'times2':   np.asarray([1., 2., 4., 8., 4., 2., 1.]) * (1./22.),
                            'pad':      3,
                            'norm_fac': 1.}

                            
    if accuracy not in list(fd_coefficients.keys()):
        raise NotImplementedError
    
    if stencil_type not in list(fd_coefficients[accuracy].keys()):
        raise NotImplementedError
    
    
    coefficient = np.asarray(fd_coefficients[accuracy]['coeff'])
    extend_mat  = np.asarray(fd_coefficients[accuracy][stencil_type])
    pad = (fd_coefficients[accuracy]['pad'],)*3
    normalization_factor = fd_coefficients[accuracy]['norm_fac']
    temp_num = len(extend_mat)
    
    Gx_temp = np.reshape(extend_mat,(temp_num,1,1)) * ( np.reshape(extend_mat,(1,temp_num,1)) * coefficient)
    Gy_temp = Gx_temp.copy().transpose(0,2,1)
    Gz_temp = Gx_temp.copy().transpose(2,1,0)
    
    Gx = (1./(normalization_factor*(hx*hx*hx*hx*hx*hx))) * Gx_temp.copy()
    Gy = (1./(normalization_factor*(hy*hy*hy*hy*hy*hy))) * Gy_temp.copy()
    Gz = (1./(normalization_factor*(hz*hz*hz*hz*hz*hz))) * Gz_temp.copy()

    G = Gx + Gy + Gz
    
    return G, Gx, Gy, Gz, pad    



def get_differenciation_conv(n, hx, hy, hz, gradient = 'first', stencil_type = 'mid', accuracy = '2'):
    implemented_gradient = ['first','second', 'third', 'fourth', 'fifth', 'sixth']
    if gradient not in implemented_gradient:
        raise NotImplementedError
    
    if gradient == 'first':
        G, Gx, Gy, Gz, pad = get_first_grad_stencil(n, hx, hy, hz, 
                                                    stencil_type = stencil_type, 
                                                    accuracy = accuracy)
    elif gradient == 'second':
        G, Gx, Gy, Gz, pad = get_second_grad_stencil(n, hx, hy, hz, 
                                                    stencil_type = stencil_type, 
                                                    accuracy = accuracy)                    
    elif gradient == 'third':
        G, Gx, Gy, Gz, pad = get_third_grad_stencil(n, hx, hy, hz, 
                                                    stencil_type = stencil_type, 
                                                    accuracy = accuracy)    
    elif gradient == 'fourth':
        G, Gx, Gy, Gz, pad = get_fourth_grad_stencil(n, hx, hy, hz, 
                                                    stencil_type = stencil_type, 
                                                    accuracy = accuracy)
                                                    
    elif gradient == 'fifth':
        G, Gx, Gy, Gz, pad = get_fifth_grad_stencil(n, hx, hy, hz, 
                                                    stencil_type = stencil_type, 
                                                    accuracy = accuracy)                                                    
                                                    
    elif gradient == 'sixth':
        G, Gx, Gy, Gz, pad = get_sixth_grad_stencil(n, hx, hy, hz, 
                                                    stencil_type = stencil_type, 
                                                    accuracy = accuracy)

    
    pad_temp = pad[0]
    
    wrapped_n = np.pad(n, pad_temp, mode='wrap')
    
    
    temp_gradient = convolve(wrapped_n,G)
    result = temp_gradient[pad_temp:-pad_temp, pad_temp:-pad_temp, pad_temp:-pad_temp]
        
    return result, pad
                                                    
'''
Integration

Basic philosophy: Initiate the dimensions of the stencils based on the r specified
                  and the grid spacing in each direction
                  
                  Get the coordinate where the sphere center is located
                  
                  for each dV, get the coordinates of the 8 vertices and store
                  
                  for each dV, check how many vertices are within the range of the sphere
                      if none, we assume there's no overlap between the sphere and dV
                      if 8, we assume all of dV is in the sphere
                      if other, subdivided dV into 8 sub-cubes and re-run the process

'''

    
def check_in_range(point_coord, center_coord,  r2):
    # Check if a given point is within the range of the sphere defined by
    # the coordinate of the center and radius r
#    return (np.linalg.norm(np.asarray(center_coord) - np.asarray(point_coord))) <= (r)
    temp1 = (center_coord[0] - point_coord[0]) ** 2.
    temp2 = (center_coord[1] - point_coord[1]) ** 2.
    temp3 = (center_coord[2] - point_coord[2]) ** 2.
    return (temp1 + temp2 + temp3 ) <= (r2) 
    
def check_num_vertices_in_range(li_vertex_coord, center_coord,r2):
    # for a list of points, return the number of the points that's within the 
    # range of the sphere defined by the coordinate of the center and radius r

    # Here we used to check how many vertices of a cube is within the range
    num_vertex_in_range = 0
    for vertex_coord in li_vertex_coord:
        if check_in_range(vertex_coord, center_coord, r2) == True:
            num_vertex_in_range += 1
    return num_vertex_in_range



def get_list_vertices_coord_subdivided_v(x_coord,y_coord,z_coord,dx,dy,dz):
    result = []
    result.append([x_coord, y_coord, z_coord])
    result.append([x_coord+dx, y_coord, z_coord])
    result.append([x_coord, y_coord+dy, z_coord])
    result.append([x_coord, y_coord, z_coord+dz])
    result.append([x_coord+dx, y_coord+dy, z_coord])
    result.append([x_coord+dx, y_coord, z_coord+dz])
    result.append([x_coord, y_coord+dy, z_coord+dz])
    result.append([x_coord+dx, y_coord+dy, z_coord+dz])
    
    return result
    
def get_subdivided_v_vertices(original_vertices_coords):
    result = []
    x_coord, y_coord, z_coord = original_vertices_coords[0]
    dx = (original_vertices_coords[1][0] - original_vertices_coords[0][0]) / 2.
    dy = (original_vertices_coords[2][1] - original_vertices_coords[0][1]) / 2.
    dz = (original_vertices_coords[3][2] - original_vertices_coords[0][2]) / 2.
    
    temp_vertices = get_list_vertices_coord_subdivided_v(x_coord,y_coord,z_coord,dx,dy,dz)
    
    for point in temp_vertices:
        result.append(get_list_vertices_coord_subdivided_v(point[0],point[1],point[2],dx,dy,dz))
    
    return result

def determine_dv_in_sphere_ratio(li_vertex_coord, center_coord,r2, acc_level, accuracy):
    # a recursive function, used to figure out the percentage of the volume that
    # is overlaping with the sphere
    
    # volume of the section, calculated based on the level of sub-division
    V = 1./((8.)**(acc_level-1))
    num_vertex_in_sphere = check_num_vertices_in_range(li_vertex_coord, center_coord,r2)
    if num_vertex_in_sphere == 8:
        return 1.* V
    elif num_vertex_in_sphere == 0:
        return 0.
    elif acc_level > accuracy:
        return float(num_vertex_in_sphere)*V/8.
    else:
        acc_level += 1
        result = 0.
        list_of_subdivided_v_vertices = get_subdivided_v_vertices(li_vertex_coord)
        for subdivided_v_vertices in list_of_subdivided_v_vertices:
            result += determine_dv_in_sphere_ratio(subdivided_v_vertices, center_coord,r2, acc_level, accuracy)
        return result

def get_list_vertices_coord(x_index,y_index,z_index,hx,hy,hz):
    # for each dV, based in the index in the array, 
    # figure out the coordinates of the 8 vertices of the dV    
    
    result = []
    
    result.append([x_index*hx, y_index*hy, z_index*hz])
    result.append([(x_index+1)*hx, y_index*hy, z_index*hz])
    result.append([x_index*hx, (y_index+1)*hy, z_index*hz])
    result.append([x_index*hx, y_index*hy, (z_index+1)*hz])
    result.append([(x_index+1)*hx, (y_index+1)*hy, z_index*hz])
    result.append([(x_index+1)*hx, y_index*hy, (z_index+1)*hz])
    result.append([x_index*hx, (y_index+1)*hy, (z_index+1)*hz])
    result.append([(x_index+1)*hx, (y_index+1)*hy, (z_index+1)*hz])
    
    return result



def get_coordinate_array(dim_x, dim_y, dim_z, hx, hy, hz):
    temp = np.zeros((int(dim_x), int(dim_y), int(dim_z))).tolist()
    for index, x in np.ndenumerate(temp):
        temp[index[0]][index[1]][index[2]] = get_list_vertices_coord(index[0],index[1],index[2],hx,hy,hz)
    result = np.asarray(temp)
    return result

def calc_integration_stencil(hx, hy, hz, r, accuracy):
    # calculate the stencil

    # initialize the stencil with right dimensions
    dim_x = int(2.* math.ceil( r/hx ))
    dim_y = int(2.* math.ceil( r/hy ))
    dim_z = int(2.* math.ceil( r/hz ))
   
    stencil = np.zeros((dim_x, dim_y, dim_z))
    coord_arr = get_coordinate_array(dim_x, dim_y, dim_z, hx, hy, hz)
    
    # caclulate the coordinate of the sphere center
    center_x = hx * float(dim_x)/2.
    center_y = hy * float(dim_y)/2.
    center_z = hz * float(dim_z)/2.
    center_coord = [center_x, center_y, center_z]
    
    
    # for each dV, get the percentage of dV that's overlaping with the sphere
    # and assign the percentage to the stencil
    r2 = r*r
    for index, x in np.ndenumerate(stencil):
        stencil[index[0]][index[1]][index[2]] = determine_dv_in_sphere_ratio(coord_arr[index[0]][index[1]][index[2]], center_coord,r2, 1, accuracy)
    
    # normalize the stencil with the volume of dV
    stencil *= hx*hy*hz
#    print stencil
    
    padx = int(math.ceil(float(dim_x)/2.))
    pady = int(math.ceil(float(dim_y)/2.))
    padz = int(math.ceil(float(dim_z)/2.))
    
    pad = (padx,pady,padz)
    
    return stencil, pad

def from_temp_stencil_to_stencil(dim_x, dim_y, dim_z, temp_stencil):
    stencil = np.zeros((dim_x, dim_y, dim_z))
    center_row_x = (dim_x-1)/2
    center_row_y = (dim_y-1)/2
    center_row_z = (dim_z-1)/2
    
    for index, number in np.ndenumerate(temp_stencil):
        x = index[0]# + center_row_x
        y = index[1]# + center_row_y
        z = index[2]# + center_row_z
        stencil[int(center_row_x + x)][int(center_row_y + y)][int(center_row_z + z)] = number
        stencil[int(center_row_x - x)][int(center_row_y + y)][int(center_row_z + z)] = number
        stencil[int(center_row_x + x)][int(center_row_y - y)][int(center_row_z + z)] = number
        stencil[int(center_row_x + x)][int(center_row_y + y)][int(center_row_z - z)] = number
        stencil[int(center_row_x - x)][int(center_row_y - y)][int(center_row_z + z)] = number
        stencil[int(center_row_x + x)][int(center_row_y - y)][int(center_row_z - z)] = number
        stencil[int(center_row_x - x)][int(center_row_y + y)][int(center_row_z - z)] = number
        stencil[int(center_row_x - x)][int(center_row_y - y)][int(center_row_z - z)] = number
    
    return stencil

def calc_integration_stencil2(hx, hy, hz, r, accuracy):
    # calculate the stencil

    # initialize the stencil with right dimensions
    dim_x = int(2.* math.ceil( r/hx )) + 1
    dim_y = int(2.* math.ceil( r/hy )) + 1
    dim_z = int(2.* math.ceil( r/hz )) + 1
       
#    stencil = np.zeros((dim_x, dim_y, dim_z))
    temp_stencil = np.zeros((int((dim_x + 1 )/2), int((dim_y + 1 )/2), int((dim_z + 1 )/2)))
#    coord_arr = get_coordinate_array(dim_x, dim_y, dim_z, hx, hy, hz)
    
    temp_coord_arr = get_coordinate_array((dim_x + 1 )/2, (dim_y + 1 )/2, (dim_z + 1 )/2, hx, hy, hz)
    
    # caclulate the coordinate of the sphere center

    
    temp_center_x = hx / 2.
    temp_center_y = hy / 2.
    temp_center_z = hz / 2.
    temp_center_coord = [temp_center_x, temp_center_y, temp_center_z]
    print temp_center_coord
    r2 = r*r
    for index, x in np.ndenumerate(temp_stencil):
       temp_stencil[index[0]][index[1]][index[2]] = determine_dv_in_sphere_ratio(temp_coord_arr[index[0]][index[1]][index[2]], temp_center_coord,r2, 1, accuracy)
    
    
    # for each dV, get the percentage of dV that's overlaping with the sphere
    # and assign the percentage to the stencil
   
    # normalize the stencil with the volume of dV
    stencil = from_temp_stencil_to_stencil(dim_x, dim_y, dim_z, temp_stencil)
    stencil *= hx*hy*hz
#    print stencil
    
    padx = int(math.ceil(float(dim_x)/2.))
    pady = int(math.ceil(float(dim_y)/2.))
    padz = int(math.ceil(float(dim_z)/2.))
    
    pad = (padx,pady,padz)

    
    return stencil, pad

def get_auto_accuracy(hx,hy,hz, r):
    h = max([hx,hy,hz])
    temp = 5 - int(math.floor((r/h)/3.))
    if temp < 1:
        return 1
    else:
        return temp
        
def get_integration_stencil(hx, hy, hz, r, accuracy):
    standard_acc = get_auto_accuracy(hx,hy,hz, r)
    if accuracy != standard_acc:
        stencil, pad = calc_integration_stencil2(hx, hy, hz, r, accuracy)
    else:
        if max(hx, hy, hz) - min(hx, hy, hz) < 0.001:
            try:
                stencil, pad = read_integration_stencil(hx, hy, hz, r)
            
            except:
                stencil, pad = calc_integration_stencil2(hx, hy, hz, r, accuracy)
        else:
            stencil, pad = calc_integration_stencil2(hx, hy, hz, r, accuracy)

    return stencil, pad

       
        

def get_integration_conv(n, hx, hy, hz, r, accuracy = 4):
    # get the stencil and do the convolution
    
    stencil, pad = get_integration_stencil(hx, hy, hz, r, accuracy)
    return convolve(n,stencil, mode = 'wrap'), pad


def get_integration_fftconv(n, hx, hy, hz, r, accuracy = 4):
    # get the stencil and do the convolution

    stencil, pad = get_integration_stencil(hx, hy, hz, r, accuracy)
    pad_temp = int(math.ceil(r*2. / min([hx,hy,hz])))
    wrapped_n = np.pad(n, pad_temp, mode='wrap')
    temp_result = fftconvolve(wrapped_n,stencil, mode = 'same')
    return temp_result[pad_temp:-pad_temp, pad_temp:-pad_temp, pad_temp:-pad_temp], pad

def get_integral_fftconv_with_known_stencil(n, hx, hy, hz, r, stencil, pad):
    # get the stencil and do the convolution

#    stencil, pad = get_integration_stencil(hx, hy, hz, r, accuracy)
    pad_temp = int(math.ceil(r*2. / min([hx,hy,hz])))
    wrapped_n = np.pad(n, pad_temp, mode='wrap')
    temp_result = fftconvolve(wrapped_n,stencil, mode = 'same')
    return temp_result[pad_temp:-pad_temp, pad_temp:-pad_temp, pad_temp:-pad_temp], pad


def get_fftconv_with_known_stencil_no_wrap(n, hx, hy, hz, r, stencil, pad):
    temp_result = fftconvolve(n,stencil, mode = 'same')
    return temp_result, pad


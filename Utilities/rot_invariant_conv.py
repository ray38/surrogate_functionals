# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 23:07:55 2017

@author: Ray
"""

import numpy as np
from scipy.ndimage.filters import convolve
from math import pi
import math

import pickle

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

    
def check_in_range(point_coord, center_coord,  r):
    # Check if a given point is within the range of the sphere defined by
    # the coordinate of the center and radius r
    return (np.linalg.norm(np.asarray(center_coord) - np.asarray(point_coord))) <= (r)
    
def check_num_vertices_in_range(li_vertex_coord, center_coord,r):
    # for a list of points, return the number of the points that's within the 
    # range of the sphere defined by the coordinate of the center and radius r

    # Here we used to check how many vertices of a cube is within the range
    num_vertex_in_range = 0
    for vertex_coord in li_vertex_coord:
        if check_in_range(vertex_coord, center_coord, r) == True:
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

def determine_dv_in_sphere_ratio(li_vertex_coord, center_coord,r, acc_level, accuracy):
    # a recursive function, used to figure out the percentage of the volume that
    # is overlaping with the sphere
    
    # volume of the section, calculated based on the level of sub-division
    V = 1./((8.)**(acc_level-1))
    num_vertex_in_sphere = check_num_vertices_in_range(li_vertex_coord, center_coord,r)
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
            result += determine_dv_in_sphere_ratio(subdivided_v_vertices, center_coord,r, acc_level, accuracy)
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
    temp = np.zeros((dim_x, dim_y, dim_z)).tolist()
    for index, x in np.ndenumerate(temp):
        temp[index[0]][index[1]][index[2]] = get_list_vertices_coord(index[0],index[1],index[2],hx,hy,hz)
    result = np.asarray(temp)
    return result

def get_integration_stencil(hx, hy, hz, r, accuracy):
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
    for index, x in np.ndenumerate(stencil):
        stencil[index[0]][index[1]][index[2]] = determine_dv_in_sphere_ratio(coord_arr[index[0]][index[1]][index[2]], center_coord,r, 1, accuracy)
    
    # normalize the stencil with the volume of dV
    stencil *= hx*hy*hz
#    print stencil
    
    padx = int(math.ceil(float(dim_x)*2.))
    pady = int(math.ceil(float(dim_y)*2.))
    padz = int(math.ceil(float(dim_z)*2.))
    
    pad = (padx,pady,padz)
    
    return stencil, pad


def get_integration_conv(n, hx, hy, hz, r, accuracy = 4):
    # get the stencil and do the convolution
    stencil, pad = get_integration_stencil(hx, hy, hz, r, accuracy)
    return convolve(n,stencil), pad














def determine_dv_in_shell_ratio(li_vertex_coord, center_coord,r_large, r_small, acc_level, accuracy):
    temp1 = determine_dv_in_sphere_ratio(li_vertex_coord, center_coord,r_large,acc_level, accuracy)
    temp2 = determine_dv_in_sphere_ratio(li_vertex_coord, center_coord,r_small,acc_level, accuracy)
    return temp1 - temp2
#    # a recursive function, used to figure out the percentage of the volume that
#    # is overlaping with the sphere
#    
#    # volume of the section, calculated based on the level of sub-division
#    V = 1./((8.)**(acc_level-1))
#    num_vertex_in_sphere = check_num_vertices_in_range(li_vertex_coord, center_coord,r)
#    if num_vertex_in_sphere == 8:
#        return 1.* V
#    elif num_vertex_in_sphere == 0:
#        return 0.
#    elif acc_level > accuracy:
#        return float(num_vertex_in_sphere)*V/8.
#    else:
#        acc_level += 1
#        result = 0.
#        list_of_subdivided_v_vertices = get_subdivided_v_vertices(li_vertex_coord)
#        for subdivided_v_vertices in list_of_subdivided_v_vertices:
#            result += determine_dv_in_sphere_ratio(subdivided_v_vertices, center_coord,r, acc_level, accuracy)
#        return result



def get_shell_area(r1, r2):
    return (4./3.) *pi * (r1*r1*r1 - r2*r2*r2)


def get_derivative_stencil(fd_coefficient_raw, stencil_accuracy, h):
#    fd_coefficient = [2., -27.,  270., -490., 270., -27., 2.]
    dim = len(fd_coefficient_raw)
    fd_coefficient = (np.asarray(fd_coefficient_raw)/np.sum(np.abs(np.asarray(fd_coefficient_raw)))).tolist()

    r = float(len(fd_coefficient))*1. / 2.
    hx = h
    hy = h
    hz = h   
    dim_x = int(2.* math.ceil( r/hx ))
    dim_y = int(2.* math.ceil( r/hy ))
    dim_z = int(2.* math.ceil( r/hz ))
  
    
    if dim_x % 2 ==0:
        dim_x += 1
    if dim_y % 2 ==0:
        dim_y += 1
    if dim_z % 2 ==0:
        dim_z += 1
        

    stencil = np.zeros((dim_x, dim_y, dim_z))
    
    coord_arr = get_coordinate_array(dim_x, dim_y, dim_z, hx, hy, hz)


    
    center_x = hx * float(dim_x)/2.
    center_y = hy * float(dim_y)/2.
    center_z = hz * float(dim_z)/2.
    center_coord = [center_x, center_y, center_z]
    print 'dimension'
    print dim_x, dim_y, dim_z  
    print center_coord
#    temp_i = 
    for i in range((len(fd_coefficient)+1) / 2):
        temp_stencil = np.zeros((dim_x, dim_y, dim_z))
        temp_r_large = ((float(dim) + 1.)/2.) - float(i)
        temp_r_small = temp_r_large - 1.
        
        temp_shell_area = get_shell_area(temp_r_large, temp_r_small)
        
        print ''
        print temp_r_large
        print temp_r_small
        print fd_coefficient[i]
        for index, x in np.ndenumerate(temp_stencil):
            stencil[index[0]][index[1]][index[2]] += fd_coefficient[i] * determine_dv_in_shell_ratio(coord_arr[index[0]][index[1]][index[2]], center_coord,temp_r_large,temp_r_small, 1, stencil_accuracy) / temp_shell_area
        
#        stencil += temp_stencil
    return stencil

if __name__ == "__main__":
    
    h = 1.0
    stencil_acc = 8
    h_name = '1_0'
#    coeff_list = [['sec_2.p',[1.,-2., 1.]],\
#                  ['sec_4.p',[-1., 16., -30., 16., -1.]],\
#                  ['sec_6.p',[2., -27.,  270., -490., 270., -27., 2.]],\
#                  ['sec_8.p',[-9., 128., -1008., 8064., -14350., 8064., -1008., 128., -9.]],\
#                  ['fourth_2.p',[1., -4., 6., -4., 1.]],\
#                  ['fourth_4.p',[-1., 12., -78., 112., -78., 12., -1.]],\
#                  ['fourth_6.p',[7., -76., 676., -1952., 2730., -1952., 676., -76., 7.]],\
#                  ['six_6.p',[1., -6., 15., -20., 15., -6., 1.]]]
    
#    for entry in coeff_list:
#        filename = entry[0]
#        fd_coeff = entry[1]
#        print filename
#        print fd_coeff
#        temp_stencil = get_derivative_stencil(fd_coeff, 3)
#        with open(filename, 'wb') as handle:
#            pickle.dump(temp_stencil, handle, protocol=pickle.HIGHEST_PROTOCOL)
#        print 'done'
        

    entry = ['sec_4_' + h_name + '.p',[-1., 16., -30., 16., -1.] ]
#    for entry in coeff_list:
    filename = entry[0]
    fd_coeff = entry[1]
    print filename
    print fd_coeff
    temp_stencil = get_derivative_stencil(fd_coeff, stencil_acc, h)
    with open(filename, 'wb') as handle:
        pickle.dump(temp_stencil, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print 'done'

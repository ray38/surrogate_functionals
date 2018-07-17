# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 19:34:19 2017

@author: ray
"""


import os
import h5py
import psi4
import sys
import json
import collections
import numpy as np
import math
import multiprocessing
import json
import itertools
import pandas as pd
import seaborn as sns
import copy

import matplotlib
matplotlib.use('Agg') 
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt

from convolutions import get_differenciation_conv, get_integration_stencil,get_auto_accuracy,get_fftconv_with_known_stencil_no_wrap,get_asym_integration_stencil,get_asym_integration_fftconv,get_asym_integral_fftconv_with_known_stencil
from convolutions import get_first_grad_stencil, get_second_grad_stencil, get_third_grad_stencil, get_harmonic_fftconv, calc_harmonic_stencil


def generate_3d2(x1, x2, x3):

    R = np.matrix([[np.cos(2 * np.pi * x1), np.sin(2 * np.pi * x1), 0],
                   [-np.sin(2 * np.pi * x1), np.cos(2 * np.pi * x1), 0],
                   [0, 0, 1]])
    v = np.matrix([[np.cos(2 * np.pi * x2) * np.sqrt(x3)],
                   [np.sin(2 * np.pi * x2) * np.sqrt(x3)],
                   [np.sqrt(1 - x3)]])
    H = np.eye(3) - 2 * v * v.T
    M = -H * R
    return M

def transform_coord_mat(coordinates,theta1,theta2,theta3,x0,y0,z0):

    translated_coordinates = np.asarray([(coordinates[0] - x0).tolist(), (coordinates[1] - y0).tolist(), (coordinates[2] - z0).tolist()])
    rot_mat = generate_3d2(theta1,theta2,theta3) 
    #temp_shape = x.shape
    #temp_coord = np.stack([x.copy().flatten(),y.copy().flatten(),z.copy().flatten()], axis=0)
    after_rotate = np.asarray(np.dot(rot_mat,translated_coordinates))
    #print np.transpose(after_rotate)

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(after_rotate[0], after_rotate[1], after_rotate[2], c='k')
    #plt.show()


    return np.transpose(after_rotate)


def log(log_filename, text):
    with open(log_filename, "a") as myfile:
        myfile.write(text)
    return

# Define helper functions to re-map into 3D array
def mapping_array(x,y,z,out_shape,x_start,x_end,y_start,y_end,z_start,z_end):
    map_array = np.zeros(out_shape,dtype='int')
    npoints_x, npoints_y, npoints_z = out_shape
 
    def get_index(xi,x_start,x_end,npoints):
        xL = list(np.linspace(x_start, x_end, npoints))
        idx = xL.index(xi)
        return idx
    for i,xi,yi,zi in zip(range(len(x)),x,y,z):
        xid = get_index(xi,x_start,x_end,npoints_x)
        yid = get_index(yi,y_start,y_end,npoints_y)
        zid = get_index(zi,z_start,z_end,npoints_z)
        map_array[xid,yid,zid] = int(i)
    return map_array.ravel()

def remap(in_vector, map_array, out_shape):
    in_vector = in_vector[map_array].reshape(out_shape)
    return in_vector
    

def process_one_section(x,y,z,w,x_start,x_end,y_start,y_end,z_start,z_end,out_shape,scf_wfn,scf_e):
        
    # Get DFT density
    C = np.array(scf_wfn.Ca_subset("AO", "OCC"))
    D = np.dot(C, C.T)
    DFT_Density = psi4.core.Matrix.from_array(D)
    
    # Set up Vxc functional
    Vpot = scf_wfn.V_potential()
    superfunc = Vpot.functional()
    
    
    # In terms of Psi4 vector objects
    xvec = psi4.core.Vector.from_array(x)
    yvec = psi4.core.Vector.from_array(y)
    zvec = psi4.core.Vector.from_array(z)
    wvec = psi4.core.Vector.from_array(w)

    # Compute the spatial extent computed
    # Used for sieving
    extents = psi4.core.BasisExtents(scf_wfn.basisset(), 1.e-32)
    # Build a "block-O-points"
    grid = psi4.core.BlockOPoints(xvec, yvec, zvec, wvec, extents)

    # Need a proper points funciton
    points_func = psi4.core.RKSFunctions(scf_wfn.basisset(), x.shape[0], scf_wfn.basisset().nbf())
    points_func.set_deriv(2)
    points_func.set_ansatz(2)
    points_func.set_pointers(DFT_Density)

    # Make sure the SuperFunctional has room
    superfunc.set_max_points(x.shape[0])
    superfunc.allocate()

    # Drop the outputs here
    ret = collections.defaultdict(list)
    
    # Grab a grid "block"
    w = np.array(grid.w())
    x = np.array(grid.x())
    y = np.array(grid.y())
    z = np.array(grid.z())

    # Global to local map
    gmap = np.array(grid.functions_local_to_global())

    # What size do I need to slice out of my buffers?
    npoints = w.shape[0]
    nfunc = gmap.shape[0]

    # Copmute the given grid
    points_func.compute_points(grid)

    # Grab quantities on a grid
    rho = np.array(points_func.point_values()["RHO_A"])[:npoints]
    rho_x = np.array(points_func.point_values()["RHO_AX"])[:npoints]
    rho_y = np.array(points_func.point_values()["RHO_AY"])[:npoints]
    rho_z = np.array(points_func.point_values()["RHO_AZ"])[:npoints]
    gamma = np.array(points_func.point_values()["GAMMA_AA"])[:npoints]
    tau = np.array(points_func.point_values()["TAU_A"])[:npoints]
    gradient = rho_x + rho_y + rho_z

    # Compute our functional
    dft_results = superfunc.compute_functional(points_func.point_values(), -1)

    # Append outputs
    ret["w"].append(w)
    ret["x"].append(x)
    ret["y"].append(y)
    ret["z"].append(z)

    ret["rho"].append(rho)
    ret["gradient"].append(gradient)
    ret["gamma"].append(gamma)
    ret["tau"].append(tau)

    ret["epsilon_xc"].append(np.array(dft_results["V"])[:npoints])
    ret["V_xc"].append(np.array(dft_results["V_RHO_A"])[:npoints])
    
    # Reformat outputs into 3D array format
    map_array = mapping_array(x,y,z,out_shape,x_start,x_end,y_start,y_end,z_start,z_end)
    output = {k : remap(np.hstack(v),map_array,out_shape) for k, v in ret.items()}
    


    return output
    
def process(X0,Y0,Z0,x_inc,y_inc,z_inc,hx,hy,hz,i,j,k ,dv,scf_wfn,scf_e, convolution_property_stencils):

    x_start = X0 + float(i) * x_inc
    y_start = Y0 + float(j) * y_inc
    z_start = Z0 + float(k) * z_inc
    
    x_end = x_start + x_inc - hx
    y_end = y_start + y_inc - hy
    z_end = z_start + z_inc - hz
    
    print "\n x: {}:{} \t {}:{} \t {}:{}".format(x_start,x_end,y_start,y_end,z_start,z_end)
    
    xyz = []
    for hi, start,end in zip((hx, hy, hz),(x_start, y_start ,z_start),(x_end, y_end ,z_end)):
        n_i = ((end-start)/hi)+1.
        xi = np.linspace(start, end, int(round(n_i,0)))
        xyz.append(xi)

    out_shape = [len(xi) for xi in xyz]
    x, y, z = np.meshgrid(*xyz)
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()
    w = np.ones_like(z)*dv
    
    temp_filename =  '{}_{}_{}_{}_{}.hdf5'.format(molecule_name,xc,i,j,k)
    temp_out = process_one_section(x,y,z,w,x_start,x_end,y_start,y_end,z_start,z_end,out_shape,scf_wfn,scf_e)



    n = temp_out["rho"]



    temp_gamma = temp_out['gamma']
    temp_gradient = temp_out['gradient']
    temp_exc = temp_out["epsilon_xc"]
    temp_tau = temp_out["tau"]

    shape = temp_gamma.shape

    x = temp_out["x"][(shape[0])/2][(shape[1])/2][(shape[2])/2]
    y = temp_out["y"][(shape[0])/2][(shape[1])/2][(shape[2])/2]
    z = temp_out["z"][(shape[0])/2][(shape[1])/2][(shape[2])/2]

    print "coordinates: {}\t {}\t {}".format(x,y,z)


    result = [temp_gamma[(shape[0])/2][(shape[1])/2][(shape[2])/2], temp_gradient[(shape[0])/2][(shape[1])/2][(shape[2])/2], temp_exc[(shape[0])/2][(shape[1])/2][(shape[2])/2], temp_tau[(shape[0])/2][(shape[1])/2][(shape[2])/2]]

    for stencil in convolution_property_stencils:
        temp_convolution_result,_ = get_fftconv_with_known_stencil_no_wrap(n,hx,hy,hz,1,stencil,0)
        result.append(temp_convolution_result[(shape[0])/2][(shape[1])/2][(shape[2])/2])

    #result = np.asarray([temp_gamma[(shape[0])/2][(shape[1])/2][(shape[2])/2], temp_gradient[(shape[0])/2][(shape[1])/2][(shape[2])/2], temp_exc[(shape[0])/2][(shape[1])/2][(shape[2])/2], temp_tau[(shape[0])/2][(shape[1])/2][(shape[2])/2]])

    return np.asarray(result)

    
def process_system(molecule, molecule_name, xc, h, cell, num_blocks, convolution_property_stencils,psi4_options=None):
    cwd = os.getcwd()
    
    if psi4_options == None:
        psi4_options = {"BASIS": "cc-pvdz",
                    "D_CONVERGENCE":1e-12,
                    "E_CONVERGENCE":1e-12,
                  'DFT_BLOCK_MAX_POINTS': 500000,
                  'DFT_BLOCK_MIN_POINTS': 100000,
                  'MAXITER': 500,
#                  'DFT_SPHERICAL_POINTS': 302,
#                  'DFT_RADIAL_POINTS':    75,
                  "SAVE_JK": True, }
    psi4.set_options(psi4_options)
    
    if isinstance(h,float):
        hx = hy = hz = h
    elif len(h) == 3:
        hx, hy, hz = h
    else:
        raise Exception('Invalid grid spacing')

    dv = hx*hy*hz #volume element size (used for integration)
    
    if isinstance(cell,float) or isinstance(cell,int): 
        Lx = Ly = Lz = float(cell)
    elif len(cell) == 3:
        Lx, Ly, Lz = cell
    else:
        raise Exception('Invalid cell')
    
    if  isinstance(num_blocks,int): 
        Nx = Ny = Nz = int(num_blocks)
    elif len(num_blocks) == 3:
        Nx, Ny, Nz = num_blocks
    else:
        raise Exception('Invalid block dividing')
    
    scf_e, scf_wfn = psi4.energy(xc, molecule=molecule, return_wfn=True)

    
    X0 = -Lx/2.
    Y0 = -Ly/2.
    Z0 = -Lz/2.

    x_inc = Lx/Nx
    y_inc = Ly/Ny
    z_inc = Lz/Nz


    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                result = process(X0,Y0,Z0,x_inc,y_inc,z_inc,hx,hy,hz,i,j,k ,dv,scf_wfn,scf_e, convolution_property_stencils)
                print result

    
    #os.chdir(cwd) 
    return result








def read_json_data(data):
    result = ''
    for i in range(len(data['atoms'])):
        temp = '{}\t{}\t{}\t{}\n'.format(data['atoms'][i], data['coordinates'][i][0],data['coordinates'][i][1],data['coordinates'][i][2])
        result += temp
    result += '\t symmetry {}'.format(data['symmetry'])
    return psi4.geometry(result)



def plot_result(data):
    plt.figure()
        
    sns.set(style="whitegrid", palette="pastel", color_codes=True)

    #ax = sns.violinplot(x = "Molecule",y="Value",row="Type", col="Property",data=data)
    
    ax = sns.factorplot(x = "Molecule",y="Value",row="Type", col="Property",data=data, kind="violin", split=True,sharey = False, size = 6, aspect = 1.5)

    plt.tight_layout()
    plt.savefig("Molecule_rotational_invariance_test1.png")



    return

def plot_result2(data):
    plt.figure()
        
    sns.set(style="whitegrid", palette="pastel", color_codes=True)

    #ax = sns.violinplot(x = "Molecule",y="Value",row="Type", col="Property",data=data)
    
    ax = sns.factorplot(x = "ID",y="Value",col="Property",hue="Label",data=data, kind="point", split=True,sharey = False, size = 6, aspect = 1.5)

    plt.tight_layout()
    plt.savefig("Molecule_rotational_invariance_test2.png")



    return


if __name__ == "__main__":


    database_filename = sys.argv[1]
    molecule_name = sys.argv[2]
    num_rot = int(sys.argv[3])
    num_grid = int(sys.argv[4])
    #h = float(sys.argv[3])
    #L = float(sys.argv[4])
    #N = int(sys.argv[5])

    h = 0.02
    L = 0.28
    N = 1
    xc = 'B3LYP'

    convolution_properties = ["FD gradient mid", "FD gradient times2", "FD 2nd gradient", "Harmonic 1 0.04", "Harmonic 1 0.1", "Harmonic 1 0.16"]
    convolution_property_stencils = []

    stencil, Gx, Gy, Gz, pad = get_first_grad_stencil(h, h, h,stencil_type = "mid",accuracy = '2')
    convolution_property_stencils.append(stencil)
    stencil, Gx, Gy, Gz, pad = get_first_grad_stencil(h, h, h,stencil_type = "times2",accuracy = '2')
    convolution_property_stencils.append(stencil)
    stencil, Gx, Gy, Gz, pad = get_second_grad_stencil(h, h, h,stencil_type = "times2",accuracy = '2')
    convolution_property_stencils.append(stencil)

    stencil_Re_1, stencil_Im_1, pad = calc_harmonic_stencil(h, h, h, 0.04, 1, -1, accuracy = 6)
    stencil_Re_2, stencil_Im_2, pad = calc_harmonic_stencil(h, h, h, 0.04, 1, 0, accuracy = 6)
    stencil_Re_3, stencil_Im_3, pad = calc_harmonic_stencil(h, h, h, 0.04, 1, 1, accuracy = 6)

    stencil = stencil_Re_1 + stencil_Re_2 + stencil_Re_3
    convolution_property_stencils.append(stencil)

    stencil_Re_1, stencil_Im_1, pad = calc_harmonic_stencil(h, h, h, 0.1, 1, -1, accuracy = 6)
    stencil_Re_2, stencil_Im_2, pad = calc_harmonic_stencil(h, h, h, 0.1, 1, 0, accuracy = 6)
    stencil_Re_3, stencil_Im_3, pad = calc_harmonic_stencil(h, h, h, 0.1, 1, 1, accuracy = 6)

    stencil = stencil_Re_1 + stencil_Re_2 + stencil_Re_3
    convolution_property_stencils.append(stencil)

    stencil_Re_1, stencil_Im_1, pad = calc_harmonic_stencil(h, h, h, 0.16, 1, -1, accuracy = 6)
    stencil_Re_2, stencil_Im_2, pad = calc_harmonic_stencil(h, h, h, 0.16, 1, 0, accuracy = 6)
    stencil_Re_3, stencil_Im_3, pad = calc_harmonic_stencil(h, h, h, 0.16, 1, 1, accuracy = 6)

    stencil = stencil_Re_1 + stencil_Re_2 + stencil_Re_3
    convolution_property_stencils.append(stencil)



    molecule_name_list = []

    value_truth_list = []
    value_truth_label_list = []
    value_truth_property_label_list = []
    molecule_name_list2 = []
    counter_list2 = []
    origin_counter_list2 = []

    value_list = []


    type_list = [] # absolute value, absolute error, percent error ## this is the row

    property_list = [] # Gamma, Gradient, exc, tau, descriptors ## this is the col

    counter_list = []
    origin_counter_list = []


    theta1_list = []
    theta2_list = []
    theta3_list = []

    x0_list = []
    y0_list = []
    z0_list = []

    temp_x0_list = np.linspace(-0.3, 0.3, num_grid)
    temp_y0_list = np.linspace(-0.3, 0.3, num_grid) 
    temp_z0_list = np.linspace(-0.3, 0.3, num_grid) 
    origin_list = list(itertools.product(temp_x0_list,temp_y0_list,temp_z0_list))

    temp_theta1_list = np.linspace(0.0, 1.0, num_rot)
    temp_theta2_list = np.linspace(0.0, 1.0, num_rot) 
    temp_theta3_list = np.linspace(0.0, 1.0, num_rot) 
    paramlist = list(itertools.product(temp_theta1_list,temp_theta2_list,temp_theta3_list))
    
    
    try:
        data = json.load(open(database_filename,'rb'))
    except:
        with open(database_filename, encoding='utf-8') as f:
            data=json.load(f)

            
    original_molecule = data[molecule_name]
    original_coordinates = np.asarray(original_molecule["coordinates"])
        
    
    
    counter = 0
    origin_counter = 0

    for x0, y0, z0 in origin_list:
        origin_counter += 1
        temp_coordinate = transform_coord_mat(np.transpose(copy.deepcopy(original_coordinates)),0., 0., 0., x0,y0,z0)

        temp_molecule = {}
        temp_molecule["atoms"] = original_molecule["atoms"]
        temp_molecule["symmetry"] = original_molecule["symmetry"]
        temp_molecule["coordinates"] = temp_coordinate

        temp_molecule_setup = read_json_data(temp_molecule)
        temp_truth = process_system(temp_molecule_setup,molecule_name,xc,h,L,N, convolution_property_stencils)


        for theta1, theta2, theta3 in paramlist:
            counter +=1

            temp_coordinate = transform_coord_mat(np.transpose(copy.deepcopy(original_coordinates)),theta1,theta2,theta3, x0,y0,z0)

            temp_molecule = {}
            temp_molecule["atoms"] = original_molecule["atoms"]
            temp_molecule["symmetry"] = original_molecule["symmetry"]
            temp_molecule["coordinates"] = temp_coordinate

            temp_molecule_setup = read_json_data(temp_molecule)
            temp_result = process_system(temp_molecule_setup,molecule_name,xc,h,L,N, convolution_property_stencils)

            temp_error = temp_result - temp_truth
            #log("rotation_test.log",str(temp_error))

            value_list.append(temp_result[0])
            type_list.append("Value")
            property_list.append("Gamma")


            value_list.append(temp_result[1])
            type_list.append("Value")
            property_list.append("Gradient")


            value_list.append(temp_result[2])
            type_list.append("Value")
            property_list.append("exc")


            value_list.append(temp_result[3])
            type_list.append("Value")
            property_list.append("tau")

            for i in range(len(convolution_properties)):
                value_list.append(temp_result[i+4])
                truth_list.append(temp_truth[i+4])
                type_list.append("Value")
                property_list.append(convolution_properties[i])


            value_list.append(temp_error[0])
            type_list.append("Error")
            property_list.append("Gamma")


            value_list.append(temp_error[1])
            type_list.append("Error")
            property_list.append("Gradient")


            value_list.append(temp_error[2])
            type_list.append("Error")
            property_list.append("exc")


            value_list.append(temp_error[3])
            type_list.append("Error")
            property_list.append("tau")

            for i in range(len(convolution_properties)):
                value_list.append(temp_error[i+4])
                type_list.append("Error")
                property_list.append(convolution_properties[i])


            value_list.append((temp_error[0]/temp_truth[0])*100.0)
            type_list.append("Relative Error")
            property_list.append("Gamma")


            value_list.append((temp_error[1]/temp_truth[1])*100.0)
            type_list.append("Relative Error")
            property_list.append("Gradient")


            value_list.append((temp_error[2]/temp_truth[2])*100.0)
            type_list.append("Relative Error")
            property_list.append("exc")


            value_list.append((temp_error[3]/temp_truth[3])*100.0)
            type_list.append("Relative Error")
            property_list.append("tau")

            for i in range(len(convolution_properties)):
                value_list.append((temp_error[i+4]/temp_truth[i+4])*100.0)
                type_list.append("Relative Error")
                property_list.append(convolution_properties[i])



            for i in range(12 + (len(convolution_properties)*3)):

                theta1_list.append(theta1)
                theta2_list.append(theta2)
                theta3_list.append(theta3)

                x0_list.append(x0)
                y0_list.append(y0)
                z0_list.append(z0)

                molecule_name_list.append(molecule_name)

                counter_list.append(counter)

                origin_counter_list.append(origin_counter)



            value_truth_list.append(temp_result[0])
            value_truth_list.append(temp_truth[0])
            value_truth_property_label_list.append("Gamma")
            value_truth_property_label_list.append("Gamma")

            value_truth_list.append(temp_result[1])
            value_truth_list.append(temp_truth[1])
            value_truth_property_label_list.append("Gradient")
            value_truth_property_label_list.append("Gradient")

            value_truth_list.append(temp_result[2])
            value_truth_list.append(temp_truth[2])
            value_truth_property_label_list.append("exc")
            value_truth_property_label_list.append("exc")

            value_truth_list.append(temp_result[3])
            value_truth_list.append(temp_truth[3])
            value_truth_property_label_list.append("tau")
            value_truth_property_label_list.append("tau")

            for i in range(len(convolution_properties)):
                value_truth_list.append(temp_result[i+4])
                value_truth_list.append(temp_truth[i+4])
                value_truth_property_label_list.append(convolution_properties[i])
                value_truth_property_label_list.append(convolution_properties[i])

            for i in range(4 + (len(convolution_properties))):
                value_truth_label_list.append("value")
                value_truth_label_list.append("truth")
                molecule_name_list2.append(molecule_name)
                counter_list2.append(counter)
                origin_counter_list2.append(origin_counter)
                molecule_name_list2.append(molecule_name)
                counter_list2.append(counter)
                origin_counter_list2.append(origin_counter)

            

            
            if (counter % 10) == 0:
                d1 = {   "Molecule": molecule_name_list,
                        "Value": value_list,
                        "Type":type_list,
                        "Property":property_list,
                        "theta1":theta1_list, 
                        "theta2":theta2_list, 
                        "theta3": theta3_list,
                        "x0":x0_list,
                        "y0":y0_list,
                        "z0":z0_list,
                        "ID": counter_list,
                        "origin ID": origin_counter_list}
                data1 = pd.DataFrame(data=d1)
                data1.to_pickle("{}_rotation_test_dataframe.p".format(molecule_name))
                plot_result(data1)

                d2 = {  "Molecule": molecule_name_list2,
                        "Value": value_truth_list,
                        "Label":value_truth_label_list,
                        "Property":value_truth_property_label_list,
                        "ID": counter_list2,
                        "origin ID": origin_counter_list2}
                data2 = pd.DataFrame(data=d2)
                data2.to_pickle("{}_rotation_test_dataframe2.p".format(molecule_name))
                plot_result(data2)




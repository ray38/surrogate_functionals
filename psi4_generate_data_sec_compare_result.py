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

    # Global to local map
    gmap = np.array(grid.functions_local_to_global())

    # What size do I need to slice out of my buffers?
    npoints = w.shape[0]
    nfunc = gmap.shape[0]

    # Copmute the given grid
    points_func.compute_points(grid)

    # Grab quantities on a grid


    # Compute our functional
    dft_results = superfunc.compute_functional(points_func.point_values(), -1)

    # Append outputs


    ret["epsilon_xc"].append(np.array(dft_results["V"])[:npoints])
    
    # Reformat outputs into 3D array format
    map_array = mapping_array(x,y,z,out_shape,x_start,x_end,y_start,y_end,z_start,z_end)
    output = {k : remap(np.hstack(v),map_array,out_shape) for k, v in ret.items()}
    


    return output
    
def process(X0,Y0,Z0,x_inc,y_inc,z_inc,hx,hy,hz,i,j,k ,dv,scf_wfn,scf_e):

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

    temp_out = process_one_section(x,y,z,w,x_start,x_end,y_start,y_end,z_start,z_end,out_shape,scf_wfn,scf_e)
    


    return np.sum(temp_out["epsilon_xc"])
    
def process_system(molecule, molecule_name, xc, h, cell, num_blocks, psi4_options=None):
    #cwd = os.getcwd()
    #dir_name = "{}_{}_{}_{}_{}".format(molecule_name,xc,str(cell).replace('.','-'),str(h).replace('.','-'),num_blocks)
    
    #if os.path.isdir(dir_name) == False:
    #    os.makedirs(cwd + '/' + dir_name)
    
    #os.chdir(cwd + '/' + dir_name)
    
    if psi4_options == None:
        psi4_options = {"BASIS": "aug-cc-pvtz",
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

    XC = 0.0


    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                temp_xc = process(X0,Y0,Z0,x_inc,y_inc,z_inc,hx,hy,hz,i,j,k ,dv,scf_wfn,scf_e)
                XC += temp_xc

    
    #os.chdir(cwd) 

    print "result XC: {}".format(XC)
    return XC








def read_json_data(data):
    result = ''
    for i in range(len(data['atoms'])):
        temp = '{}\t{}\t{}\t{}\n'.format(data['atoms'][i], data['coordinates'][i][0],data['coordinates'][i][1],data['coordinates'][i][2])
        result += temp
    result += '\t symmetry {}'.format(data['symmetry'])
    return psi4.geometry(result)




if __name__ == "__main__":


    database_filename = sys.argv[1]
    h = 0.02
    L = 10.0
    N = 5
    

    molecule_names = ["C2H2","C2H4","C2H6","CH3OH","CH4","CO","CO2","H2","H2O","HCN","HNC","N2","N2O","NH3","O3","CH3CN","CH3CHO","CH3NO2","glycine","H2CCO","H2CO","H2O2","HCOOH","N2H4","NCCN"]
    
    try:
        data = json.load(open(database_filename,'rb'))
    except:
        with open(database_filename, encoding='utf-8') as f:
            data=json.load(f)
    
    molecules = {}
    for molecule in molecule_names:
        if molecule in data:
            molecules[molecule] = read_json_data(data[molecule])
            
        
    xc_funcs = ['SVWN','PBE']
    all_data = {}
    
    def log(log_filename, text):
        with open(log_filename, "a") as myfile:
            myfile.write(text)
        return
        
    failed_filename = "failed_molecule.log"
    succ_filename = "successful_molecule.log"
    cwd = os.getcwd()

    for xc in xc_funcs:
        for mol in molecules:
            result_XC = process_system(molecules[mol],mol,xc,h,L,N)


        
        


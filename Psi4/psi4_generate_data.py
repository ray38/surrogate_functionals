# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:15:32 2017

@author: ray
"""

import os
import h5py
import psi4
import sys
import json
import collections
import numpy as np


def get_DFT_outputs(molecule, molecule_name, xc, h, cell, psi4_options=None):
    if psi4_options == None:
        psi4_options = {"BASIS": "sto-3g",
                  'DFT_BLOCK_MAX_POINTS': 500000,
                  'DFT_BLOCK_MIN_POINTS': 100000,
                  'DFT_SPHERICAL_POINTS': 50,
                  'DFT_RADIAL_POINTS':    12,
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
    
    scf_e, scf_wfn = psi4.energy(xc, molecule=molecule, return_wfn=True)
    
    # Get DFT density
    C = np.array(scf_wfn.Ca_subset("AO", "OCC"))
    D = np.dot(C, C.T)
    DFT_Density = psi4.core.Matrix.from_array(D)
    
    # Set up Vxc functional
    Vpot = scf_wfn.V_potential()
    superfunc = Vpot.functional()
    
    # Define grid
    xyz = []
    for hi, Li in zip((hx, hy, hz),(Lx, Ly, Lz)):
        n_i = int(Li/hi)
        Li2 = Li/2.
        xi = np.linspace(-Li2, Li2, n_i)
        xyz.append(xi)
    out_shape = [len(xi) for xi in xyz]
    x, y, z = np.meshgrid(*xyz)
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()
    w = np.ones_like(z)*dv #weights for integration

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
    
    # Define helper functions to re-map into 3D array
    def mapping_array(x,y,z,out_shape,Lx,Ly,Lz):
        map_array = np.zeros(out_shape,dtype='int')
        npoints_x, npoints_y, npoints_z = out_shape
 
        def get_index(xi,Lx,npoints):
            xL = list(np.linspace(-Lx/2.,Lx/2., npoints))
            idx = xL.index(xi)
            return idx
        for i,xi,yi,zi in zip(range(len(x)),x,y,z):
            xid = get_index(xi,Lx,npoints_x)
            yid = get_index(yi,Ly,npoints_y)
            zid = get_index(zi,Lz,npoints_z)
            map_array[xid,yid,zid] = int(i)
        return map_array.ravel()
    
    def remap(in_vector, map_array, out_shape):
        in_vector = in_vector[map_array].reshape(out_shape)
        return in_vector

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
    map_array = mapping_array(x,y,z,out_shape,Lx,Ly,Lz)
    output = {k : remap(np.hstack(v),map_array,out_shape) for k, v in ret.items()}
    
    return output


def read_json_data(data):
    result = ''
    for i in range(len(data['atoms'])):
        temp = '{}\t{}\t{}\t{}\n'.format(data['atoms'][i], data['coordinates'][i][0],data['coordinates'][i][1],data['coordinates'][i][2])
        result += temp
    result += '\t symmetry {}'.format(data['symmetry'])
#    print(result)
    return psi4.geometry(result)




if __name__ == "__main__":
    database_filename = sys.argv[1]
    list_molecule_filename = sys.argv[2]
    
    
    
    with open(list_molecule_filename) as f:
        molecule_names = f.readlines()
    molecule_names = [x.strip() for x in molecule_names]
    
    try:
        data = json.load(open(database_filename,'rb'))
    except:
        with open(database_filename, encoding='utf-8') as f:
            data=json.load(f)
    
    molecules = {}
    for molecule in molecule_names:
        if molecule in data:
            molecules[molecule] = read_json_data(data[molecule])
            
        
    xc_funcs = ['B3LYP','SVWN','PBE','PBE0']
    h = 0.05
    L = 10.
    all_data = {}
    
    def log(log_filename, text):
        with open(log_filename, "a") as myfile:
            myfile.write(text)
        return
        
    failed_filename = "failed_molecule.log"
    succ_filename = "successful_molecule.log"
    
    
    for mol in molecules:
        for xc in xc_funcs:
            print('#@!#@!Molecule:'+mol)
            print('!@#!@#Method:' + xc)
            filename = '{}_{}.hdf5'.format(mol,xc)
            if not os.path.exists(filename):
                try:
                    out = get_DFT_outputs(molecules[mol],mol,xc,h,L)
                    with h5py.File(filename) as data:
                        for key in out:
                            data.create_dataset(key,data=out[key])
                        data.create_dataset('h_x',data=[h])
                        data.create_dataset('h_y',data=[h])
                        data.create_dataset('h_z',data=[h])
                    log(succ_filename, '\n' + mol)
                except:
                    log(failed_filename, '\n' + mol)
            else:
                data = h5py.File(filename,'r')
            all_data[mol+'_'+xc] = data
    print('Finished')
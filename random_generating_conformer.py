# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 19:34:19 2017

@author: ray
"""



import sys
import json
#import collections
import numpy as np
import math
#import multiprocessing
import json
import random
import pprint
import time
import copy


def generate(dim):
    """Generate a random rotation matrix.

    https://github.com/qobilidop/randrot

    Args:
        dim (int): The dimension of the matrix.

    Returns:
        np.matrix: A rotation matrix.

    Raises:
        ValueError: If `dim` is not 2 or 3.

    """
    if dim == 2:
        return generate_2d()
    elif dim == 3:
        return generate_3d()
    else:
        raise ValueError('Dimension {} is not supported. Use 2 or 3 instead.'
                         .format(dim))


def generate_2d():
    """Generate a 2D random rotation matrix.

    Returns:
        np.matrix: A 2D rotation matrix.

    """
    x = np.random.random()
    M = np.matrix([[np.cos(2 * np.pi * x), -np.sin(2 * np.pi * x)],
                   [np.sin(2 * np.pi * x), np.cos(2 * np.pi * x)]])
    return M


def generate_3d():
    """Generate a 3D random rotation matrix.

    Returns:
        np.matrix: A 3D rotation matrix.

    """
    x1, x2, x3 = np.random.rand(3)
    R = np.matrix([[np.cos(2 * np.pi * x1), np.sin(2 * np.pi * x1), 0],
                   [-np.sin(2 * np.pi * x1), np.cos(2 * np.pi * x1), 0],
                   [0, 0, 1]])
    v = np.matrix([[np.cos(2 * np.pi * x2) * np.sqrt(x3)],
                   [np.sin(2 * np.pi * x2) * np.sqrt(x3)],
                   [np.sqrt(1 - x3)]])
    H = np.eye(3) - 2 * v * v.T
    M = -H * R
    return M

def generate_random_conformer_control(original_molecule_data, molecule_name,number_random):

    result = {}

    for i in range(number_random):
        #random.seed(i)
        random_rot_matrix = generate(3)
        temp_molecule_name = "{}_rot_{}".format(molecule_name,str(i))
        result[temp_molecule_name] = {}
        result[temp_molecule_name]['atoms'] = original_molecule_data['atoms']
        result[temp_molecule_name]['symmetry'] = 'c1'
        temp_coordinates = copy.deepcopy(original_molecule_data["coordinates"])
        #print temp_coordinates
        #print temp_coordinates[1]
        for j in range(len(temp_coordinates)):
            #print random.gauss(0,0.01)
            
            temp_coordinates[j][0] += random.gauss(0,0.05)
            temp_coordinates[j][1] += random.gauss(0,0.05)
            temp_coordinates[j][2] += random.gauss(0,0.05)
            #print temp_coordinates[j]
            #print np.dot(random_rot_matrix,np.asarray(temp_coordinates[j]).reshape(3,1)).reshape(1,3).tolist() == np.dot(random_rot_matrix,temp_coordinates[j])
            temp_coordinates[j] = np.dot(random_rot_matrix,temp_coordinates[j]).tolist()[0]
        #print temp_coordinates[1]

        result[temp_molecule_name]['coordinates'] = temp_coordinates


    return result

def write_to_file(log_filename, text):
    with open(log_filename, "w") as myfile:
        json.dump(text,myfile,indent=4)
    return

if __name__ == "__main__":
        

    database_filename = sys.argv[1]
    molecule_name = sys.argv[2]
    number_random = int(sys.argv[3])
    result_data_name = sys.argv[4]
    

    molecule_names = [molecule_name]
    
    try:
        data = json.load(open(database_filename,'rb'))
    except:
        with open(database_filename, encoding='utf-8') as f:
            data=json.load(f)
    
    molecules = {}
    for molecule in molecule_names:
        if molecule in data:
#            molecules[molecule] = read_json_data(data[molecule])
            result_dict = generate_random_conformer_control(data[molecule],molecule,number_random)

#    result_string = pprint.pformat(result_dict,indent = 4)
#    print result_string

    write_to_file(result_data_name,result_dict)
                


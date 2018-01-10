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

def generate_random_conformer_control(original_molecule_data, molecule_name,number_random):

    result = {}

    for i in range(number_random):
        temp_molecule_name = "{}_{}".format(molecule_name,str(i))
        result[temp_molecule_name] = {}
        result[temp_molecule_name]['atoms'] = original_molecule_data['atoms']
        result[temp_molecule_name]['symmetry'] = 'c1'
        temp_coordinates = original_molecule_data["coordinates"]
        #print temp_coordinates
        for j in range(len(temp_coordinates)):
            temp_coordinates[j][0] += random.gauss(0,0.05)
            temp_coordinates[j][1] += random.gauss(0,0.05)
            temp_coordinates[j][2] += random.gauss(0,0.05)

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
                


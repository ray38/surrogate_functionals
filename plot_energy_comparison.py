# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 16:12:24 2017

@author: ray
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 13:56:12 2017

@author: ray
"""

import os
import itertools
import h5py
import json
import sys
import csv
import numpy as np
from numpy import mean, sqrt, square, arange
try: import cPickle as pickle
except: import pickle
import time
import pprint
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt



def sae(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true))

def map_to_0_1(arr, maxx, minn):
    return np.divide(np.subtract(arr,minn),(maxx-minn))
    
def map_back_0_1(arr, maxx, minn):
    return np.add(np.multiply(arr,(maxx-minn)),minn)
    
def map_back_n1_1(arr, maxx, minn):
    temp = np.multiply(np.add(arr,1.),0.5)
    return np.add(np.multiply(temp,(maxx-minn)),minn)

def log(log_filename, text):
    with open(log_filename, "a") as myfile:
        myfile.write(text)
    return

def get_start_loss(log_filename):
    
    with open(log_filename, 'r') as f:
        for line in f:
            pass
        temp = line
    
    if temp.strip().startswith('updated'):
        return float(temp.split()[2]), temp.split()[9]
    else:
        raise ValueError

    

def read_formation_energy_file(key,setup):
    os.chdir(setup[key]["model_save_dir"])
    setup[key]["result_data"] = {}
    setup[key]["exc_error_list"] = []
    setup[key]["formation_exc_error_list"] = []

    with open(setup[key][filename],'rb') as f:
        for line in f:
            if line.strip() != '':
                temp = line.strip().split()
                temp_name = temp[0]
                setup[key]["result_data"][temp_name] = {}

                temp_original_energy = float(temp[1])
                temp_predict_energy  = float(temp[2])
                temp_energy_error  = float(temp[3])
                temp_original_formation_energy = float(temp[4])
                temp_predict_formation_energy  = float(temp[5])
                temp_formation_energy_error  = float(temp[6])
                setup[key]["result_data"][temp_name]['predict_exc'] = temp_predict_energy
                setup[key]["result_data"][temp_name]['original_exc'] = temp_original_energy
                setup[key]["result_data"][temp_name]['exc_error'] = temp_energy_error
                setup[key]["result_data"][temp_name]['predict_formation_exc'] = temp_predict_formation_energy
                setup[key]["result_data"][temp_name]['original_formation_exc'] = temp_original_formation_energy
                setup[key]["result_data"][temp_name]['formation_exc_error'] = temp_formation_energy_error

                setup[key]["exc_error_list"].append(temp_energy_error)
                setup[key]["formation_exc_error_list"].append(temp_formation_energy_error)

    return


def determine_training_test(molecule_name):
    if molecule_name in ["C2H2","C2H4","C2H6","CH3OH","CH4","CO","CO2","H2","H2O","HCN","HNC","N2","N2O","NH3","O3"]:
        return "training"
    else:
        return "test"

def prepare_pandas_dataframe(setup):
    predict_exc_list = []
    original_exc_list = []
    exc_error_list = []
    predict_formation_exc_list = []
    original_formation_exc_list = []
    formation_exc_error_list = []
    molecule_name_list = []
    training_test_list = []
    dataset_name_list = []
    model_name_list = []
    model_list = []
    filename_list = []

    for model_name in setup:
        for molecule_name in setup[model_name]["result_data"]:
            model_list.append(setup[model_name]["model"])
            filename_list.append(setup[model_name]["filename"])
            model_name_list.append(model_name)
            dataset_name_list.append(setup[model_name]["dataset"])
            training_test_list.append(determine_training_test(molecule_name))
            molecule_name_list.append(molecule_name)
            predict_exc_list.append(setup[model_name]["result_data"][molecule_name]['predict_exc'])
            original_exc_list.append(setup[model_name]["result_data"][molecule_name]['original_exc'])
            exc_error_list.append(setup[model_name]["result_data"][molecule_name]['exc_error'])
            predict_formation_exc_list.append(setup[model_name]["result_data"][molecule_name]['predict_formation_exc'])
            original_formation_exc_list.append(setup[model_name]["result_data"][molecule_name]['original_formation_exc'])
            formation_exc_error_list.append(setup[model_name]["result_data"][molecule_name]['formation_exc_error'])

    d = {"predict_exc":predict_exc_list, "original_exc":original_exc_list, "exc_error":exc_error_list, \
        "predict_formation_exc":predict_formation_exc_list, "original_formation_exc":original_formation_exc_list, "formation_exc_error": formation_exc_error_list, \
        "molecule_name": molecule_name_list, "training_test":training_test_list, "dataset":dataset_name_list, \
        "model_name":model_name_list, "model":model_list, "filename":filename_list}

    return pd.DataFrame(data=d)




def plot_result(setup):

if __name__ == "__main__":


    print "start"
    setup_filename = sys.argv[1]
    #functional = sys.argv[2]



    with open(predict_setup_filename) as f:
        setup = json.load(f)


    #setup['functional'] = functional
#    functional = setup['functional']

    with open(dataset_setup_database_filename) as f:
        setup_database = json.load(f)


    main_dir = os.getcwd()

    for key in setup:

        dir_name = "10-0_0-02_5"

        working_dir = os.getcwd() + '/' + dir_name + '/' + setup[key]["dataset"]
        setup[key]["working_dir"] = working_dir

        model_save_dir = working_dir + "/" + setup[key]["model"]
        setup[key]["model_save_dir"] = model_save_dir


    for key in setup:
        os.chdir(main_dir)
        read_formation_energy_file(key,setup):
    os.chdir(main_dir)

    data = prepare_pandas_dataframe(setup)

    sns.set(style="whitegrid", palette="pastel", color_codes=True)

    # Draw a nested violinplot and split the violins for easier comparison
    sns.violinplot(x="model_name", y="formation_exc_error", hue="training_test", data=data, split=True,
                   inner="quart", palette={"training": "b", "test": "y"})
    sns.despine(left=True)

    plt.savefig("formation_energy_grouped_violin_plot.png")


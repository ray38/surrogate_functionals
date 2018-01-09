# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:43:05 2017

@author: ray
"""

import matplotlib
matplotlib.use('Agg') 
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt

import numpy as np
import sys
import os
import time
import math
import json
from glob import glob
from sklearn import linear_model
#from keras.models import Sequential
#from keras.models import load_model
#from keras.layers import Dense, Activation
#import keras


import itertools
import multiprocessing

try: import cPickle as pickle
except: import pickle

from subsampling import subsampling_system,random_subsampling,subsampling_system_with_PCA

def map_to_n1_1(arr, maxx, minn):
    return np.divide(np.subtract(arr,minn),(maxx-minn)/2.)-1.
    
def log(log_filename, text):
    with open(log_filename, "a") as myfile:
        myfile.write(text)
    return

def map_to_0_1(arr, maxx, minn):
    return np.divide(np.subtract(arr,minn),(maxx-minn))
    
def map_back(arr, maxx, minn):
    return np.add(np.multiply(arr,(maxx-minn)),minn)


def fit_with_Linear(X,y):

    filename = "Linear_model.sav"

    try:
        li_model = pickle.load(open(filename, 'rb'))
        print 'model loaded'
    except:
        li_model = linear_model.LinearRegression()
        li_model.fit(X, y)
        pickle.dump(li_model, open(filename, 'wb'))
    
    # The coefficients
    print 'Coefficients: \n', li_model.coef_
    # The mean squared error
    print "Mean squared error: %.20f" % np.mean((li_model.predict(X) - y) ** 2)
    # Explained variance score: 1 is perfect prediction
    print 'Variance score: %.20f' % li_model.score(X, y)
    
    residual = y-li_model.predict(X)
#    plt.scatter(X, residual,  color='black')
#    plt.show()
    return residual, li_model


def read_data_from_one_dir(directory):
    temp_cwd = os.getcwd()
    os.chdir(directory)

    print directory

    subsampled_filename = "overall_subsampled_data.p"
    random_filename = "overall_random_data.p"

    try:
        molecule_subsampled_data = pickle.load(open(subsampled_filename,'rb'))
        print "read subsampled data"
    except:
        molecule_subsampled_data = []

    try:
        molecule_random_data = pickle.load(open(random_filename,'rb'))
        print "read random data"
    except:
        molecule_random_data = []

    os.chdir(temp_cwd)

    return molecule_subsampled_data, molecule_random_data



def get_training_data(dataset_name,setup):

    data_dir_name = setup["working_dir"] + "/data/*/" 
    data_paths = glob(data_dir_name)
    print data_paths


    overall_subsampled_data = []
    overall_random_data = []

    plot_molecule_name_list = []
    plot_molecule_plot_list = []
    plot_molecule_target_list = []
    plot_molecule_dens_list = []


    num_samples = len(data_paths)
    num_random_per_molecule = int(math.ceil(float(setup["random_pick"])/float(num_samples)))
    for directory in data_paths:

        temp_molecule_subsampled_data, temp_molecule_random_data = read_data_from_one_dir(directory)
        overall_subsampled_data += temp_molecule_subsampled_data
        temp_added_random_data = random_subsampling(temp_molecule_random_data, num_random_per_molecule)
        overall_random_data += temp_added_random_data

        temp_molecule_name = directory.split('/')[-2] + directory.split('/')[-1]
        print temp_molecule_name
        plot_molecule_name_list.append(temp_molecule_name)
        temp_molecule_data = temp_molecule_subsampled_data + temp_added_random_data
        plot_molecule_plot_list.append(temp_molecule_data)

        temp_target = []
        temp_dens = []
        for entry in temp_molecule_data:
            temp_dens.append(entry[1])
            temp_target.append(entry[0])

        plot_molecule_dens_list.append(temp_dens)
        plot_molecule_target_list.append(temp_target)


        


    list_subsample = setup["subsample_feature_list"]
    temp_list_subsample = setup["subsample_feature_list"]
    if temp_list_subsample == []:
        for m in range(len(overall_subsampled_data[0])):
            temp_list_subsample.append(m)

    if len(temp_list_subsample) <= 10:
        overall_subsampled_data = subsampling_system(overall_subsampled_data, list_desc = list_subsample, cutoff_sig = float(setup["subsample_cutoff_sig"]), rate = float(setup["subsample_rate"]))
    else:
        overall_subsampled_data = subsampling_system_with_PCA(overall_subsampled_data, list_desc = list_subsample, cutoff_sig = float(setup["subsample_cutoff_sig"]), rate = float(setup["subsample_rate"]),start_trial_component = 9)


    overall = overall_random_data + overall_subsampled_data



    X_train = []
    y_train = []
    dens = []

    for entry in overall:
#        if entry[0] >= lower and entry[0] <= upper:
        X_train.append(list(entry[1:]))
        dens.append(entry[1])
        y_train.append(entry[0])
    
    
    X_train = (np.asarray(X_train))
    y_train = np.asarray(y_train).reshape((len(y_train),1))
    dens = np.asarray(dens).reshape((len(dens),1))
    
    return X_train, y_train, dens, plot_molecule_name_list, plot_molecule_plot_list, plot_molecule_target_list, plot_molecule_dens_list





def plot_2Dplots(title, x_list, y_list, label_list):

#    fig=plt.figure()
#    colors = cm.rainbow(np.linspace(0, 1, molecule_counter+1))

    #colors = ['b','g','r','c','m','y','k','turquoise','navy', 'darkorange']

#    for color, i, target_name, z_order in zip(colors, range(molecule_counter+1), Overall_Molecule_Name_List, z_order_list):
#        plt.scatter(X_pca[Overall_Molecule_counter_List == i, 0],X_pca[Overall_Molecule_counter_List == i, 1], 
#                        c= color,  lw = 0,label=target_name,alpha=0.6, zorder = z_order)

    fig=plt.figure(figsize=(40,40))
    colors = cm.rainbow(np.linspace(0, 1, len(label_list)))
    for x, y, label, color in zip(x_list,y_list,label_list,colors):
        plt.scatter(x, y,  c= color,  lw = 0,label=target_name,alpha=0.6)


    plt.title(title)
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.savefig(title + '.png')

    return



def plot_3Dplots():
    fig=plt.figure()
    ax = p3.Axes3D(fig)
    for color, i, target_name, z_order in zip(colors, range(molecule_counter+1), Overall_Molecule_Name_List, z_order_list):
        temp_z = np.ones_like(X_pca[Overall_Molecule_counter_List  == i, 0].copy()) * float(i+1)*0.5
        ax.scatter(X_pca[Overall_Molecule_counter_List  == i, 0],temp_z, X_pca[Overall_Molecule_counter_List  == i, 1], 
                        c= color,  facecolor='0.5', lw = 0,label=target_name,alpha=0.6)

                        

    #                '''[color] * len(temp_z)'''

    plt.title("PCA Test")
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.savefig('PCA_3d_plot.png')




def prepare_linear_residual_data(li_model, plot_molecule_target_list, plot_molecule_dens_list):
    
    result = []
    for i in range(len(plot_molecule_target_list)):
        temp_residual = plot_molecule_target_list[i] - li_model.predict(plot_molecule_dens_list[i])
        result.append(temp_residual)

    return result






if __name__ == "__main__":


    setup_filename = sys.argv[1]
    dataset_name = sys.argv[2]

    with open(setup_filename) as f:
        setup = json.load(f)


    h = float(setup['grid_spacing'])
    L = float(setup['box_dimension'])
    N = int(setup['number_segment_per_side'])
    dir_name = "{}_{}_{}".format(str(L).replace('.','-'),str(h).replace('.','-'),N)

    working_dir = os.getcwd() + '/' + dir_name + '/' + dataset_name

    setup["working_dir"] = working_dir

    model_save_dir = working_dir + "/" + "Plot_linear_residual_{}".format(setup["random_pick"])
   
    setup["model_save_dir"] = model_save_dir

    
    
    X_train,y, dens, plot_molecule_name_list, plot_molecule_plot_list, plot_molecule_target_list, plot_molecule_dens_list = get_training_data(dataset_name,setup)
   
    if os.path.isdir(model_save_dir) == False:
        os.makedirs(model_save_dir)

    os.chdir(model_save_dir)
    
    residual,li_model = fit_with_Linear(dens,y)
    #model = fit_with_KerasNN(X_train,residual, tol, slowdown_factor, early_stop_trials)

    plot_molecule_residual_list = prepare_linear_residual_data(li_model, plot_molecule_target_list, plot_molecule_dens_list)

    plot_2Dplots('target_vs_dens', plot_molecule_dens_list, plot_molecule_target_list, plot_molecule_name_list)

    plot_2Dplots('residual_vs_dens', plot_molecule_dens_list, plot_molecule_residual_list, plot_molecule_name_list)


    
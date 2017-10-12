# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 14:13:59 2017

@author: ray
"""

import numpy as np
import sys
import os
import time
import math
from sklearn import linear_model
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation
import keras
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3

import itertools
import multiprocessing

try: import cPickle as pickle
except: import pickle
import matplotlib.pyplot as plt
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

def get_start_loss(log_filename):
    
    with open(log_filename, 'r') as f:
        for line in f:
            pass
        temp = line
    
    if temp.strip().startswith('updated'):
        return float(temp.split()[2])
    else:
        raise ValueError

def fit_with_Linear(X,y, functional, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform, lower, upper):
    filename = 'Linear_{}_model_{}_gamma{}_dev{}_devsq{}_inte{}_{}_{}_{}_{}.sav'.format(functional, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform, str(int(lower)).replace('-','n').replace('.','-'), str(int(upper)).replace('-','n').replace('.','-'))

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
    return residual

def fit_with_KerasNN(X,y,functional, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform, lower, upper, n_per_layer, n_layers, activation,tol, slowdown_factor, early_stop_trials):
    
#    colors = map_to_0_1(y, max(y),min(y))
#    fig = plt.figure()
#    ax = p3.Axes3D(fig)    
#    ax.scatter(X[:,0],X[:,1],y,  c=colors, cmap='hsv', linewidth = 0, alpha=1,s=3)
##    ax.scatter(X_train[:,0],X_train[:,1],y_train,linewidth = 0, alpha=1,s=3)
##    ax.set_ylim(-10.,10.)
#    plt.savefig('error_test2.png')    
#    plt.show()
    
    
    filename = 'NN_{}_linear_residual_{}_{}_{}_{}_gamma{}_dev{}_devsq{}_inte{}_{}_{}_{}_{}.h5'.format(functional, n_per_layer,n_layers,activation, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform, target_transform, str(int(lower)).replace('-','n').replace('.','-'), str(int(upper)).replace('-','n').replace('.','-'))
    log_filename = 'NN_{}_linear_residual_{}_{}_{}_{}_gamma{}_dev{}_devsq{}_inte{}_{}_{}_{}_{}_log.log'.format(functional, n_per_layer,n_layers,activation, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform, target_transform, str(int(lower)).replace('-','n').replace('.','-'), str(int(upper)).replace('-','n').replace('.','-'))
    try:
        model = load_model(filename)
        restart = True
        print 'model loaded: ' + filename
    except:
        restart = False
        n = int(n_per_layer)
        k = len(X[0])
        print n,k
        model = Sequential()
        model.add(Dense(output_dim =n, input_dim = k, activation = activation))
    
        if n_layers > 1:        
            for i in range(int(n_layers-1)):
                model.add(Dense(input_dim = n, output_dim  = n, activation = activation))
        model.add(Dense(input_dim = n,output_dim =1, activation = 'linear'))
    
    #    model.add(Dense(input_dim = 1,output_dim =1, activation = 'linear',  init='uniform'))
    
    print 'model set'
    default_lr = 0.001
    adam = keras.optimizers.Adam(lr=default_lr / slowdown_factor)
    model.compile(loss='mse',#custom_loss,
              optimizer=adam)
              #metrics=['mae'])
    print model.summary()
    print model.get_config()
    
    est_start = time.time()
    history_callback = model.fit(X, y, nb_epoch=1, batch_size=500000)
    est_epoch_time = time.time()-est_start
    if est_epoch_time >= 30.:
        num_epoch = 1
    else:
        num_epoch = int(math.floor(30./est_epoch_time))
    if restart == True:
        try:
            start_loss = get_start_loss(log_filename)
        except:
            loss_history = history_callback.history["loss"]
            start_loss = np.array(loss_history)[0]
    else:
        loss_history = history_callback.history["loss"]
        start_loss = np.array(loss_history)[0]
    
    log(log_filename, "\n start: "+ str(start_loss))
    
    old_loss = start_loss
    keep_going = True
    
    count_epochs = 0
    while keep_going:
        count_epochs += 1
        history_callback = model.fit(X, y, nb_epoch=num_epoch, batch_size=500000, shuffle=True)
        loss_history = history_callback.history["loss"]
        new_loss = np.array(loss_history)[-1]
        if new_loss < old_loss:
            model.save(filename)
            print 'model saved'
            log(log_filename, "\n updated best: "+ str(new_loss) + " \t epochs since last update: " + str(count_epochs))
            old_loss = new_loss
            count_epochs = 0
        if new_loss < tol:
            keep_going = False
        if count_epochs >=early_stop_trials:
            keep_going = False
    
#    plt.scatter(X, y-model.predict(X),  color='black')
    
    
    plt.scatter(X[:,0], y,  color='black')
    plt.scatter(X[:,0], model.predict(X), color='blue',
             linewidth=3)
    test_x = np.linspace(-6., 5., 10000)
    plt.plot(test_x, model.predict(test_x),color='blue')
    plt.show()
    return model


def process_one_molecule(molecule, functional,h,L,N, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform):
    temp_cwd = os.getcwd()
    molecule_dir_name = "{}_{}_{}_{}_{}".format(molecule,functional,str(L).replace('.','-'),str(h).replace('.','-'),N)
    subsampled_data_dir = "{}_{}_gamma{}_dev{}_devsq{}_inte{}_{}_{}".format(functional, target, gamma,num_desc_deri,num_desc_deri_squa,num_desc_ave_dens,desc_transform,target_transform) 
    
    if os.path.isdir(molecule_dir_name + '/' + subsampled_data_dir) == False:
        print '\n****Error: Cant find the directory! ****\n'
        raise NotImplementedError
    
    os.chdir(temp_cwd + '/' + molecule_dir_name + '/' + subsampled_data_dir)
    
    molecule_overall_filename = "{}_{}_all_{}_gamma{}_dev{}_devsq{}_inte{}_{}_{}_subsampled_data.p".format(molecule,functional, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform)
    
    try:
        molecule_overall = pickle.load(open(molecule_overall_filename,'rb'))
    
    except:
        Nx = Ny = Nz = N
        i_li = range(Nx)
        j_li = range(Ny)
        k_li = range(Nz)
        paramlist = list(itertools.product(i_li,j_li,k_li))
        
        molecule_raw_overall = []
        for i,j,k in paramlist:
            temp_filename = "{}_{}_{}_{}_{}_{}_gamma{}_dev{}_devsq{}_inte{}_{}_{}_subsampled_data.p".format(molecule,functional,i,j,k, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform)
            temp = pickle.load(open(temp_filename,'rb'))
            molecule_raw_overall += temp
        
#        for k in range(Nz):
#            temp_filename = "{}_{}_{}_{}_{}_{}_gamma{}_dev{}_devsq{}_inte{}_{}_{}_subsampled_data.p".format(molecule,functional,k,k,k, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform)
#            temp = pickle.load(open(temp_filename,'rb'))
#            molecule_raw_overall += temp
            
        
        molecule_overall = subsampling_system_with_PCA(molecule_raw_overall, list_desc = [], cutoff_sig = 0.002, rate = 0.2,start_trial_component = 9)
        
        with open(molecule_overall_filename, 'wb') as handle:
            pickle.dump(molecule_overall, handle, protocol=2)


    os.chdir(temp_cwd)
    
    return molecule_overall


def get_training_data(list_molecule_filename,functional, h,L,N,target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform, lower, upper):
    with open(list_molecule_filename) as f:
        molecule_names = f.readlines()
    molecule_names = [x.strip() for x in molecule_names]

    
    raw_overall = []
    for molecule in molecule_names:
#        try:
        temp = process_one_molecule(molecule, functional,h,L,N, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform)
        raw_overall += temp
        print 'success: ' + molecule
#        except:
#            print 'failed: ' + molecule
    print len(raw_overall)
    overall = subsampling_system_with_PCA(raw_overall, list_desc = [], cutoff_sig = 0.002, rate = 0.2,start_trial_component = 9)

    X_train = []
    y_train = []
    dens = []

    for entry in overall:
        if entry[0] >= lower and entry[0] <= upper:
            X_train.append(list(entry[1:]))
            dens.append(entry[1])
            y_train.append(entry[0])
    
    
    X_train = (np.asarray(X_train))
    y_train = np.asarray(y_train).reshape((len(y_train),1))
    dens = np.asarray(dens).reshape((len(dens),1))
    

    print X_train.shape
    print y_train.shape
    print dens.shape
    return X_train, y_train, dens


if __name__ == "__main__":

    list_molecule_filename = sys.argv[1]
    functional = sys.argv[2]
    h = float(sys.argv[3])
    L = float(sys.argv[4])
    N = int(sys.argv[5])
    gamma = int(sys.argv[6])
    num_desc_deri = int(sys.argv[7])
    num_desc_deri_squa = int(sys.argv[8])
    num_desc_ave_dens = int(sys.argv[9])
    target = sys.argv[10]
    desc_transform = sys.argv[11]
    target_transform = sys.argv[12]
    lower = float(sys.argv[13])
    upper = float(sys.argv[14])
    n_per_layer = int(sys.argv[15])
    n_layers    = int(sys.argv[16])
    activation_choice = sys.argv[17]
    slowdown_factor = float(sys.argv[18])
    tol = float(sys.argv[19])
    try:
        early_stop_trials = int(sys.argv[20])
    except:
        early_stop_trials = 100
    
    if activation_choice not in ['tanh','relu','sigmoid','softmax']:
        raise ValueError
    
    if desc_transform not in ['log','real']:
        raise ValueError
    
    if target_transform not in ['log','real','negreal']:
        raise ValueError
    
    #if dataset_choice not in ['all','dens']:
    #    raise ValueError
    
    #print device_lib.list_local_devices()
    cwd = os.getcwd()
    result_dir = "{}_{}_{}_{}_{}_gamma{}_dev{}_devsq{}_inte{}_{}_{}_models".format(functional,str(L).replace('.','-'),str(h).replace('.','-'),N,target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform)
    if os.path.isdir(result_dir) == False:
        os.makedirs(cwd + '/' + result_dir)     
    
    
    
    X_train,y, dens = get_training_data(list_molecule_filename,functional, h,L,N,target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform, lower, upper)
   
    
    os.chdir(cwd + '/' + result_dir)
    
    residual = fit_with_Linear(dens,y,functional, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform, lower, upper)
    model = fit_with_KerasNN(X_train,residual,functional, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform, lower, upper, n_per_layer, n_layers, activation_choice,tol, slowdown_factor, early_stop_trials)
    
    os.chdir(cwd)
    
    
    
    



# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:43:05 2017

@author: ray
"""

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
#    plt.scatter(X, residual,  color='black')
#    plt.show()
    return residual, li_model

def fit_with_KerasNN(X,y,functional, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform, lower, upper, n_per_layer, n_layers, activation,tol, slowdown_factor, early_stop_trials):

    
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
    history_callback = model.fit(X, y, nb_epoch=1, batch_size=50000)
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
        history_callback = model.fit(X, y, nb_epoch=num_epoch, batch_size=50000, shuffle=True)
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
    
    return model


def process_one_molecule(molecule, functional,h,L,N, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform):
    temp_cwd = os.getcwd()
    molecule_dir_name = "{}_{}_{}_{}_{}".format(molecule,functional,str(L).replace('.','-'),str(h).replace('.','-'),N)
    subsampled_data_dir = "{}_{}_gamma{}_dev{}_devsq{}_inte{}_{}_{}_tau".format(functional, target, gamma,num_desc_deri,num_desc_deri_squa,num_desc_ave_dens,desc_transform,target_transform) 
    
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
            try:
                temp = pickle.load(open(temp_filename,'rb'))
                molecule_raw_overall += temp
            except:
                print temp_filename + " load failed! passed!"
                
        
#        for k in range(Nz):
#            temp_filename = "{}_{}_{}_{}_{}_{}_gamma{}_dev{}_devsq{}_inte{}_{}_{}_subsampled_data.p".format(molecule,functional,k,k,k, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform)
#            temp = pickle.load(open(temp_filename,'rb'))
#            molecule_raw_overall += temp
            
#        molecule_overall = molecule_raw_overall
        molecule_overall = subsampling_system_with_PCA(molecule_raw_overall, list_desc = [], cutoff_sig = 0.002, rate = 0.2,start_trial_component = 9)
#        molecule_overall = random_subsampling(molecule_raw_overall,50000)
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
#    overall = subsampling_system_with_PCA(raw_overall, list_desc = [], cutoff_sig = 0.002, rate = 0.2,start_trial_component = 9)
    overall = raw_overall

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



def process_error2(X,num_intervals=40):
    
#    temp_x = np.linspace(min(X),max(X),num_intervals+1)
    temp_x = np.linspace(-0.12,0.12,num_intervals+1)
    dx = temp_x[1] - temp_x[0]
    
    x_labels = []
    for i in range(len(temp_x)-1):
        x_labels.append(str(round(temp_x[i] + 0.5*dx,3)))
    

    count_list = np.zeros(len(x_labels))
    
    print temp_x[0]
    for i in range(len(X)):

        index = abs(int((X[i]-temp_x[0])//dx))

        
        if index >num_intervals-1:
            index = num_intervals-1
        count_list[index] += 1.

    
 
    return x_labels, count_list


def process_error3(X,num_intervals=40):
    
#    temp_x = np.linspace(min(X),max(X),num_intervals+1)
    temp_x = np.linspace(-0.0012,0.0012,num_intervals+1)
    dx = temp_x[1] - temp_x[0]
    
    x_labels = []
    for i in range(len(temp_x)-1):
        x_labels.append(str(round(temp_x[i] + 0.5*dx,3)))
    

    count_list = np.zeros(len(x_labels))
    
    print temp_x[0]
    for i in range(len(X)):

        index = abs(int((X[i]-temp_x[0])//dx))

        
        if index >num_intervals-1:
            index = num_intervals-1
        count_list[index] += 1.

    
 
    return x_labels, count_list


    
    return
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
    result_dir = "{}_{}_{}_{}_{}_gamma{}_dev{}_devsq{}_inte{}_{}_{}_models_asym".format(functional,str(L).replace('.','-'),str(h).replace('.','-'),N,target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform)
    if os.path.isdir(result_dir) == False:
        os.makedirs(cwd + '/' + result_dir)     
    
    
    
    X_train,y, dens = get_training_data(list_molecule_filename,functional, h,L,N,target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform, lower, upper)
   
    
    os.chdir(cwd + '/' + result_dir)
    
    residual,li_model = fit_with_Linear(dens,y,functional, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform, lower, upper)
    model = fit_with_KerasNN(X_train,residual,functional, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,desc_transform,target_transform, lower, upper, n_per_layer, n_layers, activation_choice,tol, slowdown_factor, early_stop_trials)
    
    
    
    
    os.chdir(cwd)

    
    predict_y_log = model.predict(X_train) + li_model.predict(dens)
    y_log = y
    error_log = y_log-predict_y_log
    

    
#    error = np.multiply(-1.,(np.power(10.,model.predict(X_train)-y)))
    predict_y_real = np.multiply(-1.,(np.power(10.,model.predict(X_train) + li_model.predict(dens) )))
    y_real = np.multiply(-1.,(np.power(10.,y)))
    error_real = y_real-predict_y_real
#    fraction_error = np.divide(error, y)
    

    
    x_labels_real, counts_real = process_error3(error_real)
    x_labels_log, counts_log = process_error2(error_log)
    y_pos_log = np.arange(len(x_labels_log))
    y_pos_real = np.arange(len(x_labels_real))
    
    fig, axes = plt.subplots(2, 2,figsize = (20,20))
    ((ax1,ax2),(ax3,ax4)) = axes
    ax1.bar(y_pos_log, counts_log/len(y_log), align = 'center', alpha = 0.5)
    ax1.set_xticks(y_pos_log)
    ax1.set_xticklabels(x_labels_log, rotation=45, fontsize=18)
    for label in ax1.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    for label in ax1.yaxis.get_ticklabels():
        label.set_fontsize(18)
    ax1.text(0.045, 0.12, 'MAE:{0:.2e}'.format(np.average(np.abs(error_log))) +  '\nMSE:{0:.2e}'.format(np.average(error_log)), fontsize=20)
    ax1.set_xlabel('error in log space', fontsize=20)
    ax1.set_ylabel('error distribution', fontsize=20)
    
    ax2.bar(y_pos_real, counts_real/len(y_log), align = 'center', alpha = 0.5)
    ax2.set_xticks(y_pos_real)
    ax2.set_xticklabels(x_labels_real, rotation=45, fontsize=18)
    for label in ax2.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    for label in ax2.yaxis.get_ticklabels():
        label.set_fontsize(18)
    ax2.text(0.045, 0.12, 'MAE:{0:.2e}'.format(np.average(np.abs(error_real))) +  '\nMSE:{0:.2e}'.format(np.average(error_real)), fontsize=20)
    ax2.set_xlabel('error in real space', fontsize=20)
    ax2.set_ylabel('error distribution', fontsize=20)
    
    ax3.scatter(y_log,predict_y_log,color='blue',s=5)
    ax3.plot([-13,4],[-13,4],'r--')
    for label in ax3.xaxis.get_ticklabels():
        label.set_fontsize(18)
    for label in ax3.yaxis.get_ticklabels():
        label.set_fontsize(18)
    ax3.set_xlabel('log( B3LYP XC )', fontsize=20)
    ax3.set_ylabel('log( predicted B3LYP XC )', fontsize=20)
    
    ax4.scatter(y_real,predict_y_real,color='blue',s=5)
    ax4.plot([np.min(y_real),np.max(y_real)],[np.min(y_real),np.max(y_real)],'r--')
    for label in ax4.xaxis.get_ticklabels():
        label.set_fontsize(18)
    for label in ax4.yaxis.get_ticklabels():
        label.set_fontsize(18)
    ax4.set_xlabel('B3LYP XC', fontsize=20)
    ax4.set_ylabel('predicted B3LYP XC', fontsize=20)
    
    plt.savefig('training_set_error_hist_desc.png')
#    plt.show()





    
    
    



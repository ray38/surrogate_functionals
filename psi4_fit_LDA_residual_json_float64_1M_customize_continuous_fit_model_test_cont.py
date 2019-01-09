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
import csv
import sys
import os
import time
import math
import json
from glob import glob
from sklearn import linear_model
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation
from keras import backend as K
import keras
import scipy

import itertools
import multiprocessing

try: import cPickle as pickle
except: import pickle
import matplotlib.pyplot as plt
from subsampling import subsampling_system,random_subsampling,subsampling_system_with_PCA
import keras.backend as K

def sae(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true))

def map_to_n1_1(arr, maxx, minn):
    return np.divide(np.subtract(arr,minn),(maxx-minn)/2.)-1.
    
def log(log_filename, text):
    with open(log_filename, "a") as myfile:
        myfile.write(text)
    return


def write(log_filename, text):
    with open(log_filename, "w") as myfile:
        myfile.write(text)
    return

def map_to_0_1(arr, maxx, minn):
    return np.divide(np.subtract(arr,minn),(maxx-minn))
    
def map_back(arr, maxx, minn):
    return np.add(np.multiply(arr,(maxx-minn)),minn)

def get_start_loss(log_filename,loss):
    
    with open(log_filename, 'r') as f:
        for line in f:
            pass
        temp = line
    
    if temp.strip().startswith('updated') and temp.split()[9] == loss:
        return float(temp.split()[2])
    else:
        raise ValueError

def fit_with_KerasNN(setup, X, y, loss, tol, slowdown_factor, early_stop_trials):

    loss_list = ["mse","sae","mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error", "mean_squared_logarithmic_error", "squared_hinge", "hinge", "categorical_hinge", "logcosh", "categorical_crossentropy", "sparse_categorical_crossentropy", "binary_crossentropy", "kullback_leibler_divergence", "poisson", "cosine_proximity"]
    if loss not in loss_list:
        raise NotImplemented


    filename = "NN.h5"
    log_filename = "NN_fit_log.log"
    temp_check_filename = "NN_fit_checkpoint.log"
    num_samples = len(y)

    n_layers = setup["NN_setup"]["number_layers"]
    n_per_layer = setup["NN_setup"]["number_neuron_per_layer"]
    activation = setup["NN_setup"]["activation"]



    try:
        model = load_model(filename, custom_objects={'sae': sae})
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
    if loss == "sae":
        model.compile(loss=sae,#loss='mse',#loss='mean_absolute_percentage_error',#custom_loss,
                  optimizer=adam)
                  #metrics=['mae'])
    else:
        model.compile(loss=loss,#loss='mean_absolute_percentage_error',#custom_loss,
                  optimizer=adam)

    print model.summary()
    print model.get_config()
    
    history_callback_kickoff = model.fit(X, y, nb_epoch=1, batch_size=50000, shuffle=True)
    est_start = time.time()
    history_callback = model.fit(X, y, nb_epoch=1, batch_size=50000, shuffle=True)
    est_epoch_time = time.time()-est_start
    if est_epoch_time >= 2.:
        num_epoch = 1
    else:
        num_epoch = int(math.floor(2./est_epoch_time))
    if restart == True:
        try:
            start_loss = get_start_loss(log_filename,loss)
        except:
            loss_history = history_callback.history["loss"]
            start_loss = np.array(loss_history)[0]
    else:
        loss_history = history_callback.history["loss"]
        start_loss = np.array(loss_history)[0]
    
    log(log_filename, "\n loss: {} \t start: {} \t slowdown: {} \t early stop: {} \t target tolerence: {}".format(loss, str(start_loss), slowdown_factor, early_stop_trials, tol))
    
    best_loss = start_loss
    best_model = model
    keep_going = True
    
    count_epochs = 0
    log(log_filename, "\n updated best: "+ str(start_loss) + " \t epochs since last update: " + str(count_epochs) + " \t loss: " + loss)
    while keep_going:
        count_epochs += 1
        print count_epochs
        history_callback = model.fit(X, y, nb_epoch=num_epoch, batch_size=50000, shuffle=True)
        loss_history = history_callback.history["loss"]
        new_loss = np.array(loss_history)[-1]
        write(temp_check_filename, "\n updated best: "+ str(new_loss) + " \t epochs since last update: " + str(count_epochs) + " \t loss: " + loss + "\t num_samples: " + str(num_samples))
        if new_loss < best_loss:
            model.save(filename)
            print 'model saved'
            best_model = model
            if loss == "sae":
                log(log_filename, "\n updated best: "+ str(new_loss) + " \t epochs since last update: " + str(count_epochs) + " \t loss: " + loss + " \t projected error: " + str(((new_loss/1e6)*0.02*0.02*0.02*27.2114)*125/3)  )
            else:
                log(log_filename, "\n updated best: "+ str(new_loss) + " \t epochs since last update: " + str(count_epochs) + " \t loss: " + loss)
            best_loss = new_loss
            count_epochs = 0
        if new_loss < tol:
            keep_going = False
        if count_epochs >=early_stop_trials:
            keep_going = False
    

    best_model.save("NN_{}_{}_backup.h5".format(loss,best_loss))
    return best_model, best_loss


def fit_with_LDA(density,energy):

    filename = "LDA_model.sav"
    text_filename = "LDA_model_result.txt"

    #try: 
    #    temp_res = pickle.load(open(filename, 'rb'))
    #    res = temp_res
    #except:
    #    x0 = get_x0()
    #
    #    density = np.asarray(density)
    #    energy = np.asarray(energy)
    #
    #   res = scipy.optimize.minimize(LDA_least_suqare_fit, x0, args=(density,energy), method='nelder-mead',options={'xtol': 1e-13, 'disp': True, 'maxiter': 100000})

    #    print res.x
    #    pickle.dump(res, open(filename, 'wb'))
    temp_res = pickle.load(open(filename, 'rb'))
    res = temp_res


    predict_y = predict_LDA(density,res.x)
    residual = energy - predict_y

    return residual, res


def LDA_least_suqare_fit(x,density,energy):

    #result = 0
    #for n, e in density, energy:
    #    result += (lda_x(n,x) + lda_c(n,x) - e)**2

    result = np.mean(np.square(lda_x(density,x) + lda_c(density,x) - energy))
    return result


def get_x0():
    x = [ -0.45816529328314287, 0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294]
    return x

def optimization_constants(x):
    #C0I = x[0]
    #C1  = x[1]
    #CC1 = x[2]
    #CC2 = x[3]
    #IF2 = x[4]

    C1  = x[0]
    gamma = x[1]
    alpha1 = x[2]
    beta1 = x[3]
    beta2 = x[4]
    beta3 = x[5]
    beta4 = x[6]

    #return C0I, C1, CC1, CC2, IF2, gamma, alpha1, beta1, beta2, beta3, beta4
    return C1, gamma, alpha1, beta1, beta2, beta3, beta4

def G(rtrs, gamma, alpha1, beta1, beta2, beta3, beta4):
    Q0 = -2.0 * gamma * (1.0 + alpha1 * rtrs * rtrs)
    Q1 = 2.0 * gamma * rtrs * (beta1 +
                           rtrs * (beta2 +
                                   rtrs * (beta3 +
                                           rtrs * beta4)))
    G1 = Q0 * np.log(1.0 + 1.0 / Q1)
    return G1

def lda_x( n, x):
#    C0I, C1, CC1, CC2, IF2 = lda_constants()
    C1, gamma, alpha1, beta1, beta2, beta3, beta4 = optimization_constants(x)

    C0I = 0.238732414637843
    #C1 = -0.45816529328314287
    rs = (C0I / n) ** (1 / 3.)
    ex = C1 / rs
    return n*ex
    #e[:] += n * ex

def lda_c( n, x):
    #C0I, C1, CC1, CC2, IF2 = lda_constants()
    C1, gamma, alpha1, beta1, beta2, beta3, beta4 = optimization_constants(x)

    C0I = 0.238732414637843
    #C1 = -0.45816529328314287
    rs = (C0I / n) ** (1 / 3.)
    ec = G(rs ** 0.5, gamma, alpha1, beta1, beta2, beta3, beta4)
    return n*ec
    #e[:] += n * ec

def predict_LDA(n,LDA_x):

    n = np.asarray(n)

    return lda_x(n,LDA_x) + lda_c(n,LDA_x)

def predict_LDA_residual(n,LDA_x,X,NN_model):

    n = np.asarray(n)

    return lda_x(n,LDA_x) + lda_c(n,LDA_x) + ((NN_model.predict(X*1e6))/1e6)

def save_resulting_figure(n,LDA_x,X,NN_model,y,loss,loss_result):

    dens = n
    predict_y = predict_LDA_residual(n,LDA_x,X,NN_model)

    LDA_predict_y = predict_LDA(n,LDA_x)

    error = y - predict_y

    log_dens = np.log10(n)

    log_predict_y = np.log10(np.multiply(-1.,predict_y))
    log_y = np.log10(np.multiply(-1.,y))

    log_a_error = np.log10(np.abs(error))



    fig, axes = plt.subplots(2, 2, figsize=(100,100))
    ((ax1, ax2),(ax3,ax4)) = axes
    ax1.scatter(dens, y,            c= 'red',  lw = 0,label='original',alpha=1.0)
    ax1.scatter(dens, predict_y,    c= 'blue',  lw = 0,label='predict',alpha=1.0)
    ax1.scatter(dens, error,            c= 'yellow',  lw = 0,label='error',alpha=1.0)
    legend = ax1.legend(loc="best", shadow=False, scatterpoints=1, fontsize=80, markerscale=10)
    ax1.tick_params(labelsize=80)
    ax1.set_xlabel('density', fontsize=100)
    ax1.set_ylabel('energy density', fontsize=100)

    ax2.scatter(log_dens, y,            c= 'red',  lw = 0,label='original',alpha=1.0)
    ax2.scatter(log_dens, predict_y,    c= 'blue',  lw = 0,label='predict',alpha=1.0)
    ax2.scatter(log_dens, error,            c= 'yellow',  lw = 0,label='error',alpha=1.0)
    legend = ax2.legend(loc="best", shadow=False, scatterpoints=1, fontsize=80, markerscale=10)
    ax2.tick_params(labelsize=80)
    ax2.set_xlabel('log10 density', fontsize=100)
    ax2.set_ylabel('energy density', fontsize=100)

    ax3.scatter(dens, log_y,            c= 'red',  lw = 0,label='original',alpha=1.0)
    try:
        ax3.scatter(dens, log_predict_y,    c= 'blue',  lw = 0,label='predict',alpha=1.0)
    except:
        pass
    legend = ax3.legend(loc="best", shadow=False, scatterpoints=1, fontsize=80, markerscale=10)
    ax3.tick_params(labelsize=80)
    ax3.set_xlabel('density', fontsize=100)
    ax3.set_ylabel('log10 negative energy density', fontsize=100)

    ax4.scatter(log_dens, log_y,            c= 'red',  lw = 0,label='original',alpha=1.0)
    try:
        ax4.scatter(log_dens, log_predict_y,    c= 'blue',  lw = 0,label='predict',alpha=1.0)
    except:
        pass
    legend = ax4.legend(loc="best", shadow=False, scatterpoints=1, fontsize=80, markerscale=10)
    ax4.tick_params(labelsize=80)
    ax4.set_xlabel('log10 density', fontsize=100)
    ax4.set_ylabel('log10 negative energy density', fontsize=100)

    plt.savefig('result_plot_{}_{}.png'.format(loss,loss_result))


    fig, axes = plt.subplots(2, 2, figsize=(100,100))
    ((ax1, ax2),(ax3,ax4)) = axes
    ax1.scatter(dens, error,            c= 'red',  lw = 0,label='error',alpha=1.0)
    legend = ax1.legend(loc="best", shadow=False, scatterpoints=1, fontsize=80, markerscale=10)
    ax1.tick_params(labelsize=80)
    ax1.set_xlabel('density', fontsize=100)
    ax1.set_ylabel('error', fontsize=100)

    ax2.scatter(log_dens, error,            c= 'red',  lw = 0,label='error',alpha=1.0)
    legend = ax2.legend(loc="best", shadow=False, scatterpoints=1, fontsize=80, markerscale=10)
    ax2.tick_params(labelsize=80)
    ax2.set_xlabel('log10 density', fontsize=100)
    ax2.set_ylabel('error', fontsize=100)

    ax3.scatter(dens, log_a_error,            c= 'red',  lw = 0,label='absolute error',alpha=1.0)
    legend = ax3.legend(loc="best", shadow=False, scatterpoints=1, fontsize=80, markerscale=10)
    ax3.tick_params(labelsize=80)
    ax3.set_xlabel('density', fontsize=100)
    ax3.set_ylabel('log10 absolute error', fontsize=100)

    ax4.scatter(log_dens, log_a_error,            c= 'red',  lw = 0,label='absolute error',alpha=1.0)
    legend = ax4.legend(loc="best", shadow=False, scatterpoints=1, fontsize=80, markerscale=10)
    ax4.tick_params(labelsize=80)
    ax4.set_xlabel('log10 density', fontsize=100)
    ax4.set_ylabel('log10 absolute error', fontsize=100)

    plt.savefig('error_plot_{}_{}.png'.format(loss,loss_result))


    return

def get_training_data_entry(entry, index_list):
    result = []
    for index in index_list:
        result.append(entry[index])
    return result




def get_training_data(dataset_name,setup, dataset_setup):

    os.chdir(setup["data_dir"])
    overall = pickle.load(open(setup["training_data_filename"],'rb'))

    descriptor_index_list = dataset_setup[dataset_name]

    X_train = []
    y_train = []
    dens = []

    for entry in overall:
#        if entry[0] >= lower and entry[0] <= upper:
        #X_train.append(list(entry[1:]))
        X_train.append(get_training_data_entry(entry, descriptor_index_list))
        dens.append(entry[1])
        y_train.append(entry[0])
    
    
    X_train = (np.asarray(X_train))
    y_train = np.asarray(y_train).reshape((len(y_train),1))
    dens = np.asarray(dens).reshape((len(dens),1))
    
    return X_train, y_train, dens


def fit_model(setup,LDA_result, dens, X_train, residual, y, loss, tol, slowdown_factor, early_stop_trials):

    NN_model,loss_result = fit_with_KerasNN(setup,X_train * 1e6, residual * 1e6, loss, tol, slowdown_factor, early_stop_trials)
    save_resulting_figure(dens,LDA_result.x,X_train,NN_model,y,loss,loss_result)

    return NN_model


def process_dataset(setup_filename, dataset_setup_filename, dataset_name, fit_setup_filename):


    
    #training_data_filename = sys.argv[5]
    #test_data_filename = sys.argv[6]

    with open(setup_filename) as f:
        setup = json.load(f)

    with open(dataset_setup_filename) as f:
        dataset_setup = json.load(f)

    with open(fit_setup_filename) as f:
        fit_setup = json.load(f)

    K.set_floatx('float64')
    K.floatx()

    h = float(setup['grid_spacing'])
    L = float(setup['box_dimension'])
    N = int(setup['number_segment_per_side'])
    dir_name = "{}_{}_{}".format(str(L).replace('.','-'),str(h).replace('.','-'),N)

    data_dir = os.getcwd() + '/' + dir_name + '/epxc_MCSH_2_0-20_real_real/model_test/'

    setup["data_dir"] = data_dir

    working_dir = os.getcwd() + '/' + dir_name + '/epxc_MCSH_2_0-20_real_real/model_test/' + dataset_name

    setup["working_dir"] = working_dir

    model_save_dir = working_dir + "/" + "NN_LDA_residual_1M_{}_{}_{}".format(setup["NN_setup"]["number_neuron_per_layer"], setup["NN_setup"]["number_layers"], setup["NN_setup"]["activation"])
   
    setup["model_save_dir"] = model_save_dir

    
    
    X_train,y, dens = get_training_data(dataset_name,setup, dataset_setup)
   
    if os.path.isdir(model_save_dir) == False:
        os.makedirs(model_save_dir)

    os.chdir(data_dir)
    #residual,li_model = fit_with_Linear(dens,y)

    residual, LDA_result = fit_with_LDA(dens,y)
    setup['LDA_model'] = LDA_result

    os.chdir(model_save_dir)

    for fit_setup in fit_setup['fit_setup']:
        loss = fit_setup['loss']
        slowdown_factor = fit_setup['slowdown']
        early_stop_trials = fit_setup['early_stop']
        tol = fit_setup['tol']
        fit_model(setup,LDA_result, dens, X_train, residual, y, loss, tol, slowdown_factor, early_stop_trials)
    #NN_model = fit_with_KerasNN(X_train * 1e6, residual * 1e6, loss, tol, slowdown_factor, early_stop_trials)
    #save_resulting_figure(dens,result.x,X_train,NN_model,y)


if __name__ == "__main__":
    
    setup_filename = sys.argv[1]
    dataset_setup_filename = sys.argv[2]
    #dataset_name = sys.argv[3]
    fit_setup_filename = sys.argv[3]

    dataset_list = [ "epxc_LDA_real_real","epxc_MCSH_0_0-02_real_real","epxc_MCSH_0_0-04_real_real","epxc_MCSH_0_0-08_real_real",\
    "epxc_MCSH_0_0-20_real_real","epxc_MCSH_1_0-02_real_real","epxc_MCSH_1_0-04_real_real","epxc_MCSH_1_0-08_real_real","epxc_MCSH_1_0-20_real_real",\
    "epxc_MCSH_2_0-02_real_real","epxc_MCSH_2_0-04_real_real","epxc_MCSH_2_0-08_real_real","epxc_MCSH_2_0-20_real_real"]
    cwd = os.getcwd()

    for dataset_name in dataset_list:
        os.chdir(cwd)
        process_dataset(setup_filename, dataset_setup_filename, dataset_name, fit_setup_filename)
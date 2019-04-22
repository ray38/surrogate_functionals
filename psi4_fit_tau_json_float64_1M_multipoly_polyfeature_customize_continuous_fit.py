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

#from sklearn.decomposition import RandomizedPCA, PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def fit_poly_model(X,y,degree,filename, fit_intercept = True, log_filename = "fit.log"):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    poly_model = LinearRegression(fit_intercept = fit_intercept, n_jobs = -1).fit(X_poly, y)
    #y_predict = linear_model.predict(X_poly)


    #poly_model = make_pipeline(PolynomialFeatures(degree), Ridge())
    #poly_model.fit(X, y)
    pickle.dump(poly_model, open(filename, 'w'))

    log(log_filename, "number of points fit: {}".format(len(y)))
    log(log_filename, "fit score: {}".format(poly_model.score(X_poly, y)))
    return poly_model
def predict_poly_model(X,poly_model,degree):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    y_predict = poly_model.predict(X_poly)
    return y_predict

def fit_pca(data,filename):
    print "start fitting pca"
#    pca = RandomizedPCA()
    pca = PCA(svd_solver='randomized')
    X_pca = pca.fit_transform(data)
    pickle.dump(pca, open(filename, 'w'))
    return pca

def transform_pca(pca_model,data,n_components):
    temp = pca_model.transform(data)
    return temp[:,:n_components]

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

def fit_with_KerasNN(X, y, loss, tol, slowdown_factor, early_stop_trials):

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
    
    history_callback_kickoff = model.fit(X, y, nb_epoch=1, batch_size=100000, shuffle=True)
    est_start = time.time()
    history_callback = model.fit(X, y, nb_epoch=1, batch_size=100000)
    est_epoch_time = time.time()-est_start
    if est_epoch_time >= 30.:
        num_epoch = 1
    else:
        num_epoch = int(math.floor(30./est_epoch_time))
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
        history_callback = model.fit(X, y, nb_epoch=num_epoch, batch_size=100000, shuffle=True)
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




def predict(X,NN_model):

    #return NN_model.predict(X)
    return NN_model.predict(X*1e6)/1e6

def save_resulting_figure(n,X,NN_model,y,loss,loss_result):

    dens = n
    predict_y = predict(X,NN_model)


    error = y - predict_y

    log_dens = np.log10(n)

    log_predict_y = np.log10(predict_y)
    log_y = np.log10(y)

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


def save_data_figure(n,y_plot, filename = "starting_data_plot.png"):

    dens = n


    log_dens = np.log10(n)

    log_y = np.log10(y_plot)



    fig, axes = plt.subplots(2, 2, figsize=(100,100))
    ((ax1, ax2),(ax3,ax4)) = axes
    ax1.scatter(dens, y_plot,            c= 'red',  lw = 0,label='original',alpha=1.0)
    legend = ax1.legend(loc="best", shadow=False, scatterpoints=1, fontsize=80, markerscale=10)
    ax1.tick_params(labelsize=80)
    ax1.set_xlabel('density', fontsize=100)
    ax1.set_ylabel('energy density', fontsize=100)

    ax2.scatter(log_dens, y_plot,            c= 'red',  lw = 0,label='original',alpha=1.0)
    legend = ax2.legend(loc="best", shadow=False, scatterpoints=1, fontsize=80, markerscale=10)
    ax2.tick_params(labelsize=80)
    ax2.set_xlabel('log10 density', fontsize=100)
    ax2.set_ylabel('energy density', fontsize=100)

    ax3.scatter(dens, log_y,            c= 'red',  lw = 0,label='original',alpha=1.0)
    legend = ax3.legend(loc="best", shadow=False, scatterpoints=1, fontsize=80, markerscale=10)
    ax3.tick_params(labelsize=80)
    ax3.set_xlabel('density', fontsize=100)
    ax3.set_ylabel('log10 negative energy density', fontsize=100)

    ax4.scatter(log_dens, log_y,            c= 'red',  lw = 0,label='original',alpha=1.0)
    legend = ax4.legend(loc="best", shadow=False, scatterpoints=1, fontsize=80, markerscale=10)
    ax4.tick_params(labelsize=80)
    ax4.set_xlabel('log10 density', fontsize=100)
    ax4.set_ylabel('log10 negative energy density', fontsize=100)

    #plt.savefig('starting_data_plot.png')
    plt.savefig(filename)

    return

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
    #molecule_random_data = []

    os.chdir(temp_cwd)

    return molecule_subsampled_data, molecule_random_data



def get_training_data(dataset_name,setup):

    data_dir_name = setup["working_dir"] + "/data/*/" 
    data_paths = glob(data_dir_name)
    print data_paths


    overall_subsampled_data = []
    overall_random_data = []
    num_samples = len(data_paths)
    num_random_per_molecule = int(math.ceil(float(setup["random_pick"])/float(num_samples)))
    for directory in data_paths:
        temp_molecule_subsampled_data, temp_molecule_random_data = read_data_from_one_dir(directory)
        overall_subsampled_data += temp_molecule_subsampled_data
        overall_random_data += random_subsampling(temp_molecule_random_data, num_random_per_molecule)



    list_subsample = setup["subsample_feature_list"]
    temp_list_subsample = setup["subsample_feature_list"]
    if temp_list_subsample == []:
        for m in range(len(overall_subsampled_data[0])):
            temp_list_subsample.append(m)

    #if len(temp_list_subsample) <= 10:
    #    overall_subsampled_data = subsampling_system(overall_subsampled_data, list_desc = list_subsample, cutoff_sig = float(setup["subsample_cutoff_sig"]), rate = float(setup["subsample_rate"]))
    #else:
    #    overall_subsampled_data = subsampling_system_with_PCA(overall_subsampled_data, list_desc = list_subsample, cutoff_sig = float(setup["subsample_cutoff_sig"]), rate = float(setup["subsample_rate"]),start_trial_component = 9)

    #pickle.dump( overall_subsampled_data, open( "subsampled_data.p", "w" ) )
    #pickle.dump( overall_random_data, open( "random_data.p", "w" ) )

    overall = overall_random_data + overall_subsampled_data
    #overall = overall_subsampled_data



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
    
    return X_train, y_train, dens


def fit_model(dens, X_train, y, residual, loss, tol, slowdown_factor, early_stop_trials):

    NN_model,loss_result = fit_with_KerasNN(X_train * 1e6, residual * 1e6, loss, tol, slowdown_factor, early_stop_trials)
    save_resulting_figure(dens,X_train,NN_model,residual,loss,loss_result)

    return NN_model


if __name__ == "__main__":


    setup_filename = sys.argv[1]
    dataset_name = sys.argv[2]
    fit_setup_filename = sys.argv[3]
    polynomial_order = int(sys.argv[4])

    intercept = int(sys.argv[5])

    if intercept == 0:
        fit_intercept = False
    else:
        fit_intercept = True

    with open(setup_filename) as f:
        setup = json.load(f)

    with open(fit_setup_filename) as f:
        fit_setup = json.load(f)

    K.set_floatx('float64')
    K.floatx()

    h = float(setup['grid_spacing'])
    L = float(setup['box_dimension'])
    N = int(setup['number_segment_per_side'])
    dir_name = "{}_{}_{}".format(str(L).replace('.','-'),str(h).replace('.','-'),N)

    working_dir = os.getcwd() + '/' + dir_name + '/' + dataset_name

    setup["working_dir"] = working_dir

    model_save_dir = working_dir + "/" + "NN_1M_{}_{}_{}_multipoly_polyfeature_{}".format(setup["NN_setup"]["number_neuron_per_layer"], setup["NN_setup"]["number_layers"], setup["NN_setup"]["activation"],polynomial_order)
   
    setup["model_save_dir"] = model_save_dir

    
    
    X_train,y, dens = get_training_data(dataset_name,setup)
   
    if os.path.isdir(model_save_dir) == False:
        os.makedirs(model_save_dir)

    os.chdir(model_save_dir)
    stdandard_scaler_filename = "standard_scaler.sav"
    poly_model_filename = "poly_{}_{}_model.sav".format(polynomial_order, fit_intercept)

    try:
        standard_scaler = pickle.load(open(stdandard_scaler_filename, 'r'))
        X_train = standard_scaler.transform(X_train)
        try:
            poly_model = pickle.load(open(poly_model_filename, 'r'))
            y_poly = predict_poly_model(X_train, poly_model,polynomial_order)
            #y_poly = poly_model.predict(X_train)
        except:
            poly_model = fit_poly_model(X_train,y,polynomial_order,poly_model_filename, fit_intercept = fit_intercept)
            y_poly = predict_poly_model(X_train, poly_model,polynomial_order)



    except:
        standard_scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        X_train = standard_scaler.fit_transform(X_train)
        pickle.dump(standard_scaler, open(stdandard_scaler_filename, 'w'))
        poly_model = fit_poly_model(X_train,y,polynomial_order,poly_model_filename, fit_intercept = fit_intercept)
        y_poly = predict_poly_model(X_train, poly_model,polynomial_order)


    
    poly = PolynomialFeatures(polynomial_order)
    X_poly = poly.fit_transform(X_train)


    for fit_setup in fit_setup['fit_setup']:
        loss = fit_setup['loss']
        slowdown_factor = fit_setup['slowdown']
        early_stop_trials = fit_setup['early_stop']
        tol = fit_setup['tol']
        fit_model(dens, X_poly, y, y - y_poly, loss, tol, slowdown_factor, early_stop_trials)




    

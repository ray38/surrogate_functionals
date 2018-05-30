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
import scipy

import itertools
import multiprocessing

try: import cPickle as pickle
except: import pickle
import matplotlib.pyplot as plt
from subsampling import subsampling_system,random_subsampling,subsampling_system_with_PCA

from sklearn.decomposition import RandomizedPCA, PCA
from sklearn.cross_decomposition import PLSRegression
import pandas as pd
import seaborn as sns

from sklearn import manifold


def PCA_analysis(data, n_components = 2):
    pca = RandomizedPCA(n_components = n_components )
    X_pca = pca.fit_transform(data)
#    plot(X_pca)
    return X_pca

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
    log(text_filename, str(res.x))
    log(text_filename, '\nMSE: {}'.format(np.mean(np.square(lda_x(density,res.x) + lda_c(density,res.x) - energy))))
    log(text_filename, '\nMAE: {}'.format(np.mean(np.abs(lda_x(density,res.x) + lda_c(density,res.x) - energy))))
    log(text_filename, '\nMSD: {}'.format(np.mean(lda_x(density,res.x) + lda_c(density,res.x) - energy)))


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

    #try:
    #    molecule_random_data = pickle.load(open(random_filename,'rb'))
    #    print "read random data"
    #except:
    #    molecule_random_data = []
    molecule_random_data = []

    os.chdir(temp_cwd)

    return molecule_subsampled_data, molecule_random_data



def get_training_data(dataset_name,setup):

    colormap = {"C2H2":0,
                "C2H4":1,
                "C2H6":2,
                "CH3OH":3,
                "CH4":4,
                "CO":5,
                "CO2":6,
                "H2":7,
                "H2O":8,
                "HCN":9,
                "HNC":10,
                "N2":11,
                "N2O":12,
                "NH3":13,
                "O3":14}

    data_dir_name = setup["working_dir"] + "/data/*/" 
    data_paths = glob(data_dir_name)
    print data_paths

    #data_paths = ["/gpfs/pace1/project/chbe-medford/medford-share/users/xlei38/psi4_feature_picked_database/B3LYP_float64_test/10-0_0-02_5/epxc_mGGA_allrange_real_real_numerical/data/C2H6/", "/gpfs/pace1/project/chbe-medford/medford-share/users/xlei38/psi4_feature_picked_database/B3LYP_float64_test/10-0_0-02_5/epxc_mGGA_allrange_real_real_numerical/data/NH3/"]
    #data_paths = [  "/media/ray/Data_Archive/psi4_feature_picked_database/B3LYP_float64_test/10-0_0-02_5/epxc_mGGA_allrange_real_real_numerical/data/C2H6/",\
    #                "/media/ray/Data_Archive/psi4_feature_picked_database/B3LYP_float64_test/10-0_0-02_5/epxc_mGGA_allrange_real_real_numerical/data/NH3/"]

    overall_subsampled_data = []
    overall_random_data = []
    overall_molecule_name_list_subsampled = []
    overall_molecule_name_list_random = []
    overall_molecule_label_list_subsampled = []
    overall_molecule_label_list_random = []
    num_samples = len(data_paths)
    num_random_per_molecule = int(math.ceil(float(setup["random_pick"])/float(num_samples)))
    for directory in data_paths:
        molecule_name = directory.split('/')[-2]

        print molecule_name
        temp_molecule_subsampled_data, temp_molecule_random_data = read_data_from_one_dir(directory)
        overall_subsampled_data += temp_molecule_subsampled_data
        overall_molecule_name_list_subsampled += [colormap[molecule_name]] * len(temp_molecule_subsampled_data)
        overall_molecule_label_list_subsampled += [molecule_name] * len(temp_molecule_subsampled_data)

        #temp_random_sampled_random_data = random_subsampling(temp_molecule_random_data, num_random_per_molecule)
        #overall_random_data += temp_random_sampled_random_data
        #overall_molecule_name_list_random += [colormap[molecule_name]] * len(temp_random_sampled_random_data)
        #overall_molecule_label_list_random += [molecule_name] * len(temp_random_sampled_random_data)



    overall = overall_random_data + overall_subsampled_data
    overall_molecule_name = overall_molecule_name_list_random + overall_molecule_name_list_subsampled
    overall_molecule_label = overall_molecule_label_list_random + overall_molecule_label_list_subsampled
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
    
    return X_train, y_train, dens, overall_molecule_name, overall_molecule_label


def fit_model(LDA_result, dens, X_train, residual, loss, tol, slowdown_factor, early_stop_trials):

    NN_model,loss_result = fit_with_KerasNN(X_train * 1e6, residual * 1e6, loss, tol, slowdown_factor, early_stop_trials)
    save_resulting_figure(dens,result.x,X_train,NN_model,y,loss,loss_result)

    return NN_model

def fit_pca(data,filename,n_components = 5):
    print "start fitting pca"
    print data.shape
    pca = RandomizedPCA(n_components = n_components )
    X_pca = pca.fit_transform(data)
    pickle.dump(pca, open(filename, 'wb'))
    print X_pca.shape
    return X_pca, pca

def fit_kernel_pca(data,filename,kernel,n_components = 5):
    print "start fitting pca"
    print data.shape
    kpca = KernelPCA(n_components = n_components, kernel= kernel, fit_inverse_transform=True)

    X_kpca = kpca.fit_transform(data)
    pickle.dump(kpca, open(filename, 'wb'))
    print X_kpca.shape
    return X_kpca, kpca


def fit_pls(data,filename,n_components = 5):
    print "start fitting pls"
    pls = PLSRegression(n_components = n_components)
    X_pls = pls.fit_transform(data)
    pickle.dump(pls, open(filename, 'wb'))
    return X_pls, pls

def fit_manifold(data,filename,method,n_neighbors = 10, n_components = 2):
    print "start fitting manifold"
    print data.shape
    model = manifold.LocallyLinearEmbedding(n_neighbors, n_components,method=method)
    X_transform = model.fit_transform(data)
    pickle.dump(model, open(filename, 'wb'))
    print X_transform.shape
    return X_transform, model

def plot_result(data, molecule_name, molecule_label, filename,figure_size, edge=(0,0,0,0)):
    x_low, x_high, y_low, y_high = edge 
    print "start plotting"
    result = {}
    result["PC1"] = data[:,0]
    result["PC2"] = data[:,1]
    result["molecule_name"] = molecule_name
    result["molecule_label"] = molecule_label

    data = pd.DataFrame(data=result)
    # Use the 'hue' argument to provide a factor variable

    #plt.figure(figsize=(figure_size,figure_size))
    
    sns.set(style="whitegrid", palette="pastel", color_codes=True)
    sns.lmplot( x="PC1", y="PC2", data=data, fit_reg=False, hue='molecule_label', legend=False,size=figure_size)
     
    # Move the legend to an empty part of the plot
    plt.legend(loc='lower right')
    plt.savefig(filename)


    groups = data.groupby('molecule_label')
    fig = plt.figure(figsize=(figure_size,figure_size))
    ax3D = fig.add_subplot(111, projection='3d')
    for name, group in groups:
        ax3D.scatter(group.PC1, group.molecule_name, group.PC2, marker='o', label=name,cmap=cm.rainbow, linewidths=0)
    ax3D.scatter(np.asarray([x_low,x_high]), np.asarray([0,0]), np.asarray([y_low,y_high]), linewidths=0, marker='x',c='k')
    ax3D.legend()

    plt.savefig("3D_1_" + filename)


    fig = plt.figure(figsize=(figure_size,figure_size))
    ax3D = fig.add_subplot(111, projection='3d')
    for name, group in groups:
        ax3D.scatter(group.PC1, group.molecule_name, group.PC2, marker='o', label=name,cmap=cm.rainbow, linewidths=0)
    ax3D.scatter(np.asarray([x_low,x_high])*2, np.asarray([0,0]), np.asarray([y_low,y_high])*2, linewidths=0, marker='x',c='k')
    ax3D.legend()

    plt.savefig("3D_2_" + filename)



    fig = plt.figure(figsize=(figure_size,figure_size))
    ax3D = fig.add_subplot(111, projection='3d')
    for name, group in groups:
        ax3D.scatter(group.PC1, group.molecule_name, group.PC2, marker='o', label=name,cmap=cm.rainbow, linewidths=0)
    ax3D.scatter(np.asarray([x_low,x_high])*5, np.asarray([0,0]), np.asarray([y_low,y_high])*5, linewidths=0, marker='x',c='k')
    ax3D.legend()

    plt.savefig("3D_5_" + filename)



    fig = plt.figure(figsize=(figure_size,figure_size))
    ax3D = fig.add_subplot(111, projection='3d')
    for name, group in groups:
        ax3D.scatter(group.PC1, group.molecule_name, group.PC2, marker='o', label=name,cmap=cm.rainbow, linewidths=0)
    ax3D.scatter(np.asarray([x_low,x_high])*10, np.asarray([0,0]), np.asarray([y_low,y_high])*10, linewidths=0, marker='x',c='k')
    ax3D.legend()

    plt.savefig("3D_10_" + filename)

    return


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

    model_save_dir = working_dir + "/" + "PCA_result_{}_no_target".format(setup["random_pick"])
   
    setup["model_save_dir"] = model_save_dir

    
    
    X_train,y, dens, molecule_name, molecule_label = get_training_data(dataset_name,setup)

    print np.isnan(X_train.any())
    print np.isfinite(X_train.all())
   
    if os.path.isdir(model_save_dir) == False:
        os.makedirs(model_save_dir)

    os.chdir(model_save_dir)

    #try:
    #X_pca, pca = fit_pca(X_train.copy(),'pca_model_{}.sav'.format(dataset_name),n_components = 3)
    #plot_result(X_pca, molecule_name, molecule_label, "PCA_result_plot_{}_{}.png".format(dataset_name,10),10, edge=(-100000,600000,-6000,6000))
    #plot_result(X_pca, molecule_name, molecule_label, "PCA_result_plot_{}_{}.png".format(dataset_name,20),20, edge=(-100000,600000,-6000,6000))
    #except:
    #    pass


    #try:
    X_pls, pls = fit_pls(X_train.copy(),'pls_model_{}.sav'.format(dataset_name),n_components = 3)
    plot_result(X_pls, molecule_name, molecule_label, "PLS_result_plot_{}_{}.png".format(dataset_name,10),10)
    plot_result(X_pls, molecule_name, molecule_label, "PLS_result_plot_{}_{}.png".format(dataset_name,20),20)
    #except:
    #    pass


    for kernel in ["poly","rbf","sigmoid"]:

        #try:
        X_kpca, kpca = fit_kernel_pca(X_train.copy(),'kpca_model_{}_{}.sav'.format(dataset_name,kernel),kernel,n_components = 3)
        plot_result(X_kpca, molecule_name, molecule_label, "kPCA_result_plot_{}_{}_{}.png".format(dataset_name,kernel,10),10)
        plot_result(X_kpca, molecule_name, molecule_label, "kPCA_result_plot_{}_{}_{}.png".format(dataset_name,kernel,20),20)
        #except:
        #    pass

    for method in ['standard', 'ltsa', 'hessian', 'modified']:
        #try:
        X_transform, model = fit_manifold(X_train.copy(),'manifold_model_{}_{}.sav'.format(dataset_name,method),method,n_components = 3)
        plot_result(X_transform, molecule_name, molecule_label, "manifold_result_plot_{}_{}_{}.png".format(dataset_name,method,10),10)
        plot_result(X_transform, molecule_name, molecule_label, "manifold_result_plot_{}_{}_{}.png".format(dataset_name,method,20),20)
        #except:
        #    pass


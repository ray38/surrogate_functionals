

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
import scipy

import itertools
import multiprocessing

try: import cPickle as pickle
except: import pickle
#import matplotlib.pyplot as plt
#from subsampling import subsampling_system,random_subsampling,subsampling_system_with_PCA

def get_x0():
    x = [ -0.35598593,   0.07902084,  -1.43830151,   5.79974916, -14.19322583 , 9.80903158,  -0.88783471]
    return x

def optimization_constants(x):

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
    print rtrs
    Q0 = -2.0 * gamma * (1.0 + alpha1 * rtrs * rtrs)
    Q1 = 2.0 * gamma * rtrs * (beta1 +
                           rtrs * (beta2 +
                                   rtrs * (beta3 +
                                           rtrs * beta4)))
    print Q1
    print Q0
    G1 = Q0 * np.log(1.0 + 1.0 / Q1)
    print 1.0 + 1.0 / Q1
    return G1

def lda_x( n, x):
    C1, gamma, alpha1, beta1, beta2, beta3, beta4 = optimization_constants(x)

    C0I = 0.238732414637843
    rs = (C0I / n) ** (1 / 3.)
    ex = C1 / rs
    return n*ex
    #e[:] += n * ex

def lda_c( n, x):
    C1, gamma, alpha1, beta1, beta2, beta3, beta4 = optimization_constants(x)

    C0I = 0.238732414637843
    rs = (C0I / n) ** (1 / 3.)
    ec = G(rs ** 0.5, gamma, alpha1, beta1, beta2, beta3, beta4)
    return n*ec

def predict(n,x):

    n = 3.39*np.asarray(n)

    return lda_x(n,x) + lda_c(n,x)


x0 = get_x0()
n=[1e-20,1e-19,1e-18,1e-17,1e-16,1e-15, 1e-14, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2,1e3,1e4,1e5]
#n = [1e-7]
print predict(n,x0)
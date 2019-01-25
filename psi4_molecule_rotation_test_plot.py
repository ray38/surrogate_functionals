import os
import h5py
import psi4
import sys
import json
import collections
import numpy as np
import math
import multiprocessing
import json
import itertools
import pandas as pd
import seaborn as sns
import copy

import matplotlib
#matplotlib.use('agg')
matplotlib.pyplot.switch_backend('agg')
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt

from convolutions import get_differenciation_conv, get_integration_stencil,get_auto_accuracy,get_fftconv_with_known_stencil_no_wrap,get_asym_integration_stencil,get_asym_integration_fftconv,get_asym_integral_fftconv_with_known_stencil
from convolutions import get_first_grad_stencil, get_second_grad_stencil, get_third_grad_stencil, get_harmonic_fftconv, calc_harmonic_stencil


def plot_result(data):
    plt.figure()
        
    sns.set(style="whitegrid", palette="pastel", color_codes=True)

    #ax = sns.violinplot(x = "Molecule",y="Value",row="Type", col="Property",data=data)
    
    ax = sns.factorplot(x = "Molecule",y="Value",row="Type", col="Property",data=data, kind="violin", split=True,sharey = False, size = 6, aspect = 1.5)

    plt.tight_layout()
    plt.savefig("Molecule_rotational_invariance_test1.png")
    plt.cla()
    plt.close()



    return

def plot_result1(d,property_list):
    if property_list == []:
        data = d
    else:
        print property_list
        data = d[d.Property.isin(property_list)]
    print data
    plt.figure()
        
    sns.set(style="whitegrid", palette="pastel", color_codes=True)
    sns.set(font_scale=1.9)

    #ax = sns.violinplot(x = "Molecule",y="Value",row="Type", col="Property",data=data)
    
    ax = sns.factorplot(x = "Molecule",y="Value",row="Type", col="Property",data=data, kind="violin", split=True,sharey = False, size = 6, aspect = 1.5)
    plt.xticks(rotation=‌​45)
    plt.tight_layout()
    plt.savefig("Molecule_rotational_invariance_test1.png")
    plt.cla()
    plt.close()



    return

def plot_result2(data):
    plt.figure()
        
    sns.set(style="whitegrid", palette="pastel", color_codes=True)

    #ax = sns.violinplot(x = "Molecule",y="Value",row="Type", col="Property",data=data)
    
    ax = sns.factorplot(x = "ID",y="Value",row="Property",hue="Label",data=data, kind="point", split=True,sharey = False, size = 6, aspect = 25.0)
    #plt.setp(ax.collections, sizes=[2])

    plt.tight_layout()
    plt.savefig("Molecule_rotational_invariance_test2.png")
    plt.cla()
    plt.close()



    return

if __name__ == "__main__":


    data_filename = sys.argv[1]
    choice = int(sys.argv[2])
    d = pd.read_pickle(data_filename)

    if choice == 1:
        plot_quantity = str(sys.argv[3])
        if plot_quantity == "original":
            property_list = ["Gamma","Gradient","exc", "tau"]
        if plot_quantity == "MCSH":
            r = float(sys.argv[4])
            property_list = ["MCSH 0,1 {}".format(r), "MCSH 1,1 {}".format(r), "MCSH 2,1 {}".format(r), "MCSH 2,2 {}".format(r), "MCSH 3,1 {}".format(r), "MCSH 3,2 {}".format(r), "MCSH 3,3 {}".format(r)]
        if plot_quantity == "all":
            property_list = []
        plot_result1(d,property_list)
    

    #if choice == 1:
    #	plot_result(data)

    elif choice ==2:
    	plot_result2(d)

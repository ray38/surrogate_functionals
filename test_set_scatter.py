# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 11:31:45 2017

@author: ray
"""

import cPickle as pickle

import sys
import os
import numpy as np
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
import seaborn

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

list_of_molecule_filename = sys.argv[1]
fit_type = sys.argv[2]


with open(list_of_molecule_filename) as f:
    molecule_names = f.readlines()
molecule_names = [x.strip() for x in molecule_names]


data = []
for molecule in molecule_names:
    filename = "{}_B3LYP_10-0_0-02_5_{}_test_error_for_scatter.p".format(molecule,fit_type)
    try:
        temp = pickle.load(open(filename,'rb'))
        data.extend(temp)
    except:
        pass
seaborn.set()
data = np.asarray(data)
predict_y_real = data[:,1]
y_real = data[:,0]
error_real = y_real-predict_y_real


predict_y_log = np.log10(np.multiply(-1.,predict_y_real))
y_log = np.log10(np.multiply(-1.,y_real))
error_log = y_log - predict_y_log

#predict_y_log = model.predict(X_train) + li_model.predict(dens)
#y_log = y
#error_log = y_log-predict_y_log
#
#
#
#predict_y_real = np.multiply(-1.,(np.power(10.,model.predict(X_train) + li_model.predict(dens) )))
#y_real = np.multiply(-1.,(np.power(10.,y)))
#error_real = y_real-predict_y_real



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

plt.savefig('test_set_error_hist_desc_{}.png'.format(fit_type))
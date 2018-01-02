# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 11:21:50 2017

@author: ray
"""

import numpy as np

def get_xyz(X0,Y0,Z0,x_inc,y_inc,z_inc,hx,hy,hz,i,j,k):

    x_start = X0 + float(i) * x_inc
    y_start = Y0 + float(j) * y_inc
    z_start = Z0 + float(k) * z_inc
    
    x_end = x_start + x_inc - hx
    y_end = y_start + y_inc - hy
    z_end = z_start + z_inc - hz
    
    xyz = []
    for hi, start,end in zip((hx, hy, hz),(x_start, y_start ,z_start),(x_end, y_end ,z_end)):
        n_i = ((end-start)/hi)+1.
        xi = np.linspace(start, end, int(round(n_i,0)))
        xyz.append(xi)

    out_shape = [len(xi) for xi in xyz]
    x, y, z = np.meshgrid(*xyz)
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()
    return x, y,z

L = 10.
h = 0.02
N = 5
X0 = Y0 = Z0 = -L/2.
x_inc = y_inc = z_inc = L/N
hx = hy = hz = h
i=0
j=0
k=0
x_plot,y_plot,z_plot = get_xyz(X0,Y0,Z0,x_inc,y_inc,z_inc,hx,hy,hz,i,j,k)
print x_plot
print y_plot
print z_plot
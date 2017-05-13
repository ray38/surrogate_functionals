# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 10:12:40 2017

@author: ray
"""

#path = "/home/ray/Documents/gpaw_test/gpawDFT"
#path_ase = "/home/ray/Documents/gpaw_test/ase-master-Jan-2017"
#path_utility = "/home/ray/Documents/Utilities"

import sys
#sys.path.insert(0,path)
#sys.path.insert(0,path_ase)
#sys.path.insert(0,path_utility)


#from convolutions import *
#from integration import *
#from getDescriptors import *

#from gpaw import GPAW, restart
import numpy as np
from operator import itemgetter

#from scipy.ndimage.filters import convolve
import pickle

import random
import pygal
import math


from sklearn.decomposition import PCA, IncrementalPCA
import matplotlib.pyplot as plt

from pygal.style import Style
from pygal.style import DefaultStyle

#from slidesPlot3D_test import *
from slidesPlot3D_noScatter_test import *
import colorsys
import math
import sys

#def get_enhancemend_func1(r, order_resolve)




def find_h_s_v_pc(full_desc_with_color_entry):
    pc_x = full_desc_with_color_entry[-5]
    pc_y = full_desc_with_color_entry[-4]
    h = full_desc_with_color_entry[-3]
    s = full_desc_with_color_entry[-2]
    v = full_desc_with_color_entry[-1]
    return h,s,v, pc_x, pc_y


descriptor_filename = sys.argv[1]
try:
    mode = sys.argv[2]
    order_resolve = float(sys.argv[3])
except:
    mode = 'no'
#    order_resolve = 10.

mode_list = ['no','sig']
if mode not in mode_list:
    raise NotImplementedError



#dict_descriptors = pickle.load( open( descriptor_filename, "rb" ) )
discriptors = pickle.load( open( descriptor_filename, "rb" ) )

#discriptors = dict_descriptors[dict_descriptors.keys()[0]]
plot_nx = discriptors[0][3]
plot_ny = discriptors[0][4]
plot_nz = discriptors[0][5]
    
x = np.linspace(-4,4,plot_nx)
y = np.linspace(-4,4,plot_ny)
z = np.linspace(0,8,plot_nz)

pc_x_li = []
pc_y_li = []
pc_u_li = []

xx,yy,zz = meshgrid3(x,y,z)

result = np.zeros((plot_nx,plot_ny,plot_nz,3)).tolist()

for entry in discriptors:
    h, s, v, pc_x, pc_y = find_h_s_v_pc(entry)
    if h < 0.:
        h+=2*math.pi
    if h > 2*math.pi:
        h-= 2*math.pi
    if mode == 'no':
        R,G,B = colorsys.hsv_to_rgb(h/6.,s,v)
    elif mode == 'sig':
        R,G,B = colorsys.hsv_to_rgb(h/6.,1/(1+math.exp(-order_resolve*s)),v)
    
#    R,G,B = colorsys.hsv_to_rgb(h,1/(1+math.exp(-order_resolve*s)),v) #1/(1+math.exp(-order_resolve*s))
    result[int(entry[0])][int(entry[1])][int(entry[2])] = [R,G,B]
    pc_x_li.append(pc_x)
    pc_y_li.append(pc_y)
    pc_u_li.append([R,G,B])

    

u = np.asarray(result)




 
s3 = slice3_noScatter(xx,yy,zz,u)
s3.xlabel('x',fontsize=18)
s3.ylabel('y',fontsize=18)
s3.zlabel('z',fontsize=18)
 
 

fig =plt.figure()
ax = fig.add_subplot(111)
ax.scatter(np.asarray(pc_x_li),np.asarray(pc_y_li),c = np.asarray(pc_u_li), lw = 0)
plt.savefig('temp_PCA_color_reference.png')

s3.show()
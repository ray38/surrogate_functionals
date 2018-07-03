import numpy as np
from convolutions import get_differenciation_conv, get_integration_stencil,get_auto_accuracy,get_fftconv_with_known_stencil_no_wrap,get_asym_integration_stencil,get_asym_integration_fftconv,get_asym_integral_fftconv_with_known_stencil
import itertools
from math import cos,sin,tan,acos,asin,pi

import sys
import matplotlib
#matplotlib.use('Agg') 
#import matplotlib.cm as cm
#from mpl_toolkits.mplot3d import Axes3D
#import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt


def f(x,y,z,sig_x,sig_y,sig_z,x0,y0,z0):
	return np.exp(-(np.square(x-x0)/(2*sig_x*sig_x)) -(np.square(y-y0)/(2*sig_y*sig_y)) -(np.square(z-z0)/(2*sig_z*sig_z))  )


def rotate_coord_mat(x,y,z,theta1,theta2,theta3):
	x_result = x
	y_result = y
	z_result = z

	x_result = x_result
	y_result = y_result*cos(theta1) - z_result*sin(theta1)
	z_result = y_result*sin(theta1) + z_result*cos(theta1)

	x_result = x_result*cos(theta2) + z_result*sin(theta2)
	y_result = y_result
	z_result = z_result*cos(theta2) - x_result*sin(theta2)

	x_result = x_result*cos(theta3) - y_result*sin(theta3)
	y_result = x_result*sin(theta3) + y_result*cos(theta3)
	z_result = z_result

	return x_result,y_result,z_result

def get_result(x,y,z,sig_x,sig_y,sig_z,x0,y0,z0, r, h, stencil, pad):

	n = f(x,y,z,sig_x,sig_y,sig_z,x0,y0,z0)
	
	#temp_stencil,temp_pad = get_integration_stencil(h, h, h, r, accuracy = get_auto_accuracy(h,h,h, r))
	temp,_ = get_fftconv_with_known_stencil_no_wrap(n,h,h,h,r,stencil,pad)
	print temp[50,50,50]

	return temp[50,50,50]


r = float(sys.argv[1])
num_random = int(sys.argv[2])

nx, ny, nz = (101,101,101)
h = 0.02


xv = np.linspace(-1.0,1.0,nx)
print xv
yv = np.linspace(-1.0,1.0,ny)
zv = np.linspace(-1.0,1.0,nz)

x, y, z = np.meshgrid(xv, yv, zv)

x0,y0,z0 = (0.0, 0.0, 0.0)
sig_x = np.random.uniform(0, 1.0)
sig_y = np.random.uniform(0, 1.0)
sig_z = np.random.uniform(0, 1.0)

stencil,pad = get_integration_stencil(h, h, h, r, accuracy = get_auto_accuracy(h,h,h, r))
print stencil

truth = get_result(x,y,z,sig_x,sig_y,sig_z,x0,y0,z0, h, r, stencil, pad)


result = []

for i in range(num_random):

	theta1 = theta2 = theta3 = 0.0
	theta1 = np.random.uniform(0.0, 2.0*pi)
	theta2 = np.random.uniform(0.0, 2.0*pi)
	theta3 = np.random.uniform(0.0, 2.0*pi)
	print i, theta1, theta2, theta3

	x_temp, y_temp, z_temp = rotate_coord_mat(x,y,z,theta1,theta2,theta3)
	error = get_result(x_temp, y_temp, z_temp,sig_x,sig_y,sig_z,x0,y0,z0, h, r, stencil, pad) - truth

	result.append(error/truth)

fig,ax = plt.subplots(figsize=(10,5))
plt.plot(np.arange(1,num_random+1),result, linewidth=5.0)
plt.show()



#get function matrix

#convolve

#get middle (50,50,50)
import numpy as np
from convolutions import get_differenciation_conv, get_integration_stencil,get_auto_accuracy,get_fftconv_with_known_stencil_no_wrap,get_asym_integration_stencil,get_asym_integration_fftconv,get_asym_integral_fftconv_with_known_stencil
import itertools
from math import cos,sin,tan,acos,asin,pi

import sys
import matplotlib
#matplotlib.use('Agg') 
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt

def generate(dim):
    """Generate a random rotation matrix.

    https://github.com/qobilidop/randrot

    Args:
        dim (int): The dimension of the matrix.

    Returns:
        np.matrix: A rotation matrix.

    Raises:
        ValueError: If `dim` is not 2 or 3.

    """
    if dim == 2:
        return generate_2d()
    elif dim == 3:
        return generate_3d()
    else:
        raise ValueError('Dimension {} is not supported. Use 2 or 3 instead.'
                         .format(dim))


def generate_2d():
    """Generate a 2D random rotation matrix.

    Returns:
        np.matrix: A 2D rotation matrix.

    """
    x = np.random.random()
    M = np.matrix([[np.cos(2 * np.pi * x), -np.sin(2 * np.pi * x)],
                   [np.sin(2 * np.pi * x), np.cos(2 * np.pi * x)]])
    return M


def generate_3d():
    """Generate a 3D random rotation matrix.

    Returns:
        np.matrix: A 3D rotation matrix.

    """
    x1, x2, x3 = np.random.rand(3)
    R = np.matrix([[np.cos(2 * np.pi * x1), np.sin(2 * np.pi * x1), 0],
                   [-np.sin(2 * np.pi * x1), np.cos(2 * np.pi * x1), 0],
                   [0, 0, 1]])
    v = np.matrix([[np.cos(2 * np.pi * x2) * np.sqrt(x3)],
                   [np.sin(2 * np.pi * x2) * np.sqrt(x3)],
                   [np.sqrt(1 - x3)]])
    H = np.eye(3) - 2 * v * v.T
    M = -H * R
    return M

def rotate_coord_mat2(x,y,z):
	rot_mat = generate(3)
	x.flatten().reshape(25,25,25)
	temp_coord = np.stack([x.flatten(),y.flatten(),z.flatten()], axis=0)
	temp_coord[0].reshape(25,25,25)
	after_rotate = np.dot(rot_mat,temp_coord)

	#temp_x = after_rotate[0]
	#print temp_x.shape
	print after_rotate.shape
	print after_rotate.reshape(3,25,25,25)
	#print temp_x.reshape(25,25,25)


	return after_rotate[0].reshape(25,25,25), after_rotate[1].reshape(25,25,25), after_rotate[2].reshape(25,25,25)


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

def get_result(x,y,z,sig_x,sig_y,sig_z,x0,y0,z0, r, h, stencil, pad,x_org, y_org,z_org):

	n = f(x,y,z,sig_x,sig_y,sig_z,x0,y0,z0)
	print n.shape
	
	#temp_stencil,temp_pad = get_integration_stencil(h, h, h, r, accuracy = get_auto_accuracy(h,h,h, r))
	#temp = np.sum(n)
	temp,_ = get_fftconv_with_known_stencil_no_wrap(n,h,h,h,r,stencil,pad)
	#print temp[50,50,50]

	fig = plt.figure()
	cmap = plt.get_cmap("RdPu")
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(x_org, y_org, z_org, c=n, cmap=cmap,linewidths=0,s=5.0)
	plt.show()

	return temp[12,12,12]


r = float(sys.argv[1])
num_random = int(sys.argv[2])

nx, ny, nz = (25,25,25)
h = 0.02


xv = np.linspace(-1.0,1.0,nx)
print xv
yv = np.linspace(-1.0,1.0,ny)
zv = np.linspace(-1.0,1.0,nz)

x, y, z = np.meshgrid(xv, yv, zv)

x0,y0,z0 = (0.0, 0.0, 0.0)
sig_x = 0.001
sig_y = 0.05
sig_z = 0.2
#sig_x = np.random.uniform(0, 1.0)
#sig_y = np.random.uniform(0, 1.0)
#sig_z = np.random.uniform(0, 1.0)

stencil,pad = get_integration_stencil(h, h, h, r, accuracy = get_auto_accuracy(h,h,h, r))
print stencil

truth = get_result(x,y,z,sig_x,sig_y,sig_z,x0,y0,z0, h, r, stencil, pad,x,y,z)


result = []

for i in range(num_random):
	x_temp, y_temp, z_temp = rotate_coord_mat2(x,y,z)

	#theta1 = theta2 = theta3 = 0.0
	#theta1 = np.random.uniform(0.0, 2.0*pi)
	#theta2 = np.random.uniform(0.0, 2.0*pi)
	#theta3 = np.random.uniform(0.0, 2.0*pi)
	#print i, theta1, theta2, theta3

	#x_temp, y_temp, z_temp = rotate_coord_mat(x,y,z,theta1,theta2,theta3)
	error = get_result(x_temp, y_temp, z_temp,sig_x,sig_y,sig_z,x0,y0,z0, h, r, stencil, pad,x,y,z) - truth

	result.append(error/truth)

fig,ax = plt.subplots(figsize=(10,5))
plt.plot(np.arange(1,num_random+1),result, linewidth=5.0)
plt.show()



#get function matrix

#convolve

#get middle (50,50,50)
from __future__ import print_function
import numpy as np
from scipy.ndimage.filters import convolve
from scipy.signal import fftconvolve
from math import pi
import math
import itertools
try: import cPickle as pickle
except: import pickle
from sklearn.preprocessing import normalize
from math import sqrt
from scipy.special import legendre

def shift_array(array, a,b,c):
    A = np.roll(array,a,axis = 0)
    B = np.roll(A,b,axis = 1)
    C = np.roll(B,c,axis = 2)
    return C

def matrix_convolve2(image,kernel, mode = "periodic"):
    if mode not in ["periodic"]:
        raise NotImplemented
    if mode is "periodic":
        Nx, Ny, Nz = image.shape
        nx, ny, nz = kernel.shape
        rx = nx//2
        ry = ny//2
        rz = nz//2
        result = np.zeros((Nx, Ny, Nz))
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    result += kernel[i,j,k] * shift_array(image, rx-i, ry-j, rz-k) 
        return result

#http://paulbourke.net/geometry/circlesphere/
def check_line_intersect_with_sphere(p1, p2, r, origin = (0., 0., 0.)):
    x1,y1,z1 = p1
    x2,y2,z2 = p2
    x3,y3,z3 = origin
    a = (x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2
    b = 2*((x2-x1)*(x1-x3) + (y2-y1)*(y1-y3) + (z2-z1)*(z1-z3))
    c = x3*x3 + y3*y3 + z3*z3 + x1*x1 + y1*y1 + z1*z1 - 2*(x3*x1+y3*y1+z3*z1) - r*r
    
    if b*b-4*a*c > 0:
        return True
    else:
        return False

#https://www.geeksforgeeks.org/program-to-find-equation-of-a-plane-passing-through-3-points/
def get_plane_equation(p1,p2,p3):
    x1,y1,z1 = p1
    x2,y2,z2 = p2
    x3,y3,z3 = p3
    a1 = x2 - x1 
    b1 = y2 - y1 
    c1 = z2 - z1 
    a2 = x3 - x1 
    b2 = y3 - y1 
    c2 = z3 - z1 
    a = b1 * c2 - b2 * c1 
    b = a2 * c1 - a1 * c2 
    c = a1 * b2 - b1 * a2 
    d = (- a * x1 - b * y1 - c * z1) 
    return a,b,c,d

def check_sphere_intersect_with_sphere(p1, p2, p3, r, origin = (0., 0., 0.)):
    xs,ys,zs = origin
    a,b,c,d = get_plane_equation(p1,p2,p3)
    test_d = abs(a * xs + b * ys + c * zs + d) / sqrt(a*a + b*b + c*c)
    if r > test_d:
        return True
    else:
        return False

def shift_array(array, a,b,c):
    A = np.roll(array,a,axis = 0)
    B = np.roll(A,b,axis = 1)
    C = np.roll(B,c,axis = 2)
    return C

def matrix_convolve(image,kernel, mode = "periodic"):
    if mode not in ["periodic"]:
        raise NotImplemented
    if mode is "periodic":
        Nx, Ny, Nz = image.shape
        nx, ny, nz = kernel.shape
        rx = nx//2
        ry = ny//2
        rz = nz//2
        result = np.zeros((Nx, Ny, Nz))
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    result += kernel[i,j,k] * shift_array(image, rx-i, ry-j, rz-k) 
        return result


def get_fftconv_with_known_stencil_no_wrap(n, hx, hy, hz, r, stencil, pad):
    temp_result = fftconvolve(n,stencil, mode = 'same')
    return temp_result, pad

def get_fftconv_with_known_stencil_periodic(n, stencil):
    temp_result = fftconvolve(n,stencil, mode = 'same')
    return temp_result

def sum_magnitude(li):
    result = np.zeros_like(li[0])
    for entry in li:
        result = np.add(result,np.square(entry))
    return

def get_MC_surface_harmonic_fftconv(n, hx, hy, hz, r, l, m, accuracy = 5):
    # get the stencil and do the convolution
    
    stencil_Re, pad = calc_MC_surface_harmonic_stencil(hx, hy, hz, r, l, m, accuracy = accuracy)
    pad_temp = int(math.ceil(r*2. / min([hx,hy,hz])))
    wrapped_n = np.pad(n, pad_temp, mode='wrap')
    temp_result_Re = fftconvolve(wrapped_n,stencil_Re, mode = 'same')
    return temp_result_Re[pad_temp:-pad_temp, pad_temp:-pad_temp, pad_temp:-pad_temp], pad


def MC_surface_spherical_harmonic_n_np(x, y, z, l, n, r):
    #r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    x_hat = np.divide(x,r)
    y_hat = np.divide(y,r)
    z_hat = np.divide(z,r)

    if l == 0:
        return np.ones_like(x)

    if l == 1:
        if n == "100":
            return  x_hat 
        if n == "010":
            return  y_hat
        if n == "001":
            return  z_hat
    if l == 2:
        if n == "200":
            return np.subtract(np.multiply(np.power(x_hat, 2), 3.0), 1.0)
        if n == "110":
            return np.multiply(3.0,np.multiply(x_hat, y_hat))
        if n == "101":
            return np.multiply(3.0,np.multiply(x_hat, z_hat))
        if n == "020":
            return np.subtract(np.multiply(np.power(y_hat, 2), 3.0), 1.0)
        if n == "011":
            return np.multiply(3.0,np.multiply(y_hat, z_hat))
        if n == "002":
            return np.subtract(np.multiply(np.power(z_hat, 2), 3.0), 1.0) 
    if l == 3:
        if n == "300":
            #return  15.0 * np.power(x_hat, 3) -9.0 * x_hat
            return np.subtract(np.multiply(np.power(x_hat, 3), 15.0), np.multiply(x_hat, 9.0))
        if n == "210":
            return  15.0 * np.power(x_hat, 2) * y_hat -3.0 * y_hat 
        if n == "201":
            return  15.0 * np.power(x_hat, 2) * z_hat -3.0 * z_hat 
        if n == "120":
            return  15.0 * x_hat * np.power(y_hat, 2) -3.0 * x_hat 
        if n == "111":
            return  15.0 * x_hat * y_hat * z_hat 
        if n == "102":
            return  15.0 * x_hat * np.power(z_hat, 2) -3.0 * x_hat 
        if n == "030":
            #return  15.0 * np.power(y_hat, 3) -9.0 * y_hat
            return np.subtract(np.multiply(np.power(y_hat, 3), 15.0), np.multiply(y_hat, 9.0))
        if n == "021":
            return  15.0 * np.power(y_hat, 2) * z_hat -3.0 * z_hat 
        if n == "012":
            return  15.0 * y_hat * np.power(z_hat, 2) -3.0 * y_hat 
        if n == "003":
            #return  15.0 * np.power(z_hat, 3) -9.0 * z_hat
            return np.subtract(np.multiply(np.power(z_hat, 3), 15.0), np.multiply(z_hat, 9.0))

    if l == 4:
        if n == "400":
            return  105.0 * np.power(x_hat, 4) -90.0 * np.power(x_hat, 2) + 9.0 
        if n == "310":
            return  105.0 * np.power(x_hat, 3) * y_hat -45.0 * x_hat * y_hat 
        if n == "301":
            return  105.0 * np.power(x_hat, 3) * z_hat -45.0 * x_hat * z_hat 
        if n == "220":
            return  105.0 * np.power(x_hat, 2) * np.power(y_hat, 2) -15.0 * np.power(x_hat, 2) -15.0 * np.power(y_hat, 2) + 3.0 
        if n == "211":
            return  105.0 * np.power(x_hat, 2) * y_hat * z_hat -15.0 * y_hat * z_hat 
        if n == "202":
            return  105.0 * np.power(x_hat, 2) * np.power(z_hat, 2) -15.0 * np.power(x_hat, 2) -15.0 * np.power(z_hat, 2) + 3.0 
        if n == "130":
            return  105.0 * x_hat * np.power(y_hat, 3) -45.0 * x_hat * y_hat 
        if n == "121":
            return  105.0 * x_hat * np.power(y_hat, 2) * z_hat -15.0 * x_hat * z_hat 
        if n == "112":
            return  105.0 * x_hat * y_hat * np.power(z_hat, 2) -15.0 * x_hat * y_hat 
        if n == "103":
            return  105.0 * x_hat * np.power(z_hat, 3) -45.0 * x_hat * z_hat 
        if n == "040":
            return  105.0 * np.power(y_hat, 4) -90.0 * np.power(y_hat, 2) + 9.0 
        if n == "031":
            return  105.0 * np.power(y_hat, 3) * z_hat -45.0 * y_hat * z_hat 
        if n == "022":
            return  105.0 * np.power(y_hat, 2) * np.power(z_hat, 2) -15.0 * np.power(y_hat, 2) -15.0 * np.power(z_hat, 2) + 3.0 
        if n == "013":
            return  105.0 * y_hat * np.power(z_hat, 3) -45.0 * y_hat * z_hat 
        if n == "004":
            return  105.0 * np.power(z_hat, 4) -90.0 * np.power(z_hat, 2) + 9.0 
    if l == 5:
        if n == "500":
            return  945.0 * np.power(x_hat, 5) -1050.0 * np.power(x_hat, 3) + 225.0 * x_hat 
        if n == "410":
            return  945.0 * np.power(x_hat, 4) * y_hat -630.0 * np.power(x_hat, 2) * y_hat + 45.0 * y_hat 
        if n == "401":
            return  945.0 * np.power(x_hat, 4) * z_hat -630.0 * np.power(x_hat, 2) * z_hat + 45.0 * z_hat 
        if n == "320":
            return  945.0 * np.power(x_hat, 3) * np.power(y_hat, 2) -105.0 * np.power(x_hat, 3) -315.0 * x_hat * np.power(y_hat, 2) + 45.0 * x_hat 
        if n == "311":
            return  945.0 * np.power(x_hat, 3) * y_hat * z_hat -315.0 * x_hat * y_hat * z_hat 
        if n == "302":
            return  945.0 * np.power(x_hat, 3) * np.power(z_hat, 2) -105.0 * np.power(x_hat, 3) -315.0 * x_hat * np.power(z_hat, 2) + 45.0 * x_hat 
        if n == "230":
            return  945.0 * np.power(x_hat, 2) * np.power(y_hat, 3) -315.0 * np.power(x_hat, 2) * y_hat -105.0 * np.power(y_hat, 3) + 45.0 * y_hat 
        if n == "221":
            return  945.0 * np.power(x_hat, 2) * np.power(y_hat, 2) * z_hat -105.0 * np.power(x_hat, 2) * z_hat -105.0 * np.power(y_hat, 2) * z_hat + 15.0 * z_hat 
        if n == "212":
            return  945.0 * np.power(x_hat, 2) * y_hat * np.power(z_hat, 2) -105.0 * np.power(x_hat, 2) * y_hat -105.0 * y_hat * np.power(z_hat, 2) + 15.0 * y_hat 
        if n == "203":
            return  945.0 * np.power(x_hat, 2) * np.power(z_hat, 3) -315.0 * np.power(x_hat, 2) * z_hat -105.0 * np.power(z_hat, 3) + 45.0 * z_hat 
        if n == "140":
            return  945.0 * x_hat * np.power(y_hat, 4) -630.0 * x_hat * np.power(y_hat, 2) + 45.0 * x_hat 
        if n == "131":
            return  945.0 * x_hat * np.power(y_hat, 3) * z_hat -315.0 * x_hat * y_hat * z_hat 
        if n == "122":
            return  945.0 * x_hat * np.power(y_hat, 2) * np.power(z_hat, 2) -105.0 * x_hat * np.power(y_hat, 2) -105.0 * x_hat * np.power(z_hat, 2) + 15.0 * x_hat 
        if n == "113":
            return  945.0 * x_hat * y_hat * np.power(z_hat, 3) -315.0 * x_hat * y_hat * z_hat 
        if n == "104":
            return  945.0 * x_hat * np.power(z_hat, 4) -630.0 * x_hat * np.power(z_hat, 2) + 45.0 * x_hat 
        if n == "050":
            return  945.0 * np.power(y_hat, 5) -1050.0 * np.power(y_hat, 3) + 225.0 * y_hat 
        if n == "041":
            return  945.0 * np.power(y_hat, 4) * z_hat -630.0 * np.power(y_hat, 2) * z_hat + 45.0 * z_hat 
        if n == "032":
            return  945.0 * np.power(y_hat, 3) * np.power(z_hat, 2) -105.0 * np.power(y_hat, 3) -315.0 * y_hat * np.power(z_hat, 2) + 45.0 * y_hat 
        if n == "023":
            return  945.0 * np.power(y_hat, 2) * np.power(z_hat, 3) -315.0 * np.power(y_hat, 2) * z_hat -105.0 * np.power(z_hat, 3) + 45.0 * z_hat 
        if n == "014":
            return  945.0 * y_hat * np.power(z_hat, 4) -630.0 * y_hat * np.power(z_hat, 2) + 45.0 * y_hat 
        if n == "005":
            return  945.0 * np.power(z_hat, 5) -1050.0 * np.power(z_hat, 3) + 225.0 * z_hat 
    if l == 6:
        if n == "600":
            return  10395.0 * np.power(x_hat, 6) -14175.0 * np.power(x_hat, 4) + 4725.0 * np.power(x_hat, 2) -225.0 
        if n == "510":
            return  10395.0 * np.power(x_hat, 5) * y_hat -9450.0 * np.power(x_hat, 3) * y_hat + 1575.0 * x_hat * y_hat 
        if n == "501":
            return  10395.0 * np.power(x_hat, 5) * z_hat -9450.0 * np.power(x_hat, 3) * z_hat + 1575.0 * x_hat * z_hat 
        if n == "420":
            return  10395.0 * np.power(x_hat, 4) * np.power(y_hat, 2) -945.0 * np.power(x_hat, 4) -5670.0 * np.power(x_hat, 2) * np.power(y_hat, 2) + 630.0 * np.power(x_hat, 2) + 315.0 * np.power(y_hat, 2) -45.0 
        if n == "411":
            return  10395.0 * np.power(x_hat, 4) * y_hat * z_hat -5670.0 * np.power(x_hat, 2) * y_hat * z_hat + 315.0 * y_hat * z_hat 
        if n == "402":
            return  10395.0 * np.power(x_hat, 4) * np.power(z_hat, 2) -945.0 * np.power(x_hat, 4) -5670.0 * np.power(x_hat, 2) * np.power(z_hat, 2) + 630.0 * np.power(x_hat, 2) + 315.0 * np.power(z_hat, 2) -45.0 
        if n == "330":
            return  10395.0 * np.power(x_hat, 3) * np.power(y_hat, 3) -2835.0 * np.power(x_hat, 3) * y_hat -2835.0 * x_hat * np.power(y_hat, 3) + 945.0 * x_hat * y_hat 
        if n == "321":
            return  10395.0 * np.power(x_hat, 3) * np.power(y_hat, 2) * z_hat -945.0 * np.power(x_hat, 3) * z_hat -2835.0 * x_hat * np.power(y_hat, 2) * z_hat + 315.0 * x_hat * z_hat 
        if n == "312":
            return  10395.0 * np.power(x_hat, 3) * y_hat * np.power(z_hat, 2) -945.0 * np.power(x_hat, 3) * y_hat -2835.0 * x_hat * y_hat * np.power(z_hat, 2) + 315.0 * x_hat * y_hat 
        if n == "303":
            return  10395.0 * np.power(x_hat, 3) * np.power(z_hat, 3) -2835.0 * np.power(x_hat, 3) * z_hat -2835.0 * x_hat * np.power(z_hat, 3) + 945.0 * x_hat * z_hat 
        if n == "240":
            return  10395.0 * np.power(x_hat, 2) * np.power(y_hat, 4) -5670.0 * np.power(x_hat, 2) * np.power(y_hat, 2) + 315.0 * np.power(x_hat, 2) -945.0 * np.power(y_hat, 4) + 630.0 * np.power(y_hat, 2) -45.0 
        if n == "231":
            return  10395.0 * np.power(x_hat, 2) * np.power(y_hat, 3) * z_hat -2835.0 * np.power(x_hat, 2) * y_hat * z_hat -945.0 * np.power(y_hat, 3) * z_hat + 315.0 * y_hat * z_hat 
        if n == "222":
            return  10395.0 * np.power(x_hat, 2) * np.power(y_hat, 2) * np.power(z_hat, 2) -945.0 * np.power(x_hat, 2) * np.power(y_hat, 2) -945.0 * np.power(x_hat, 2) * np.power(z_hat, 2) + 105.0 * np.power(x_hat, 2) -945.0 * np.power(y_hat, 2) * np.power(z_hat, 2) + 105.0 * np.power(y_hat, 2) + 105.0 * np.power(z_hat, 2) -15.0 
        if n == "213":
            return  10395.0 * np.power(x_hat, 2) * y_hat * np.power(z_hat, 3) -2835.0 * np.power(x_hat, 2) * y_hat * z_hat -945.0 * y_hat * np.power(z_hat, 3) + 315.0 * y_hat * z_hat 
        if n == "204":
            return  10395.0 * np.power(x_hat, 2) * np.power(z_hat, 4) -5670.0 * np.power(x_hat, 2) * np.power(z_hat, 2) + 315.0 * np.power(x_hat, 2) -945.0 * np.power(z_hat, 4) + 630.0 * np.power(z_hat, 2) -45.0 
        if n == "150":
            return  10395.0 * x_hat * np.power(y_hat, 5) -9450.0 * x_hat * np.power(y_hat, 3) + 1575.0 * x_hat * y_hat 
        if n == "141":
            return  10395.0 * x_hat * np.power(y_hat, 4) * z_hat -5670.0 * x_hat * np.power(y_hat, 2) * z_hat + 315.0 * x_hat * z_hat 
        if n == "132":
            return  10395.0 * x_hat * np.power(y_hat, 3) * np.power(z_hat, 2) -945.0 * x_hat * np.power(y_hat, 3) -2835.0 * x_hat * y_hat * np.power(z_hat, 2) + 315.0 * x_hat * y_hat 
        if n == "123":
            return  10395.0 * x_hat * np.power(y_hat, 2) * np.power(z_hat, 3) -2835.0 * x_hat * np.power(y_hat, 2) * z_hat -945.0 * x_hat * np.power(z_hat, 3) + 315.0 * x_hat * z_hat 
        if n == "114":
            return  10395.0 * x_hat * y_hat * np.power(z_hat, 4) -5670.0 * x_hat * y_hat * np.power(z_hat, 2) + 315.0 * x_hat * y_hat 
        if n == "105":
            return  10395.0 * x_hat * np.power(z_hat, 5) -9450.0 * x_hat * np.power(z_hat, 3) + 1575.0 * x_hat * z_hat 
        if n == "060":
            return  10395.0 * np.power(y_hat, 6) -14175.0 * np.power(y_hat, 4) + 4725.0 * np.power(y_hat, 2) -225.0 
        if n == "051":
            return  10395.0 * np.power(y_hat, 5) * z_hat -9450.0 * np.power(y_hat, 3) * z_hat + 1575.0 * y_hat * z_hat 
        if n == "042":
            return  10395.0 * np.power(y_hat, 4) * np.power(z_hat, 2) -945.0 * np.power(y_hat, 4) -5670.0 * np.power(y_hat, 2) * np.power(z_hat, 2) + 630.0 * np.power(y_hat, 2) + 315.0 * np.power(z_hat, 2) -45.0 
        if n == "033":
            return  10395.0 * np.power(y_hat, 3) * np.power(z_hat, 3) -2835.0 * np.power(y_hat, 3) * z_hat -2835.0 * y_hat * np.power(z_hat, 3) + 945.0 * y_hat * z_hat 
        if n == "024":
            return  10395.0 * np.power(y_hat, 2) * np.power(z_hat, 4) -5670.0 * np.power(y_hat, 2) * np.power(z_hat, 2) + 315.0 * np.power(y_hat, 2) -945.0 * np.power(z_hat, 4) + 630.0 * np.power(z_hat, 2) -45.0 
        if n == "015":
            return  10395.0 * y_hat * np.power(z_hat, 5) -9450.0 * y_hat * np.power(z_hat, 3) + 1575.0 * y_hat * z_hat 
        if n == "006":
            return  10395.0 * np.power(z_hat, 6) -14175.0 * np.power(z_hat, 4) + 4725.0 * np.power(z_hat, 2) -225.0 
    if l == 7:
        if n == "700":
            return  135135.0 * np.power(x_hat, 7) -218295.0 * np.power(x_hat, 5) + 99225.0 * np.power(x_hat, 3) -11025.0 * x_hat 
        if n == "610":
            return  135135.0 * np.power(x_hat, 6) * y_hat -155925.0 * np.power(x_hat, 4) * y_hat + 42525.0 * np.power(x_hat, 2) * y_hat -1575.0 * y_hat 
        if n == "601":
            return  135135.0 * np.power(x_hat, 6) * z_hat -155925.0 * np.power(x_hat, 4) * z_hat + 42525.0 * np.power(x_hat, 2) * z_hat -1575.0 * z_hat 
        if n == "520":
            return  135135.0 * np.power(x_hat, 5) * np.power(y_hat, 2) -10395.0 * np.power(x_hat, 5) -103950.0 * np.power(x_hat, 3) * np.power(y_hat, 2) + 9450.0 * np.power(x_hat, 3) + 14175.0 * x_hat * np.power(y_hat, 2) -1575.0 * x_hat 
        if n == "511":
            return  135135.0 * np.power(x_hat, 5) * y_hat * z_hat -103950.0 * np.power(x_hat, 3) * y_hat * z_hat + 14175.0 * x_hat * y_hat * z_hat 
        if n == "502":
            return  135135.0 * np.power(x_hat, 5) * np.power(z_hat, 2) -10395.0 * np.power(x_hat, 5) -103950.0 * np.power(x_hat, 3) * np.power(z_hat, 2) + 9450.0 * np.power(x_hat, 3) + 14175.0 * x_hat * np.power(z_hat, 2) -1575.0 * x_hat 
        if n == "430":
            return  135135.0 * np.power(x_hat, 4) * np.power(y_hat, 3) -31185.0 * np.power(x_hat, 4) * y_hat -62370.0 * np.power(x_hat, 2) * np.power(y_hat, 3) + 17010.0 * np.power(x_hat, 2) * y_hat + 2835.0 * np.power(y_hat, 3) -945.0 * y_hat 
        if n == "421":
            return  135135.0 * np.power(x_hat, 4) * np.power(y_hat, 2) * z_hat -10395.0 * np.power(x_hat, 4) * z_hat -62370.0 * np.power(x_hat, 2) * np.power(y_hat, 2) * z_hat + 5670.0 * np.power(x_hat, 2) * z_hat + 2835.0 * np.power(y_hat, 2) * z_hat -315.0 * z_hat 
        if n == "412":
            return  135135.0 * np.power(x_hat, 4) * y_hat * np.power(z_hat, 2) -10395.0 * np.power(x_hat, 4) * y_hat -62370.0 * np.power(x_hat, 2) * y_hat * np.power(z_hat, 2) + 5670.0 * np.power(x_hat, 2) * y_hat + 2835.0 * y_hat * np.power(z_hat, 2) -315.0 * y_hat 
        if n == "403":
            return  135135.0 * np.power(x_hat, 4) * np.power(z_hat, 3) -31185.0 * np.power(x_hat, 4) * z_hat -62370.0 * np.power(x_hat, 2) * np.power(z_hat, 3) + 17010.0 * np.power(x_hat, 2) * z_hat + 2835.0 * np.power(z_hat, 3) -945.0 * z_hat 
        if n == "340":
            return  135135.0 * np.power(x_hat, 3) * np.power(y_hat, 4) -62370.0 * np.power(x_hat, 3) * np.power(y_hat, 2) + 2835.0 * np.power(x_hat, 3) -31185.0 * x_hat * np.power(y_hat, 4) + 17010.0 * x_hat * np.power(y_hat, 2) -945.0 * x_hat 
        if n == "331":
            return  135135.0 * np.power(x_hat, 3) * np.power(y_hat, 3) * z_hat -31185.0 * np.power(x_hat, 3) * y_hat * z_hat -31185.0 * x_hat * np.power(y_hat, 3) * z_hat + 8505.0 * x_hat * y_hat * z_hat 
        if n == "322":
            return  135135.0 * np.power(x_hat, 3) * np.power(y_hat, 2) * np.power(z_hat, 2) -10395.0 * np.power(x_hat, 3) * np.power(y_hat, 2) -10395.0 * np.power(x_hat, 3) * np.power(z_hat, 2) + 945.0 * np.power(x_hat, 3) -31185.0 * x_hat * np.power(y_hat, 2) * np.power(z_hat, 2) + 2835.0 * x_hat * np.power(y_hat, 2) + 2835.0 * x_hat * np.power(z_hat, 2) -315.0 * x_hat 
        if n == "313":
            return  135135.0 * np.power(x_hat, 3) * y_hat * np.power(z_hat, 3) -31185.0 * np.power(x_hat, 3) * y_hat * z_hat -31185.0 * x_hat * y_hat * np.power(z_hat, 3) + 8505.0 * x_hat * y_hat * z_hat 
        if n == "304":
            return  135135.0 * np.power(x_hat, 3) * np.power(z_hat, 4) -62370.0 * np.power(x_hat, 3) * np.power(z_hat, 2) + 2835.0 * np.power(x_hat, 3) -31185.0 * x_hat * np.power(z_hat, 4) + 17010.0 * x_hat * np.power(z_hat, 2) -945.0 * x_hat 
        if n == "250":
            return  135135.0 * np.power(x_hat, 2) * np.power(y_hat, 5) -103950.0 * np.power(x_hat, 2) * np.power(y_hat, 3) + 14175.0 * np.power(x_hat, 2) * y_hat -10395.0 * np.power(y_hat, 5) + 9450.0 * np.power(y_hat, 3) -1575.0 * y_hat 
        if n == "241":
            return  135135.0 * np.power(x_hat, 2) * np.power(y_hat, 4) * z_hat -62370.0 * np.power(x_hat, 2) * np.power(y_hat, 2) * z_hat + 2835.0 * np.power(x_hat, 2) * z_hat -10395.0 * np.power(y_hat, 4) * z_hat + 5670.0 * np.power(y_hat, 2) * z_hat -315.0 * z_hat 
        if n == "232":
            return  135135.0 * np.power(x_hat, 2) * np.power(y_hat, 3) * np.power(z_hat, 2) -10395.0 * np.power(x_hat, 2) * np.power(y_hat, 3) -31185.0 * np.power(x_hat, 2) * y_hat * np.power(z_hat, 2) + 2835.0 * np.power(x_hat, 2) * y_hat -10395.0 * np.power(y_hat, 3) * np.power(z_hat, 2) + 945.0 * np.power(y_hat, 3) + 2835.0 * y_hat * np.power(z_hat, 2) -315.0 * y_hat 
        if n == "223":
            return  135135.0 * np.power(x_hat, 2) * np.power(y_hat, 2) * np.power(z_hat, 3) -31185.0 * np.power(x_hat, 2) * np.power(y_hat, 2) * z_hat -10395.0 * np.power(x_hat, 2) * np.power(z_hat, 3) + 2835.0 * np.power(x_hat, 2) * z_hat -10395.0 * np.power(y_hat, 2) * np.power(z_hat, 3) + 2835.0 * np.power(y_hat, 2) * z_hat + 945.0 * np.power(z_hat, 3) -315.0 * z_hat 
        if n == "214":
            return  135135.0 * np.power(x_hat, 2) * y_hat * np.power(z_hat, 4) -62370.0 * np.power(x_hat, 2) * y_hat * np.power(z_hat, 2) + 2835.0 * np.power(x_hat, 2) * y_hat -10395.0 * y_hat * np.power(z_hat, 4) + 5670.0 * y_hat * np.power(z_hat, 2) -315.0 * y_hat 
        if n == "205":
            return  135135.0 * np.power(x_hat, 2) * np.power(z_hat, 5) -103950.0 * np.power(x_hat, 2) * np.power(z_hat, 3) + 14175.0 * np.power(x_hat, 2) * z_hat -10395.0 * np.power(z_hat, 5) + 9450.0 * np.power(z_hat, 3) -1575.0 * z_hat 
        if n == "160":
            return  135135.0 * x_hat * np.power(y_hat, 6) -155925.0 * x_hat * np.power(y_hat, 4) + 42525.0 * x_hat * np.power(y_hat, 2) -1575.0 * x_hat 
        if n == "151":
            return  135135.0 * x_hat * np.power(y_hat, 5) * z_hat -103950.0 * x_hat * np.power(y_hat, 3) * z_hat + 14175.0 * x_hat * y_hat * z_hat 
        if n == "142":
            return  135135.0 * x_hat * np.power(y_hat, 4) * np.power(z_hat, 2) -10395.0 * x_hat * np.power(y_hat, 4) -62370.0 * x_hat * np.power(y_hat, 2) * np.power(z_hat, 2) + 5670.0 * x_hat * np.power(y_hat, 2) + 2835.0 * x_hat * np.power(z_hat, 2) -315.0 * x_hat 
        if n == "133":
            return  135135.0 * x_hat * np.power(y_hat, 3) * np.power(z_hat, 3) -31185.0 * x_hat * np.power(y_hat, 3) * z_hat -31185.0 * x_hat * y_hat * np.power(z_hat, 3) + 8505.0 * x_hat * y_hat * z_hat 
        if n == "124":
            return  135135.0 * x_hat * np.power(y_hat, 2) * np.power(z_hat, 4) -62370.0 * x_hat * np.power(y_hat, 2) * np.power(z_hat, 2) + 2835.0 * x_hat * np.power(y_hat, 2) -10395.0 * x_hat * np.power(z_hat, 4) + 5670.0 * x_hat * np.power(z_hat, 2) -315.0 * x_hat 
        if n == "115":
            return  135135.0 * x_hat * y_hat * np.power(z_hat, 5) -103950.0 * x_hat * y_hat * np.power(z_hat, 3) + 14175.0 * x_hat * y_hat * z_hat 
        if n == "106":
            return  135135.0 * x_hat * np.power(z_hat, 6) -155925.0 * x_hat * np.power(z_hat, 4) + 42525.0 * x_hat * np.power(z_hat, 2) -1575.0 * x_hat 
        if n == "070":
            return  135135.0 * np.power(y_hat, 7) -218295.0 * np.power(y_hat, 5) + 99225.0 * np.power(y_hat, 3) -11025.0 * y_hat 
        if n == "061":
            return  135135.0 * np.power(y_hat, 6) * z_hat -155925.0 * np.power(y_hat, 4) * z_hat + 42525.0 * np.power(y_hat, 2) * z_hat -1575.0 * z_hat 
        if n == "052":
            return  135135.0 * np.power(y_hat, 5) * np.power(z_hat, 2) -10395.0 * np.power(y_hat, 5) -103950.0 * np.power(y_hat, 3) * np.power(z_hat, 2) + 9450.0 * np.power(y_hat, 3) + 14175.0 * y_hat * np.power(z_hat, 2) -1575.0 * y_hat 
        if n == "043":
            return  135135.0 * np.power(y_hat, 4) * np.power(z_hat, 3) -31185.0 * np.power(y_hat, 4) * z_hat -62370.0 * np.power(y_hat, 2) * np.power(z_hat, 3) + 17010.0 * np.power(y_hat, 2) * z_hat + 2835.0 * np.power(z_hat, 3) -945.0 * z_hat 
        if n == "034":
            return  135135.0 * np.power(y_hat, 3) * np.power(z_hat, 4) -62370.0 * np.power(y_hat, 3) * np.power(z_hat, 2) + 2835.0 * np.power(y_hat, 3) -31185.0 * y_hat * np.power(z_hat, 4) + 17010.0 * y_hat * np.power(z_hat, 2) -945.0 * y_hat 
        if n == "025":
            return  135135.0 * np.power(y_hat, 2) * np.power(z_hat, 5) -103950.0 * np.power(y_hat, 2) * np.power(z_hat, 3) + 14175.0 * np.power(y_hat, 2) * z_hat -10395.0 * np.power(z_hat, 5) + 9450.0 * np.power(z_hat, 3) -1575.0 * z_hat 
        if n == "016":
            return  135135.0 * y_hat * np.power(z_hat, 6) -155925.0 * y_hat * np.power(z_hat, 4) + 42525.0 * y_hat * np.power(z_hat, 2) -1575.0 * y_hat 
        if n == "007":
            return  135135.0 * np.power(z_hat, 7) -218295.0 * np.power(z_hat, 5) + 99225.0 * np.power(z_hat, 3) -11025.0 * z_hat 
    if l == 8:
        if n == "800":
            return  2027025.0 * np.power(x_hat, 8) -3783780.0 * np.power(x_hat, 6) + 2182950.0 * np.power(x_hat, 4) -396900.0 * np.power(x_hat, 2) + 11025.0 
        if n == "710":
            return  2027025.0 * np.power(x_hat, 7) * y_hat -2837835.0 * np.power(x_hat, 5) * y_hat + 1091475.0 * np.power(x_hat, 3) * y_hat -99225.0 * x_hat * y_hat 
        if n == "701":
            return  2027025.0 * np.power(x_hat, 7) * z_hat -2837835.0 * np.power(x_hat, 5) * z_hat + 1091475.0 * np.power(x_hat, 3) * z_hat -99225.0 * x_hat * z_hat 
        if n == "620":
            return  2027025.0 * np.power(x_hat, 6) * np.power(y_hat, 2) -135135.0 * np.power(x_hat, 6) -2027025.0 * np.power(x_hat, 4) * np.power(y_hat, 2) + 155925.0 * np.power(x_hat, 4) + 467775.0 * np.power(x_hat, 2) * np.power(y_hat, 2) -42525.0 * np.power(x_hat, 2) -14175.0 * np.power(y_hat, 2) + 1575.0 
        if n == "611":
            return  2027025.0 * np.power(x_hat, 6) * y_hat * z_hat -2027025.0 * np.power(x_hat, 4) * y_hat * z_hat + 467775.0 * np.power(x_hat, 2) * y_hat * z_hat -14175.0 * y_hat * z_hat 
        if n == "602":
            return  2027025.0 * np.power(x_hat, 6) * np.power(z_hat, 2) -135135.0 * np.power(x_hat, 6) -2027025.0 * np.power(x_hat, 4) * np.power(z_hat, 2) + 155925.0 * np.power(x_hat, 4) + 467775.0 * np.power(x_hat, 2) * np.power(z_hat, 2) -42525.0 * np.power(x_hat, 2) -14175.0 * np.power(z_hat, 2) + 1575.0 
        if n == "530":
            return  2027025.0 * np.power(x_hat, 5) * np.power(y_hat, 3) -405405.0 * np.power(x_hat, 5) * y_hat -1351350.0 * np.power(x_hat, 3) * np.power(y_hat, 3) + 311850.0 * np.power(x_hat, 3) * y_hat + 155925.0 * x_hat * np.power(y_hat, 3) -42525.0 * x_hat * y_hat 
        if n == "521":
            return  2027025.0 * np.power(x_hat, 5) * np.power(y_hat, 2) * z_hat -135135.0 * np.power(x_hat, 5) * z_hat -1351350.0 * np.power(x_hat, 3) * np.power(y_hat, 2) * z_hat + 103950.0 * np.power(x_hat, 3) * z_hat + 155925.0 * x_hat * np.power(y_hat, 2) * z_hat -14175.0 * x_hat * z_hat 
        if n == "512":
            return  2027025.0 * np.power(x_hat, 5) * y_hat * np.power(z_hat, 2) -135135.0 * np.power(x_hat, 5) * y_hat -1351350.0 * np.power(x_hat, 3) * y_hat * np.power(z_hat, 2) + 103950.0 * np.power(x_hat, 3) * y_hat + 155925.0 * x_hat * y_hat * np.power(z_hat, 2) -14175.0 * x_hat * y_hat 
        if n == "503":
            return  2027025.0 * np.power(x_hat, 5) * np.power(z_hat, 3) -405405.0 * np.power(x_hat, 5) * z_hat -1351350.0 * np.power(x_hat, 3) * np.power(z_hat, 3) + 311850.0 * np.power(x_hat, 3) * z_hat + 155925.0 * x_hat * np.power(z_hat, 3) -42525.0 * x_hat * z_hat 
        if n == "440":
            return  2027025.0 * np.power(x_hat, 4) * np.power(y_hat, 4) -810810.0 * np.power(x_hat, 4) * np.power(y_hat, 2) + 31185.0 * np.power(x_hat, 4) -810810.0 * np.power(x_hat, 2) * np.power(y_hat, 4) + 374220.0 * np.power(x_hat, 2) * np.power(y_hat, 2) -17010.0 * np.power(x_hat, 2) + 31185.0 * np.power(y_hat, 4) -17010.0 * np.power(y_hat, 2) + 945.0 
        if n == "431":
            return  2027025.0 * np.power(x_hat, 4) * np.power(y_hat, 3) * z_hat -405405.0 * np.power(x_hat, 4) * y_hat * z_hat -810810.0 * np.power(x_hat, 2) * np.power(y_hat, 3) * z_hat + 187110.0 * np.power(x_hat, 2) * y_hat * z_hat + 31185.0 * np.power(y_hat, 3) * z_hat -8505.0 * y_hat * z_hat 
        if n == "422":
            return  2027025.0 * np.power(x_hat, 4) * np.power(y_hat, 2) * np.power(z_hat, 2) -135135.0 * np.power(x_hat, 4) * np.power(y_hat, 2) -135135.0 * np.power(x_hat, 4) * np.power(z_hat, 2) + 10395.0 * np.power(x_hat, 4) -810810.0 * np.power(x_hat, 2) * np.power(y_hat, 2) * np.power(z_hat, 2) + 62370.0 * np.power(x_hat, 2) * np.power(y_hat, 2) + 62370.0 * np.power(x_hat, 2) * np.power(z_hat, 2) -5670.0 * np.power(x_hat, 2) + 31185.0 * np.power(y_hat, 2) * np.power(z_hat, 2) -2835.0 * np.power(y_hat, 2) -2835.0 * np.power(z_hat, 2) + 315.0 
        if n == "413":
            return  2027025.0 * np.power(x_hat, 4) * y_hat * np.power(z_hat, 3) -405405.0 * np.power(x_hat, 4) * y_hat * z_hat -810810.0 * np.power(x_hat, 2) * y_hat * np.power(z_hat, 3) + 187110.0 * np.power(x_hat, 2) * y_hat * z_hat + 31185.0 * y_hat * np.power(z_hat, 3) -8505.0 * y_hat * z_hat 
        if n == "404":
            return  2027025.0 * np.power(x_hat, 4) * np.power(z_hat, 4) -810810.0 * np.power(x_hat, 4) * np.power(z_hat, 2) + 31185.0 * np.power(x_hat, 4) -810810.0 * np.power(x_hat, 2) * np.power(z_hat, 4) + 374220.0 * np.power(x_hat, 2) * np.power(z_hat, 2) -17010.0 * np.power(x_hat, 2) + 31185.0 * np.power(z_hat, 4) -17010.0 * np.power(z_hat, 2) + 945.0 
        if n == "350":
            return  2027025.0 * np.power(x_hat, 3) * np.power(y_hat, 5) -1351350.0 * np.power(x_hat, 3) * np.power(y_hat, 3) + 155925.0 * np.power(x_hat, 3) * y_hat -405405.0 * x_hat * np.power(y_hat, 5) + 311850.0 * x_hat * np.power(y_hat, 3) -42525.0 * x_hat * y_hat 
        if n == "341":
            return  2027025.0 * np.power(x_hat, 3) * np.power(y_hat, 4) * z_hat -810810.0 * np.power(x_hat, 3) * np.power(y_hat, 2) * z_hat + 31185.0 * np.power(x_hat, 3) * z_hat -405405.0 * x_hat * np.power(y_hat, 4) * z_hat + 187110.0 * x_hat * np.power(y_hat, 2) * z_hat -8505.0 * x_hat * z_hat 
        if n == "332":
            return  2027025.0 * np.power(x_hat, 3) * np.power(y_hat, 3) * np.power(z_hat, 2) -135135.0 * np.power(x_hat, 3) * np.power(y_hat, 3) -405405.0 * np.power(x_hat, 3) * y_hat * np.power(z_hat, 2) + 31185.0 * np.power(x_hat, 3) * y_hat -405405.0 * x_hat * np.power(y_hat, 3) * np.power(z_hat, 2) + 31185.0 * x_hat * np.power(y_hat, 3) + 93555.0 * x_hat * y_hat * np.power(z_hat, 2) -8505.0 * x_hat * y_hat 
        if n == "323":
            return  2027025.0 * np.power(x_hat, 3) * np.power(y_hat, 2) * np.power(z_hat, 3) -405405.0 * np.power(x_hat, 3) * np.power(y_hat, 2) * z_hat -135135.0 * np.power(x_hat, 3) * np.power(z_hat, 3) + 31185.0 * np.power(x_hat, 3) * z_hat -405405.0 * x_hat * np.power(y_hat, 2) * np.power(z_hat, 3) + 93555.0 * x_hat * np.power(y_hat, 2) * z_hat + 31185.0 * x_hat * np.power(z_hat, 3) -8505.0 * x_hat * z_hat 
        if n == "314":
            return  2027025.0 * np.power(x_hat, 3) * y_hat * np.power(z_hat, 4) -810810.0 * np.power(x_hat, 3) * y_hat * np.power(z_hat, 2) + 31185.0 * np.power(x_hat, 3) * y_hat -405405.0 * x_hat * y_hat * np.power(z_hat, 4) + 187110.0 * x_hat * y_hat * np.power(z_hat, 2) -8505.0 * x_hat * y_hat 
        if n == "305":
            return  2027025.0 * np.power(x_hat, 3) * np.power(z_hat, 5) -1351350.0 * np.power(x_hat, 3) * np.power(z_hat, 3) + 155925.0 * np.power(x_hat, 3) * z_hat -405405.0 * x_hat * np.power(z_hat, 5) + 311850.0 * x_hat * np.power(z_hat, 3) -42525.0 * x_hat * z_hat 
        if n == "260":
            return  2027025.0 * np.power(x_hat, 2) * np.power(y_hat, 6) -2027025.0 * np.power(x_hat, 2) * np.power(y_hat, 4) + 467775.0 * np.power(x_hat, 2) * np.power(y_hat, 2) -14175.0 * np.power(x_hat, 2) -135135.0 * np.power(y_hat, 6) + 155925.0 * np.power(y_hat, 4) -42525.0 * np.power(y_hat, 2) + 1575.0 
        if n == "251":
            return  2027025.0 * np.power(x_hat, 2) * np.power(y_hat, 5) * z_hat -1351350.0 * np.power(x_hat, 2) * np.power(y_hat, 3) * z_hat + 155925.0 * np.power(x_hat, 2) * y_hat * z_hat -135135.0 * np.power(y_hat, 5) * z_hat + 103950.0 * np.power(y_hat, 3) * z_hat -14175.0 * y_hat * z_hat 
        if n == "242":
            return  2027025.0 * np.power(x_hat, 2) * np.power(y_hat, 4) * np.power(z_hat, 2) -135135.0 * np.power(x_hat, 2) * np.power(y_hat, 4) -810810.0 * np.power(x_hat, 2) * np.power(y_hat, 2) * np.power(z_hat, 2) + 62370.0 * np.power(x_hat, 2) * np.power(y_hat, 2) + 31185.0 * np.power(x_hat, 2) * np.power(z_hat, 2) -2835.0 * np.power(x_hat, 2) -135135.0 * np.power(y_hat, 4) * np.power(z_hat, 2) + 10395.0 * np.power(y_hat, 4) + 62370.0 * np.power(y_hat, 2) * np.power(z_hat, 2) -5670.0 * np.power(y_hat, 2) -2835.0 * np.power(z_hat, 2) + 315.0 
        if n == "233":
            return  2027025.0 * np.power(x_hat, 2) * np.power(y_hat, 3) * np.power(z_hat, 3) -405405.0 * np.power(x_hat, 2) * np.power(y_hat, 3) * z_hat -405405.0 * np.power(x_hat, 2) * y_hat * np.power(z_hat, 3) + 93555.0 * np.power(x_hat, 2) * y_hat * z_hat -135135.0 * np.power(y_hat, 3) * np.power(z_hat, 3) + 31185.0 * np.power(y_hat, 3) * z_hat + 31185.0 * y_hat * np.power(z_hat, 3) -8505.0 * y_hat * z_hat 
        if n == "224":
            return  2027025.0 * np.power(x_hat, 2) * np.power(y_hat, 2) * np.power(z_hat, 4) -810810.0 * np.power(x_hat, 2) * np.power(y_hat, 2) * np.power(z_hat, 2) + 31185.0 * np.power(x_hat, 2) * np.power(y_hat, 2) -135135.0 * np.power(x_hat, 2) * np.power(z_hat, 4) + 62370.0 * np.power(x_hat, 2) * np.power(z_hat, 2) -2835.0 * np.power(x_hat, 2) -135135.0 * np.power(y_hat, 2) * np.power(z_hat, 4) + 62370.0 * np.power(y_hat, 2) * np.power(z_hat, 2) -2835.0 * np.power(y_hat, 2) + 10395.0 * np.power(z_hat, 4) -5670.0 * np.power(z_hat, 2) + 315.0 
        if n == "215":
            return  2027025.0 * np.power(x_hat, 2) * y_hat * np.power(z_hat, 5) -1351350.0 * np.power(x_hat, 2) * y_hat * np.power(z_hat, 3) + 155925.0 * np.power(x_hat, 2) * y_hat * z_hat -135135.0 * y_hat * np.power(z_hat, 5) + 103950.0 * y_hat * np.power(z_hat, 3) -14175.0 * y_hat * z_hat 
        if n == "206":
            return  2027025.0 * np.power(x_hat, 2) * np.power(z_hat, 6) -2027025.0 * np.power(x_hat, 2) * np.power(z_hat, 4) + 467775.0 * np.power(x_hat, 2) * np.power(z_hat, 2) -14175.0 * np.power(x_hat, 2) -135135.0 * np.power(z_hat, 6) + 155925.0 * np.power(z_hat, 4) -42525.0 * np.power(z_hat, 2) + 1575.0 
        if n == "170":
            return  2027025.0 * x_hat * np.power(y_hat, 7) -2837835.0 * x_hat * np.power(y_hat, 5) + 1091475.0 * x_hat * np.power(y_hat, 3) -99225.0 * x_hat * y_hat 
        if n == "161":
            return  2027025.0 * x_hat * np.power(y_hat, 6) * z_hat -2027025.0 * x_hat * np.power(y_hat, 4) * z_hat + 467775.0 * x_hat * np.power(y_hat, 2) * z_hat -14175.0 * x_hat * z_hat 
        if n == "152":
            return  2027025.0 * x_hat * np.power(y_hat, 5) * np.power(z_hat, 2) -135135.0 * x_hat * np.power(y_hat, 5) -1351350.0 * x_hat * np.power(y_hat, 3) * np.power(z_hat, 2) + 103950.0 * x_hat * np.power(y_hat, 3) + 155925.0 * x_hat * y_hat * np.power(z_hat, 2) -14175.0 * x_hat * y_hat 
        if n == "143":
            return  2027025.0 * x_hat * np.power(y_hat, 4) * np.power(z_hat, 3) -405405.0 * x_hat * np.power(y_hat, 4) * z_hat -810810.0 * x_hat * np.power(y_hat, 2) * np.power(z_hat, 3) + 187110.0 * x_hat * np.power(y_hat, 2) * z_hat + 31185.0 * x_hat * np.power(z_hat, 3) -8505.0 * x_hat * z_hat 
        if n == "134":
            return  2027025.0 * x_hat * np.power(y_hat, 3) * np.power(z_hat, 4) -810810.0 * x_hat * np.power(y_hat, 3) * np.power(z_hat, 2) + 31185.0 * x_hat * np.power(y_hat, 3) -405405.0 * x_hat * y_hat * np.power(z_hat, 4) + 187110.0 * x_hat * y_hat * np.power(z_hat, 2) -8505.0 * x_hat * y_hat 
        if n == "125":
            return  2027025.0 * x_hat * np.power(y_hat, 2) * np.power(z_hat, 5) -1351350.0 * x_hat * np.power(y_hat, 2) * np.power(z_hat, 3) + 155925.0 * x_hat * np.power(y_hat, 2) * z_hat -135135.0 * x_hat * np.power(z_hat, 5) + 103950.0 * x_hat * np.power(z_hat, 3) -14175.0 * x_hat * z_hat 
        if n == "116":
            return  2027025.0 * x_hat * y_hat * np.power(z_hat, 6) -2027025.0 * x_hat * y_hat * np.power(z_hat, 4) + 467775.0 * x_hat * y_hat * np.power(z_hat, 2) -14175.0 * x_hat * y_hat 
        if n == "107":
            return  2027025.0 * x_hat * np.power(z_hat, 7) -2837835.0 * x_hat * np.power(z_hat, 5) + 1091475.0 * x_hat * np.power(z_hat, 3) -99225.0 * x_hat * z_hat 
        if n == "080":
            return  2027025.0 * np.power(y_hat, 8) -3783780.0 * np.power(y_hat, 6) + 2182950.0 * np.power(y_hat, 4) -396900.0 * np.power(y_hat, 2) + 11025.0 
        if n == "071":
            return  2027025.0 * np.power(y_hat, 7) * z_hat -2837835.0 * np.power(y_hat, 5) * z_hat + 1091475.0 * np.power(y_hat, 3) * z_hat -99225.0 * y_hat * z_hat 
        if n == "062":
            return  2027025.0 * np.power(y_hat, 6) * np.power(z_hat, 2) -135135.0 * np.power(y_hat, 6) -2027025.0 * np.power(y_hat, 4) * np.power(z_hat, 2) + 155925.0 * np.power(y_hat, 4) + 467775.0 * np.power(y_hat, 2) * np.power(z_hat, 2) -42525.0 * np.power(y_hat, 2) -14175.0 * np.power(z_hat, 2) + 1575.0 
        if n == "053":
            return  2027025.0 * np.power(y_hat, 5) * np.power(z_hat, 3) -405405.0 * np.power(y_hat, 5) * z_hat -1351350.0 * np.power(y_hat, 3) * np.power(z_hat, 3) + 311850.0 * np.power(y_hat, 3) * z_hat + 155925.0 * y_hat * np.power(z_hat, 3) -42525.0 * y_hat * z_hat 
        if n == "044":
            return  2027025.0 * np.power(y_hat, 4) * np.power(z_hat, 4) -810810.0 * np.power(y_hat, 4) * np.power(z_hat, 2) + 31185.0 * np.power(y_hat, 4) -810810.0 * np.power(y_hat, 2) * np.power(z_hat, 4) + 374220.0 * np.power(y_hat, 2) * np.power(z_hat, 2) -17010.0 * np.power(y_hat, 2) + 31185.0 * np.power(z_hat, 4) -17010.0 * np.power(z_hat, 2) + 945.0 
        if n == "035":
            return  2027025.0 * np.power(y_hat, 3) * np.power(z_hat, 5) -1351350.0 * np.power(y_hat, 3) * np.power(z_hat, 3) + 155925.0 * np.power(y_hat, 3) * z_hat -405405.0 * y_hat * np.power(z_hat, 5) + 311850.0 * y_hat * np.power(z_hat, 3) -42525.0 * y_hat * z_hat 
        if n == "026":
            return  2027025.0 * np.power(y_hat, 2) * np.power(z_hat, 6) -2027025.0 * np.power(y_hat, 2) * np.power(z_hat, 4) + 467775.0 * np.power(y_hat, 2) * np.power(z_hat, 2) -14175.0 * np.power(y_hat, 2) -135135.0 * np.power(z_hat, 6) + 155925.0 * np.power(z_hat, 4) -42525.0 * np.power(z_hat, 2) + 1575.0 
        if n == "017":
            return  2027025.0 * y_hat * np.power(z_hat, 7) -2837835.0 * y_hat * np.power(z_hat, 5) + 1091475.0 * y_hat * np.power(z_hat, 3) -99225.0 * y_hat * z_hat 
        if n == "008":
            return  2027025.0 * np.power(z_hat, 8) -3783780.0 * np.power(z_hat, 6) + 2182950.0 * np.power(z_hat, 4) -396900.0 * np.power(z_hat, 2) + 11025.0 
    if l == 9:
        if n == "900":
            return  34459425.0 * np.power(x_hat, 9) -72972900.0 * np.power(x_hat, 7) + 51081030.0 * np.power(x_hat, 5) -13097700.0 * np.power(x_hat, 3) + 893025.0 * x_hat 
        if n == "810":
            return  34459425.0 * np.power(x_hat, 8) * y_hat -56756700.0 * np.power(x_hat, 6) * y_hat + 28378350.0 * np.power(x_hat, 4) * y_hat -4365900.0 * np.power(x_hat, 2) * y_hat + 99225.0 * y_hat 
        if n == "801":
            return  34459425.0 * np.power(x_hat, 8) * z_hat -56756700.0 * np.power(x_hat, 6) * z_hat + 28378350.0 * np.power(x_hat, 4) * z_hat -4365900.0 * np.power(x_hat, 2) * z_hat + 99225.0 * z_hat 
        if n == "720":
            return  34459425.0 * np.power(x_hat, 7) * np.power(y_hat, 2) -2027025.0 * np.power(x_hat, 7) -42567525.0 * np.power(x_hat, 5) * np.power(y_hat, 2) + 2837835.0 * np.power(x_hat, 5) + 14189175.0 * np.power(x_hat, 3) * np.power(y_hat, 2) -1091475.0 * np.power(x_hat, 3) -1091475.0 * x_hat * np.power(y_hat, 2) + 99225.0 * x_hat 
        if n == "711":
            return  34459425.0 * np.power(x_hat, 7) * y_hat * z_hat -42567525.0 * np.power(x_hat, 5) * y_hat * z_hat + 14189175.0 * np.power(x_hat, 3) * y_hat * z_hat -1091475.0 * x_hat * y_hat * z_hat 
        if n == "702":
            return  34459425.0 * np.power(x_hat, 7) * np.power(z_hat, 2) -2027025.0 * np.power(x_hat, 7) -42567525.0 * np.power(x_hat, 5) * np.power(z_hat, 2) + 2837835.0 * np.power(x_hat, 5) + 14189175.0 * np.power(x_hat, 3) * np.power(z_hat, 2) -1091475.0 * np.power(x_hat, 3) -1091475.0 * x_hat * np.power(z_hat, 2) + 99225.0 * x_hat 
        if n == "630":
            return  34459425.0 * np.power(x_hat, 6) * np.power(y_hat, 3) -6081075.0 * np.power(x_hat, 6) * y_hat -30405375.0 * np.power(x_hat, 4) * np.power(y_hat, 3) + 6081075.0 * np.power(x_hat, 4) * y_hat + 6081075.0 * np.power(x_hat, 2) * np.power(y_hat, 3) -1403325.0 * np.power(x_hat, 2) * y_hat -155925.0 * np.power(y_hat, 3) + 42525.0 * y_hat 
        if n == "621":
            return  34459425.0 * np.power(x_hat, 6) * np.power(y_hat, 2) * z_hat -2027025.0 * np.power(x_hat, 6) * z_hat -30405375.0 * np.power(x_hat, 4) * np.power(y_hat, 2) * z_hat + 2027025.0 * np.power(x_hat, 4) * z_hat + 6081075.0 * np.power(x_hat, 2) * np.power(y_hat, 2) * z_hat -467775.0 * np.power(x_hat, 2) * z_hat -155925.0 * np.power(y_hat, 2) * z_hat + 14175.0 * z_hat 
        if n == "612":
            return  34459425.0 * np.power(x_hat, 6) * y_hat * np.power(z_hat, 2) -2027025.0 * np.power(x_hat, 6) * y_hat -30405375.0 * np.power(x_hat, 4) * y_hat * np.power(z_hat, 2) + 2027025.0 * np.power(x_hat, 4) * y_hat + 6081075.0 * np.power(x_hat, 2) * y_hat * np.power(z_hat, 2) -467775.0 * np.power(x_hat, 2) * y_hat -155925.0 * y_hat * np.power(z_hat, 2) + 14175.0 * y_hat 
        if n == "603":
            return  34459425.0 * np.power(x_hat, 6) * np.power(z_hat, 3) -6081075.0 * np.power(x_hat, 6) * z_hat -30405375.0 * np.power(x_hat, 4) * np.power(z_hat, 3) + 6081075.0 * np.power(x_hat, 4) * z_hat + 6081075.0 * np.power(x_hat, 2) * np.power(z_hat, 3) -1403325.0 * np.power(x_hat, 2) * z_hat -155925.0 * np.power(z_hat, 3) + 42525.0 * z_hat 
        if n == "540":
            return  34459425.0 * np.power(x_hat, 5) * np.power(y_hat, 4) -12162150.0 * np.power(x_hat, 5) * np.power(y_hat, 2) + 405405.0 * np.power(x_hat, 5) -20270250.0 * np.power(x_hat, 3) * np.power(y_hat, 4) + 8108100.0 * np.power(x_hat, 3) * np.power(y_hat, 2) -311850.0 * np.power(x_hat, 3) + 2027025.0 * x_hat * np.power(y_hat, 4) -935550.0 * x_hat * np.power(y_hat, 2) + 42525.0 * x_hat 
        if n == "531":
            return  34459425.0 * np.power(x_hat, 5) * np.power(y_hat, 3) * z_hat -6081075.0 * np.power(x_hat, 5) * y_hat * z_hat -20270250.0 * np.power(x_hat, 3) * np.power(y_hat, 3) * z_hat + 4054050.0 * np.power(x_hat, 3) * y_hat * z_hat + 2027025.0 * x_hat * np.power(y_hat, 3) * z_hat -467775.0 * x_hat * y_hat * z_hat 
        if n == "522":
            return  34459425.0 * np.power(x_hat, 5) * np.power(y_hat, 2) * np.power(z_hat, 2) -2027025.0 * np.power(x_hat, 5) * np.power(y_hat, 2) -2027025.0 * np.power(x_hat, 5) * np.power(z_hat, 2) + 135135.0 * np.power(x_hat, 5) -20270250.0 * np.power(x_hat, 3) * np.power(y_hat, 2) * np.power(z_hat, 2) + 1351350.0 * np.power(x_hat, 3) * np.power(y_hat, 2) + 1351350.0 * np.power(x_hat, 3) * np.power(z_hat, 2) -103950.0 * np.power(x_hat, 3) + 2027025.0 * x_hat * np.power(y_hat, 2) * np.power(z_hat, 2) -155925.0 * x_hat * np.power(y_hat, 2) -155925.0 * x_hat * np.power(z_hat, 2) + 14175.0 * x_hat 
        if n == "513":
            return  34459425.0 * np.power(x_hat, 5) * y_hat * np.power(z_hat, 3) -6081075.0 * np.power(x_hat, 5) * y_hat * z_hat -20270250.0 * np.power(x_hat, 3) * y_hat * np.power(z_hat, 3) + 4054050.0 * np.power(x_hat, 3) * y_hat * z_hat + 2027025.0 * x_hat * y_hat * np.power(z_hat, 3) -467775.0 * x_hat * y_hat * z_hat 
        if n == "504":
            return  34459425.0 * np.power(x_hat, 5) * np.power(z_hat, 4) -12162150.0 * np.power(x_hat, 5) * np.power(z_hat, 2) + 405405.0 * np.power(x_hat, 5) -20270250.0 * np.power(x_hat, 3) * np.power(z_hat, 4) + 8108100.0 * np.power(x_hat, 3) * np.power(z_hat, 2) -311850.0 * np.power(x_hat, 3) + 2027025.0 * x_hat * np.power(z_hat, 4) -935550.0 * x_hat * np.power(z_hat, 2) + 42525.0 * x_hat 
        if n == "450":
            return  34459425.0 * np.power(x_hat, 4) * np.power(y_hat, 5) -20270250.0 * np.power(x_hat, 4) * np.power(y_hat, 3) + 2027025.0 * np.power(x_hat, 4) * y_hat -12162150.0 * np.power(x_hat, 2) * np.power(y_hat, 5) + 8108100.0 * np.power(x_hat, 2) * np.power(y_hat, 3) -935550.0 * np.power(x_hat, 2) * y_hat + 405405.0 * np.power(y_hat, 5) -311850.0 * np.power(y_hat, 3) + 42525.0 * y_hat 
        if n == "441":
            return  34459425.0 * np.power(x_hat, 4) * np.power(y_hat, 4) * z_hat -12162150.0 * np.power(x_hat, 4) * np.power(y_hat, 2) * z_hat + 405405.0 * np.power(x_hat, 4) * z_hat -12162150.0 * np.power(x_hat, 2) * np.power(y_hat, 4) * z_hat + 4864860.0 * np.power(x_hat, 2) * np.power(y_hat, 2) * z_hat -187110.0 * np.power(x_hat, 2) * z_hat + 405405.0 * np.power(y_hat, 4) * z_hat -187110.0 * np.power(y_hat, 2) * z_hat + 8505.0 * z_hat 
        if n == "432":
            return  34459425.0 * np.power(x_hat, 4) * np.power(y_hat, 3) * np.power(z_hat, 2) -2027025.0 * np.power(x_hat, 4) * np.power(y_hat, 3) -6081075.0 * np.power(x_hat, 4) * y_hat * np.power(z_hat, 2) + 405405.0 * np.power(x_hat, 4) * y_hat -12162150.0 * np.power(x_hat, 2) * np.power(y_hat, 3) * np.power(z_hat, 2) + 810810.0 * np.power(x_hat, 2) * np.power(y_hat, 3) + 2432430.0 * np.power(x_hat, 2) * y_hat * np.power(z_hat, 2) -187110.0 * np.power(x_hat, 2) * y_hat + 405405.0 * np.power(y_hat, 3) * np.power(z_hat, 2) -31185.0 * np.power(y_hat, 3) -93555.0 * y_hat * np.power(z_hat, 2) + 8505.0 * y_hat 
        if n == "423":
            return  34459425.0 * np.power(x_hat, 4) * np.power(y_hat, 2) * np.power(z_hat, 3) -6081075.0 * np.power(x_hat, 4) * np.power(y_hat, 2) * z_hat -2027025.0 * np.power(x_hat, 4) * np.power(z_hat, 3) + 405405.0 * np.power(x_hat, 4) * z_hat -12162150.0 * np.power(x_hat, 2) * np.power(y_hat, 2) * np.power(z_hat, 3) + 2432430.0 * np.power(x_hat, 2) * np.power(y_hat, 2) * z_hat + 810810.0 * np.power(x_hat, 2) * np.power(z_hat, 3) -187110.0 * np.power(x_hat, 2) * z_hat + 405405.0 * np.power(y_hat, 2) * np.power(z_hat, 3) -93555.0 * np.power(y_hat, 2) * z_hat -31185.0 * np.power(z_hat, 3) + 8505.0 * z_hat 
        if n == "414":
            return  34459425.0 * np.power(x_hat, 4) * y_hat * np.power(z_hat, 4) -12162150.0 * np.power(x_hat, 4) * y_hat * np.power(z_hat, 2) + 405405.0 * np.power(x_hat, 4) * y_hat -12162150.0 * np.power(x_hat, 2) * y_hat * np.power(z_hat, 4) + 4864860.0 * np.power(x_hat, 2) * y_hat * np.power(z_hat, 2) -187110.0 * np.power(x_hat, 2) * y_hat + 405405.0 * y_hat * np.power(z_hat, 4) -187110.0 * y_hat * np.power(z_hat, 2) + 8505.0 * y_hat 
        if n == "405":
            return  34459425.0 * np.power(x_hat, 4) * np.power(z_hat, 5) -20270250.0 * np.power(x_hat, 4) * np.power(z_hat, 3) + 2027025.0 * np.power(x_hat, 4) * z_hat -12162150.0 * np.power(x_hat, 2) * np.power(z_hat, 5) + 8108100.0 * np.power(x_hat, 2) * np.power(z_hat, 3) -935550.0 * np.power(x_hat, 2) * z_hat + 405405.0 * np.power(z_hat, 5) -311850.0 * np.power(z_hat, 3) + 42525.0 * z_hat 
        if n == "360":
            return  34459425.0 * np.power(x_hat, 3) * np.power(y_hat, 6) -30405375.0 * np.power(x_hat, 3) * np.power(y_hat, 4) + 6081075.0 * np.power(x_hat, 3) * np.power(y_hat, 2) -155925.0 * np.power(x_hat, 3) -6081075.0 * x_hat * np.power(y_hat, 6) + 6081075.0 * x_hat * np.power(y_hat, 4) -1403325.0 * x_hat * np.power(y_hat, 2) + 42525.0 * x_hat 
        if n == "351":
            return  34459425.0 * np.power(x_hat, 3) * np.power(y_hat, 5) * z_hat -20270250.0 * np.power(x_hat, 3) * np.power(y_hat, 3) * z_hat + 2027025.0 * np.power(x_hat, 3) * y_hat * z_hat -6081075.0 * x_hat * np.power(y_hat, 5) * z_hat + 4054050.0 * x_hat * np.power(y_hat, 3) * z_hat -467775.0 * x_hat * y_hat * z_hat 
        if n == "342":
            return  34459425.0 * np.power(x_hat, 3) * np.power(y_hat, 4) * np.power(z_hat, 2) -2027025.0 * np.power(x_hat, 3) * np.power(y_hat, 4) -12162150.0 * np.power(x_hat, 3) * np.power(y_hat, 2) * np.power(z_hat, 2) + 810810.0 * np.power(x_hat, 3) * np.power(y_hat, 2) + 405405.0 * np.power(x_hat, 3) * np.power(z_hat, 2) -31185.0 * np.power(x_hat, 3) -6081075.0 * x_hat * np.power(y_hat, 4) * np.power(z_hat, 2) + 405405.0 * x_hat * np.power(y_hat, 4) + 2432430.0 * x_hat * np.power(y_hat, 2) * np.power(z_hat, 2) -187110.0 * x_hat * np.power(y_hat, 2) -93555.0 * x_hat * np.power(z_hat, 2) + 8505.0 * x_hat 
        if n == "333":
            return  34459425.0 * np.power(x_hat, 3) * np.power(y_hat, 3) * np.power(z_hat, 3) -6081075.0 * np.power(x_hat, 3) * np.power(y_hat, 3) * z_hat -6081075.0 * np.power(x_hat, 3) * y_hat * np.power(z_hat, 3) + 1216215.0 * np.power(x_hat, 3) * y_hat * z_hat -6081075.0 * x_hat * np.power(y_hat, 3) * np.power(z_hat, 3) + 1216215.0 * x_hat * np.power(y_hat, 3) * z_hat + 1216215.0 * x_hat * y_hat * np.power(z_hat, 3) -280665.0 * x_hat * y_hat * z_hat 
        if n == "324":
            return  34459425.0 * np.power(x_hat, 3) * np.power(y_hat, 2) * np.power(z_hat, 4) -12162150.0 * np.power(x_hat, 3) * np.power(y_hat, 2) * np.power(z_hat, 2) + 405405.0 * np.power(x_hat, 3) * np.power(y_hat, 2) -2027025.0 * np.power(x_hat, 3) * np.power(z_hat, 4) + 810810.0 * np.power(x_hat, 3) * np.power(z_hat, 2) -31185.0 * np.power(x_hat, 3) -6081075.0 * x_hat * np.power(y_hat, 2) * np.power(z_hat, 4) + 2432430.0 * x_hat * np.power(y_hat, 2) * np.power(z_hat, 2) -93555.0 * x_hat * np.power(y_hat, 2) + 405405.0 * x_hat * np.power(z_hat, 4) -187110.0 * x_hat * np.power(z_hat, 2) + 8505.0 * x_hat 
        if n == "315":
            return  34459425.0 * np.power(x_hat, 3) * y_hat * np.power(z_hat, 5) -20270250.0 * np.power(x_hat, 3) * y_hat * np.power(z_hat, 3) + 2027025.0 * np.power(x_hat, 3) * y_hat * z_hat -6081075.0 * x_hat * y_hat * np.power(z_hat, 5) + 4054050.0 * x_hat * y_hat * np.power(z_hat, 3) -467775.0 * x_hat * y_hat * z_hat 
        if n == "306":
            return  34459425.0 * np.power(x_hat, 3) * np.power(z_hat, 6) -30405375.0 * np.power(x_hat, 3) * np.power(z_hat, 4) + 6081075.0 * np.power(x_hat, 3) * np.power(z_hat, 2) -155925.0 * np.power(x_hat, 3) -6081075.0 * x_hat * np.power(z_hat, 6) + 6081075.0 * x_hat * np.power(z_hat, 4) -1403325.0 * x_hat * np.power(z_hat, 2) + 42525.0 * x_hat 
        if n == "270":
            return  34459425.0 * np.power(x_hat, 2) * np.power(y_hat, 7) -42567525.0 * np.power(x_hat, 2) * np.power(y_hat, 5) + 14189175.0 * np.power(x_hat, 2) * np.power(y_hat, 3) -1091475.0 * np.power(x_hat, 2) * y_hat -2027025.0 * np.power(y_hat, 7) + 2837835.0 * np.power(y_hat, 5) -1091475.0 * np.power(y_hat, 3) + 99225.0 * y_hat 
        if n == "261":
            return  34459425.0 * np.power(x_hat, 2) * np.power(y_hat, 6) * z_hat -30405375.0 * np.power(x_hat, 2) * np.power(y_hat, 4) * z_hat + 6081075.0 * np.power(x_hat, 2) * np.power(y_hat, 2) * z_hat -155925.0 * np.power(x_hat, 2) * z_hat -2027025.0 * np.power(y_hat, 6) * z_hat + 2027025.0 * np.power(y_hat, 4) * z_hat -467775.0 * np.power(y_hat, 2) * z_hat + 14175.0 * z_hat 
        if n == "252":
            return  34459425.0 * np.power(x_hat, 2) * np.power(y_hat, 5) * np.power(z_hat, 2) -2027025.0 * np.power(x_hat, 2) * np.power(y_hat, 5) -20270250.0 * np.power(x_hat, 2) * np.power(y_hat, 3) * np.power(z_hat, 2) + 1351350.0 * np.power(x_hat, 2) * np.power(y_hat, 3) + 2027025.0 * np.power(x_hat, 2) * y_hat * np.power(z_hat, 2) -155925.0 * np.power(x_hat, 2) * y_hat -2027025.0 * np.power(y_hat, 5) * np.power(z_hat, 2) + 135135.0 * np.power(y_hat, 5) + 1351350.0 * np.power(y_hat, 3) * np.power(z_hat, 2) -103950.0 * np.power(y_hat, 3) -155925.0 * y_hat * np.power(z_hat, 2) + 14175.0 * y_hat 
        if n == "243":
            return  34459425.0 * np.power(x_hat, 2) * np.power(y_hat, 4) * np.power(z_hat, 3) -6081075.0 * np.power(x_hat, 2) * np.power(y_hat, 4) * z_hat -12162150.0 * np.power(x_hat, 2) * np.power(y_hat, 2) * np.power(z_hat, 3) + 2432430.0 * np.power(x_hat, 2) * np.power(y_hat, 2) * z_hat + 405405.0 * np.power(x_hat, 2) * np.power(z_hat, 3) -93555.0 * np.power(x_hat, 2) * z_hat -2027025.0 * np.power(y_hat, 4) * np.power(z_hat, 3) + 405405.0 * np.power(y_hat, 4) * z_hat + 810810.0 * np.power(y_hat, 2) * np.power(z_hat, 3) -187110.0 * np.power(y_hat, 2) * z_hat -31185.0 * np.power(z_hat, 3) + 8505.0 * z_hat 
        if n == "234":
            return  34459425.0 * np.power(x_hat, 2) * np.power(y_hat, 3) * np.power(z_hat, 4) -12162150.0 * np.power(x_hat, 2) * np.power(y_hat, 3) * np.power(z_hat, 2) + 405405.0 * np.power(x_hat, 2) * np.power(y_hat, 3) -6081075.0 * np.power(x_hat, 2) * y_hat * np.power(z_hat, 4) + 2432430.0 * np.power(x_hat, 2) * y_hat * np.power(z_hat, 2) -93555.0 * np.power(x_hat, 2) * y_hat -2027025.0 * np.power(y_hat, 3) * np.power(z_hat, 4) + 810810.0 * np.power(y_hat, 3) * np.power(z_hat, 2) -31185.0 * np.power(y_hat, 3) + 405405.0 * y_hat * np.power(z_hat, 4) -187110.0 * y_hat * np.power(z_hat, 2) + 8505.0 * y_hat 
        if n == "225":
            return  34459425.0 * np.power(x_hat, 2) * np.power(y_hat, 2) * np.power(z_hat, 5) -20270250.0 * np.power(x_hat, 2) * np.power(y_hat, 2) * np.power(z_hat, 3) + 2027025.0 * np.power(x_hat, 2) * np.power(y_hat, 2) * z_hat -2027025.0 * np.power(x_hat, 2) * np.power(z_hat, 5) + 1351350.0 * np.power(x_hat, 2) * np.power(z_hat, 3) -155925.0 * np.power(x_hat, 2) * z_hat -2027025.0 * np.power(y_hat, 2) * np.power(z_hat, 5) + 1351350.0 * np.power(y_hat, 2) * np.power(z_hat, 3) -155925.0 * np.power(y_hat, 2) * z_hat + 135135.0 * np.power(z_hat, 5) -103950.0 * np.power(z_hat, 3) + 14175.0 * z_hat 
        if n == "216":
            return  34459425.0 * np.power(x_hat, 2) * y_hat * np.power(z_hat, 6) -30405375.0 * np.power(x_hat, 2) * y_hat * np.power(z_hat, 4) + 6081075.0 * np.power(x_hat, 2) * y_hat * np.power(z_hat, 2) -155925.0 * np.power(x_hat, 2) * y_hat -2027025.0 * y_hat * np.power(z_hat, 6) + 2027025.0 * y_hat * np.power(z_hat, 4) -467775.0 * y_hat * np.power(z_hat, 2) + 14175.0 * y_hat 
        if n == "207":
            return  34459425.0 * np.power(x_hat, 2) * np.power(z_hat, 7) -42567525.0 * np.power(x_hat, 2) * np.power(z_hat, 5) + 14189175.0 * np.power(x_hat, 2) * np.power(z_hat, 3) -1091475.0 * np.power(x_hat, 2) * z_hat -2027025.0 * np.power(z_hat, 7) + 2837835.0 * np.power(z_hat, 5) -1091475.0 * np.power(z_hat, 3) + 99225.0 * z_hat 
        if n == "180":
            return  34459425.0 * x_hat * np.power(y_hat, 8) -56756700.0 * x_hat * np.power(y_hat, 6) + 28378350.0 * x_hat * np.power(y_hat, 4) -4365900.0 * x_hat * np.power(y_hat, 2) + 99225.0 * x_hat 
        if n == "171":
            return  34459425.0 * x_hat * np.power(y_hat, 7) * z_hat -42567525.0 * x_hat * np.power(y_hat, 5) * z_hat + 14189175.0 * x_hat * np.power(y_hat, 3) * z_hat -1091475.0 * x_hat * y_hat * z_hat 
        if n == "162":
            return  34459425.0 * x_hat * np.power(y_hat, 6) * np.power(z_hat, 2) -2027025.0 * x_hat * np.power(y_hat, 6) -30405375.0 * x_hat * np.power(y_hat, 4) * np.power(z_hat, 2) + 2027025.0 * x_hat * np.power(y_hat, 4) + 6081075.0 * x_hat * np.power(y_hat, 2) * np.power(z_hat, 2) -467775.0 * x_hat * np.power(y_hat, 2) -155925.0 * x_hat * np.power(z_hat, 2) + 14175.0 * x_hat 
        if n == "153":
            return  34459425.0 * x_hat * np.power(y_hat, 5) * np.power(z_hat, 3) -6081075.0 * x_hat * np.power(y_hat, 5) * z_hat -20270250.0 * x_hat * np.power(y_hat, 3) * np.power(z_hat, 3) + 4054050.0 * x_hat * np.power(y_hat, 3) * z_hat + 2027025.0 * x_hat * y_hat * np.power(z_hat, 3) -467775.0 * x_hat * y_hat * z_hat 
        if n == "144":
            return  34459425.0 * x_hat * np.power(y_hat, 4) * np.power(z_hat, 4) -12162150.0 * x_hat * np.power(y_hat, 4) * np.power(z_hat, 2) + 405405.0 * x_hat * np.power(y_hat, 4) -12162150.0 * x_hat * np.power(y_hat, 2) * np.power(z_hat, 4) + 4864860.0 * x_hat * np.power(y_hat, 2) * np.power(z_hat, 2) -187110.0 * x_hat * np.power(y_hat, 2) + 405405.0 * x_hat * np.power(z_hat, 4) -187110.0 * x_hat * np.power(z_hat, 2) + 8505.0 * x_hat 
        if n == "135":
            return  34459425.0 * x_hat * np.power(y_hat, 3) * np.power(z_hat, 5) -20270250.0 * x_hat * np.power(y_hat, 3) * np.power(z_hat, 3) + 2027025.0 * x_hat * np.power(y_hat, 3) * z_hat -6081075.0 * x_hat * y_hat * np.power(z_hat, 5) + 4054050.0 * x_hat * y_hat * np.power(z_hat, 3) -467775.0 * x_hat * y_hat * z_hat 
        if n == "126":
            return  34459425.0 * x_hat * np.power(y_hat, 2) * np.power(z_hat, 6) -30405375.0 * x_hat * np.power(y_hat, 2) * np.power(z_hat, 4) + 6081075.0 * x_hat * np.power(y_hat, 2) * np.power(z_hat, 2) -155925.0 * x_hat * np.power(y_hat, 2) -2027025.0 * x_hat * np.power(z_hat, 6) + 2027025.0 * x_hat * np.power(z_hat, 4) -467775.0 * x_hat * np.power(z_hat, 2) + 14175.0 * x_hat 
        if n == "117":
            return  34459425.0 * x_hat * y_hat * np.power(z_hat, 7) -42567525.0 * x_hat * y_hat * np.power(z_hat, 5) + 14189175.0 * x_hat * y_hat * np.power(z_hat, 3) -1091475.0 * x_hat * y_hat * z_hat 
        if n == "108":
            return  34459425.0 * x_hat * np.power(z_hat, 8) -56756700.0 * x_hat * np.power(z_hat, 6) + 28378350.0 * x_hat * np.power(z_hat, 4) -4365900.0 * x_hat * np.power(z_hat, 2) + 99225.0 * x_hat 
        if n == "090":
            return  34459425.0 * np.power(y_hat, 9) -72972900.0 * np.power(y_hat, 7) + 51081030.0 * np.power(y_hat, 5) -13097700.0 * np.power(y_hat, 3) + 893025.0 * y_hat 
        if n == "081":
            return  34459425.0 * np.power(y_hat, 8) * z_hat -56756700.0 * np.power(y_hat, 6) * z_hat + 28378350.0 * np.power(y_hat, 4) * z_hat -4365900.0 * np.power(y_hat, 2) * z_hat + 99225.0 * z_hat 
        if n == "072":
            return  34459425.0 * np.power(y_hat, 7) * np.power(z_hat, 2) -2027025.0 * np.power(y_hat, 7) -42567525.0 * np.power(y_hat, 5) * np.power(z_hat, 2) + 2837835.0 * np.power(y_hat, 5) + 14189175.0 * np.power(y_hat, 3) * np.power(z_hat, 2) -1091475.0 * np.power(y_hat, 3) -1091475.0 * y_hat * np.power(z_hat, 2) + 99225.0 * y_hat 
        if n == "063":
            return  34459425.0 * np.power(y_hat, 6) * np.power(z_hat, 3) -6081075.0 * np.power(y_hat, 6) * z_hat -30405375.0 * np.power(y_hat, 4) * np.power(z_hat, 3) + 6081075.0 * np.power(y_hat, 4) * z_hat + 6081075.0 * np.power(y_hat, 2) * np.power(z_hat, 3) -1403325.0 * np.power(y_hat, 2) * z_hat -155925.0 * np.power(z_hat, 3) + 42525.0 * z_hat 
        if n == "054":
            return  34459425.0 * np.power(y_hat, 5) * np.power(z_hat, 4) -12162150.0 * np.power(y_hat, 5) * np.power(z_hat, 2) + 405405.0 * np.power(y_hat, 5) -20270250.0 * np.power(y_hat, 3) * np.power(z_hat, 4) + 8108100.0 * np.power(y_hat, 3) * np.power(z_hat, 2) -311850.0 * np.power(y_hat, 3) + 2027025.0 * y_hat * np.power(z_hat, 4) -935550.0 * y_hat * np.power(z_hat, 2) + 42525.0 * y_hat 
        if n == "045":
            return  34459425.0 * np.power(y_hat, 4) * np.power(z_hat, 5) -20270250.0 * np.power(y_hat, 4) * np.power(z_hat, 3) + 2027025.0 * np.power(y_hat, 4) * z_hat -12162150.0 * np.power(y_hat, 2) * np.power(z_hat, 5) + 8108100.0 * np.power(y_hat, 2) * np.power(z_hat, 3) -935550.0 * np.power(y_hat, 2) * z_hat + 405405.0 * np.power(z_hat, 5) -311850.0 * np.power(z_hat, 3) + 42525.0 * z_hat 
        if n == "036":
            return  34459425.0 * np.power(y_hat, 3) * np.power(z_hat, 6) -30405375.0 * np.power(y_hat, 3) * np.power(z_hat, 4) + 6081075.0 * np.power(y_hat, 3) * np.power(z_hat, 2) -155925.0 * np.power(y_hat, 3) -6081075.0 * y_hat * np.power(z_hat, 6) + 6081075.0 * y_hat * np.power(z_hat, 4) -1403325.0 * y_hat * np.power(z_hat, 2) + 42525.0 * y_hat 
        if n == "027":
            return  34459425.0 * np.power(y_hat, 2) * np.power(z_hat, 7) -42567525.0 * np.power(y_hat, 2) * np.power(z_hat, 5) + 14189175.0 * np.power(y_hat, 2) * np.power(z_hat, 3) -1091475.0 * np.power(y_hat, 2) * z_hat -2027025.0 * np.power(z_hat, 7) + 2837835.0 * np.power(z_hat, 5) -1091475.0 * np.power(z_hat, 3) + 99225.0 * z_hat 
        if n == "018":
            return  34459425.0 * y_hat * np.power(z_hat, 8) -56756700.0 * y_hat * np.power(z_hat, 6) + 28378350.0 * y_hat * np.power(z_hat, 4) -4365900.0 * y_hat * np.power(z_hat, 2) + 99225.0 * y_hat 
        if n == "009":
            return  34459425.0 * np.power(z_hat, 9) -72972900.0 * np.power(z_hat, 7) + 51081030.0 * np.power(z_hat, 5) -13097700.0 * np.power(z_hat, 3) + 893025.0 * z_hat         
 

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations

def draw(hx,hy,hz,dim_x,dim_y,dim_z,r,U):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax.set_aspect("equal")

    # draw cube
    #r = [-1, 1]
    #for s, e in combinations(np.array(list(product(r, r, r))), 2):
    #    if np.sum(np.abs(s-e)) == r[1]-r[0]:
    #        ax.plot3D(*zip(s, e), color="b")
    
            
    ref_x_min = - hx * dim_x * 0.5
    ref_x_max = hx * dim_x * 0.5
    ref_y_min = - hy * dim_y * 0.5
    ref_y_max = hy * dim_y * 0.5
    ref_z_min = - hz * dim_z * 0.5
    ref_z_max = hz * dim_z * 0.5
    
    for s, e in combinations(np.array(list(product([ref_x_min, ref_x_max], [ref_y_min, ref_y_max], [ref_z_min, ref_z_max]))), 2):
        temp_s = np.matmul(s, U)
        temp_e = np.matmul(e, U)
        ax.plot3D(*zip(temp_s, temp_e), color="b")

    # draw sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = r * np.cos(u)*np.sin(v)
    y = r * np.sin(u)*np.sin(v)
    z = r * np.cos(v)
    ax.plot_wireframe(x, y, z, color="r")

    # draw a point
    ax.scatter([0], [0], [0], color="g", s=100)

    #ax.add_artist(a)
    plt.show()

    return



"""
new function
"""


def calc_MC_surface_harmonic_stencil_n(hx, hy, hz, r, l, n, accuracy = 5, U = None):
    
    if U is None:
        U = np.array([[1, 0, 0],[0, 1, 0], [0, 0, 1]])
    else:
        U = normalize(U)

    # initialize the stencil with right dimensions
    #dim_x = int(2.* math.ceil( r/hx )) + 1
    #dim_y = int(2.* math.ceil( r/hy )) + 1
    #dim_z = int(2.* math.ceil( r/hz )) + 1
    
    dim_x, dim_y, dim_z = get_dimensions_plane(hx,hy,hz,r,U)
    
    #dv = hx*hy*hz / (float(accuracy) ** 3)
    
    l1 = (hx / float(accuracy - 1)) * U[0,:]
    l2 = (hy / float(accuracy - 1)) * U[1,:]
    l3 = (hz / float(accuracy - 1)) * U[2,:]
    dv = np.abs(np.dot(l1, np.cross(l2, l3)))

    print(dim_x, dim_y, dim_z)

    
    x,y,z = get_eval_coordinates(dim_x, dim_y, dim_z, hx, hy, hz, accuracy, U)
    r_array = np.sqrt(np.square(x) + np.square(y) + np.square(z))

    
    eval_result = MC_surface_spherical_harmonic_n_np(x, y, z, l, n, r_array)
    eval_result_masked = np.multiply(np.where(r_array > r, 0, eval_result) , dv)
    #plot(x,y,z,eval_result_masked)
    #stencil = eval_result
    stencil = np.sum(eval_result_masked, axis = -1)
    #plot(np.mean(x, axis = -1),np.mean(y, axis = -1),np.mean(z, axis = -1),stencil)
    
    padx = int(math.ceil(float(dim_x)/2.))
    pady = int(math.ceil(float(dim_y)/2.))
    padz = int(math.ceil(float(dim_z)/2.))
    
    pad = (padx,pady,padz)
    
    print("stencil shape")
    print(stencil.shape)
    
  
    return stencil, 

def gaussian(x, mu, sig):
    A = 1/(sig * sqrt(2*np.pi))
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def calc_MC_surface_harmonic_legendre_gaussian_stencil_n(hx, hy, hz, r, l, n, legendre_order, sigma, accuracy = 5, U = None):
    
    if U is None:
        U = np.array([[1, 0, 0],[0, 1, 0], [0, 0, 1]])
    else:
        U = normalize(U)

    # initialize the stencil with right dimensions
    #dim_x = int(2.* math.ceil( r/hx )) + 1
    #dim_y = int(2.* math.ceil( r/hy )) + 1
    #dim_z = int(2.* math.ceil( r/hz )) + 1
    
    dim_x, dim_y, dim_z = get_dimensions_plane(hx,hy,hz,r,U)
    
    #dv = hx*hy*hz / (float(accuracy) ** 3)
    
    l1 = (hx / float(accuracy - 1)) * U[0,:]
    l2 = (hy / float(accuracy - 1)) * U[1,:]
    l3 = (hz / float(accuracy - 1)) * U[2,:]
    dv = np.abs(np.dot(l1, np.cross(l2, l3)))

    print(dim_x, dim_y, dim_z)

    
    x,y,z = get_eval_coordinates(dim_x, dim_y, dim_z, hx, hy, hz, accuracy, U)
    r_array = np.sqrt(np.square(x) + np.square(y) + np.square(z))

    Pn = legendre(legendre_order)
    legendre_array = pn((2*r_array-r)/r)

    gaussian_array = gaussian(r_array, 0, sigma)

    
    eval_result = MC_surface_spherical_harmonic_n_np(x, y, z, l, n, r_array)
    eval_result = np.multiply(np.multiply(eval_result, gaussian_array), legendre_array)

    eval_result_masked = np.multiply(np.where(r_array > r, 0, eval_result) , dv)


    #plot(x,y,z,eval_result_masked)
    #stencil = eval_result
    stencil = np.sum(eval_result_masked, axis = -1)
    #plot(np.mean(x, axis = -1),np.mean(y, axis = -1),np.mean(z, axis = -1),stencil)
    
    padx = int(math.ceil(float(dim_x)/2.))
    pady = int(math.ceil(float(dim_y)/2.))
    padz = int(math.ceil(float(dim_z)/2.))
    
    pad = (padx,pady,padz)
    
    print("stencil shape")
    print(stencil.shape)
    
  
    return stencil, pad




def U_transform(x,y,z,U):
    original = np.array([[x,y,z]])
    transformed = np.matmul(original, U)
    return transformed[0][0], transformed[0][1], transformed[0][2]

def get_dimensions_plane(hx,hy,hz,r,U):
    #U = normalize(np.array([[1, 0, 0],[0, 1, 0], [0, 0, 1]]))
    #U = normalize(np.array([[1, 1, 0],[1, 0, 1], [0, 1, 1]]))
    
    dim_x = int(2.* math.ceil( r/hx )) + 1
    dim_y = int(2.* math.ceil( r/hy )) + 1
    dim_z = int(2.* math.ceil( r/hz )) + 1
    
    print("start: {}\t {}\t {}".format(dim_x,dim_y,dim_z))
    
    while True:
        ref_x_min = - hx * dim_x * 0.5
        ref_x_max = hx * dim_x * 0.5
        ref_y_min = - hy * dim_y * 0.5
        ref_y_max = hy * dim_y * 0.5
        ref_z_min = - hz * dim_z * 0.5
        ref_z_max = hz * dim_z * 0.5
        
        p1x_1 = U_transform(ref_x_min,ref_y_min, ref_z_min,U)
        p2x_1 = U_transform(ref_x_min,ref_y_max, ref_z_min,U)
        p3x_1 = U_transform(ref_x_min,ref_y_min, ref_z_max,U)
        
        p1x_2 = U_transform(ref_x_max,ref_y_min, ref_z_min,U)
        p2x_2 = U_transform(ref_x_max,ref_y_max, ref_z_min,U)
        p3x_2 = U_transform(ref_x_max,ref_y_min, ref_z_max,U)
            
        if check_sphere_intersect_with_sphere(p1x_1, p2x_1, p3x_1, r) or \
        check_sphere_intersect_with_sphere(p1x_2, p2x_2, p3x_2, r):
            dim_x += 2
        else:
            break    
            
    while True:
        ref_x_min = - hx * dim_x * 0.5
        ref_x_max = hx * dim_x * 0.5
        ref_y_min = - hy * dim_y * 0.5
        ref_y_max = hy * dim_y * 0.5
        ref_z_min = - hz * dim_z * 0.5
        ref_z_max = hz * dim_z * 0.5
        
        p1y_1 = U_transform(ref_x_min,ref_y_min, ref_z_min,U)
        p2y_1 = U_transform(ref_x_max,ref_y_min, ref_z_min,U)
        p3y_1 = U_transform(ref_x_min,ref_y_min, ref_z_max,U)
        
        p1y_2 = U_transform(ref_x_min,ref_y_max, ref_z_min,U)
        p2y_2 = U_transform(ref_x_max,ref_y_max, ref_z_min,U)
        p3y_2 = U_transform(ref_x_min,ref_y_max, ref_z_max,U)
            
        if check_sphere_intersect_with_sphere(p1y_1, p2y_1, p3y_1, r) or \
        check_sphere_intersect_with_sphere(p1y_2, p2y_2, p3y_2, r):
            dim_y += 2
        else:
            break  
            
    while True:
        ref_x_min = - hx * dim_x * 0.5
        ref_x_max = hx * dim_x * 0.5
        ref_y_min = - hy * dim_y * 0.5
        ref_y_max = hy * dim_y * 0.5
        ref_z_min = - hz * dim_z * 0.5
        ref_z_max = hz * dim_z * 0.5
        
        p1z_1 = U_transform(ref_x_min,ref_y_min, ref_z_min,U)
        p2z_1 = U_transform(ref_x_max,ref_y_min, ref_z_min,U)
        p3z_1 = U_transform(ref_x_min,ref_y_max, ref_z_min,U)
        
        p1z_2 = U_transform(ref_x_min,ref_y_min, ref_z_max,U)
        p2z_2 = U_transform(ref_x_max,ref_y_min, ref_z_max,U)
        p3z_2 = U_transform(ref_x_min,ref_y_max, ref_z_max,U)
            
        if check_sphere_intersect_with_sphere(p1z_1, p2z_1, p3z_1, r) or \
        check_sphere_intersect_with_sphere(p1z_2, p2z_2, p3z_2, r):
            dim_z += 2
        else:
            break 
        
    
    #print("result: {}\t {}\t {}".format(dim_x,dim_y,dim_z))
    
    #draw(hx,hy,hz,dim_x,dim_y,dim_z,r,U)
    return dim_x,dim_y,dim_z


    

def get_eval_coordinates(dim_x, dim_y, dim_z, hx, hy, hz, accuracy, U ):
    
    
    ref_x_min = - hx * 0.5
    ref_x_max = hx * 0.5
    ref_y_min = - hy * 0.5
    ref_y_max = hy * 0.5
    ref_z_min = - hz * 0.5
    ref_z_max = hz * 0.5
    
    ref_x_li = np.linspace(ref_x_min, ref_x_max, num=accuracy)
    ref_y_li = np.linspace(ref_y_min, ref_y_max, num=accuracy)
    ref_z_li = np.linspace(ref_z_min, ref_z_max, num=accuracy)
    
    ref_coord_list = list(itertools.product(ref_x_li,ref_y_li,ref_z_li))
    ref_coord_array = np.array(ref_coord_list)

    ref_temp_x = ref_coord_array[:,0]
    ref_temp_y = ref_coord_array[:,1]
    ref_temp_z = ref_coord_array[:,2]
    
    
    center_x = int((dim_x - 1)/2) 
    center_y = int((dim_y - 1)/2)
    center_z = int((dim_z - 1)/2)

    num_eval_per_cell = accuracy ** 3


    x = [[[[] for i in range(dim_z)] for j in range(dim_y)] for k in range(dim_x)]
    y = [[[[] for i in range(dim_z)] for j in range(dim_y)] for k in range(dim_x)]
    z = [[[[] for i in range(dim_z)] for j in range(dim_y)] for k in range(dim_x)]

    print(np.array(x).shape)

    for i in range(dim_x):
        for j in range(dim_y):
            for k in range(dim_z):
                x_offset = float(i-center_x) * hx
                y_offset = float(j-center_y) * hy
                z_offset = float(k-center_z) * hz
                
                
                temp_coord_array = np.column_stack((ref_temp_x + x_offset, ref_temp_y + y_offset, ref_temp_z + z_offset))
                temp = np.matmul(temp_coord_array, U)
                x[i][j][k] = temp[:,0]
                y[i][j][k] = temp[:,1]
                z[i][j][k] = temp[:,2]
                
                
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    

    return x,y,z
import numpy as np
import sys
import math

from convolutions import get_differenciation_conv, get_integration_stencil,get_auto_accuracy,get_fftconv_with_known_stencil_no_wrap,get_asym_integration_stencil,get_asym_integration_fftconv,get_asym_integral_fftconv_with_known_stencil
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def pad(array, reference_shape):

	result = np.zeros((reference_shape, reference_shape))

	array_shape = array.shape
	x_offset = (reference_shape - array_shape[0])/2
	y_offset = (reference_shape - array_shape[1])/2

	result[x_offset: array_shape[0] + x_offset, y_offset: array_shape[1] + y_offset] = array

	return result

h = 0.02
r_list = [0.6, 0.1, 0.14]
stencil_list = []

stencil_dict = {}
pad_list = []
for r in r_list:
    temp_stencil,temp_pad = get_integration_stencil(h, h, h, r, accuracy = get_auto_accuracy(h,h,h, r))
    stencil_list.append(temp_stencil[(temp_stencil.shape[1]-1)/2])
    stencil_dict[r] = temp_stencil[(temp_stencil.shape[1]-1)/2]
    pad_list.append(temp_pad)


for key in stencil_dict:
	plt.figure()
	sns.heatmap(pad(stencil_dict[r],21), cmap="RdBu_r", center=0.)
	plt.savefig("integration_stencil_plot_{}.png".format(r))


'''
test = np.random.rand(3,3)

sns.heatmap(test, cmap="PiYG", center=0.)
sns.plt.show()

sns.heatmap(pad(test,7), cmap="PiYG", center=0.)
sns.plt.show()

sns.heatmap(pad(test,21), cmap="PiYG", center=0.)
sns.plt.show()
'''
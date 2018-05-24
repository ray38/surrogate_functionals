import numpy as np
import sys
import math

from convolutions import get_differenciation_conv_stencil, get_differenciation_conv, get_integration_stencil,get_auto_accuracy,get_fftconv_with_known_stencil_no_wrap,get_asym_integration_stencil,get_asym_integration_fftconv,get_asym_integral_fftconv_with_known_stencil
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


def plot(array, reference_shape, filename):
	plt.figure()
	sns.heatmap(pad(array,reference_shape), cmap="RdBu_r", center=0., yticklabels=False, xticklabels=False, cbar=False)
	plt.savefig(filename)
	return


h = 0.02
r_list = [0.06, 0.1, 0.14]
stencil_list = []

stencil_dict = {}
pad_list = []
for r in r_list:
    temp_stencil,temp_pad = get_integration_stencil(h, h, h, r, accuracy = get_auto_accuracy(h,h,h, r))
    temp_stencil= temp_stencil / (0.02*0.02*0.02)
    stencil_list.append(temp_stencil[(temp_stencil.shape[1]-1)/2])
    stencil_dict[r] = temp_stencil[(temp_stencil.shape[1]-1)/2]
    pad_list.append(temp_pad)


for key in stencil_dict:
	plot(stencil_dict[key], 21, "integration_stencil_plot_{}.png".format(key))


temp_first_deri, temp_pad = get_differenciation_conv_stencil(h, h, h, gradient = 'first',
                                               stencil_type = 'mid', accuracy = '2')

plot(temp_first_deri[(temp_first_deri.shape[1]-1)/2], 21, "gradient_stencil_plot_1.png")


temp_sec_deri, temp_pad   = get_differenciation_conv_stencil(h, h, h, gradient = 'second',
                                       stencil_type = 'times2', accuracy = '2')

plot(temp_sec_deri[(temp_sec_deri.shape[1]-1)/2], 21, "gradient_stencil_plot_2.png")



temp_third_deri, temp_pad = get_differenciation_conv_stencil(h, h, h, gradient = 'third',
                                       stencil_type = 'times2', accuracy = '2')

plot(temp_third_deri[(temp_third_deri.shape[1]-1)/2], 21, "gradient_stencil_plot_3.png")

'''
test = np.random.rand(3,3)

sns.heatmap(test, cmap="PiYG", center=0.)
sns.plt.show()

sns.heatmap(pad(test,7), cmap="PiYG", center=0.)
sns.plt.show()

sns.heatmap(pad(test,21), cmap="PiYG", center=0.)
sns.plt.show()
'''
import numpy as np
import sys
import math

from convolutions import get_differenciation_conv, get_integration_stencil,get_auto_accuracy,get_fftconv_with_known_stencil_no_wrap,get_asym_integration_stencil,get_asym_integration_fftconv,get_asym_integral_fftconv_with_known_stencil
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt



h = 0.02
r_list = [0.1, 0.2, 0.3]
stencil_list = []
pad_list = []
for r in r_list:
    temp_stencil,temp_pad = get_integration_stencil(h, h, h, r, accuracy = get_auto_accuracy(h,h,h, r))
    stencil_list.append(temp_stencil[(temp_stencil.shape[1]-1)/2])
    pad_list.append(temp_pad)
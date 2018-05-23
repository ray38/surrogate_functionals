import os

import sys
import csv
try: import cPickle as pickle
except: import pickle
import time
import pprint
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt




if __name__ == "__main__":

	filename = sys.argv[1]

	with open(filename, 'rb') as handle:
		data = pickle.load(handle)


	sns.set(style="whitegrid", palette="pastel", color_codes=True)

    # Draw a nested violinplot and split the violins for easier comparison
    sns.violinplot(x="model_name",y="formation_exc_error",hue="training_test",data=data,split=True,inner="quartile",palette={"training":"b","test":"y"})
    sns.despine(left=True)

    plt.savefig("formation_energy_grouped_violin_plot.png")

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
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
import json




if __name__ == "__main__":


	pkmn_type_colors = ['#78C850',  # Grass
	                    '#F08030',  # Fire
	                    '#6890F0',  # Water
	                    '#A8B820',  # Bug
	                    '#A8A878',  # Normal
	                    '#A040A0',  # Poison
	                    '#F8D030',  # Electric
	                    '#E0C068',  # Ground
	                    '#EE99AC',  # Fairy
	                    '#C03028',  # Fighting
	                    '#F85888',  # Psychic
	                    '#B8A038',  # Rock
	                    '#705898',  # Ghost
	                    '#98D8D8',  # Ice
	                    '#7038F8',  # Dragon
	                   ]



	filename = sys.argv[1]
	order_filename = sys.argv[2]

	with open(filename, 'rb') as handle:
		data = pickle.load(handle)

	with open(order_filename) as f:
		temp_order = json.load(f)
	order = temp_order["order"]

	print data








	plt.figure()

	sns.set(style="whitegrid", palette="pastel", color_codes=True)

	ax1 = sns.violinplot(x="model_name",y="formation_exc_error",hue="training_test",order = order,data=data,split=True,inner=None,palette={"training":"b","test":"y"})
	sns.despine(left=True)
	ax2 = sns.swarmplot(x="model_name",y="formation_exc_error",data=data, hue='training_test',order = order, split=True, color='k', alpha=0.7, palette=("Dark2"))
	ax1.set_xticklabels(ax1.get_xticklabels(),rotation=45)
	ax2.set_xticklabels(ax2.get_xticklabels(),rotation=45)
	plt.xlabel("Model", fontsize=25)
	plt.ylabel("Error in Formation Energy (eV)", fontsize=25)
	plt.tick_params('both',labelsize='15')

	handles1, _ = ax1.get_legend_handles_labels()
	ax1.legend(handles1, ["Training set", "Test set"],fontsize=20,bbox_to_anchor=(0.6, 0.4), loc=2)

	plt.tight_layout()
	plt.savefig("formation_energy_violin_swarm_plot2.png")







	plt.figure()

	sns.set(style="whitegrid", palette="pastel", color_codes=True)

	ax1 = sns.violinplot(x="model_name",y="exc_error",hue="training_test",order = order,data=data,split=True,inner=None,palette={"training":"b","test":"y"})
	sns.despine(left=True)
	ax2 = sns.swarmplot(x="model_name",y="exc_error",data=data, hue='training_test',order = order, split=True, color='k', alpha=0.7, palette=("Dark2"))
	ax1.set_xticklabels(ax1.get_xticklabels(),rotation=45)
	ax2.set_xticklabels(ax2.get_xticklabels(),rotation=45)
	plt.xlabel("Model", fontsize=25)
	plt.ylabel("Error in Absolute Energy (eV)", fontsize=25)
	plt.tick_params('both',labelsize='15')

	handles1, _ = ax1.get_legend_handles_labels()
	ax1.legend(handles1, ["Training set", "Test set"],fontsize=20,bbox_to_anchor=(0.6, 0.4), loc=2)

	plt.tight_layout()
	plt.savefig("energy_violin_swarm_plot2.png")





















	plt.figure()
	sns.set(style="whitegrid", palette="pastel", color_codes=True)
	sns.set_context("poster")
	g = sns.factorplot(x="molecule_name", y="formation_exc_error", hue="model_name",hue_order = order, data=data,
                   capsize=.2, palette="YlGnBu_d", size=12, aspect=1.5,legend=False, order = ["C2H2","C2H4","C2H6","CH3OH","CH4","CO","CO2","H2","H2O","HCN","HNC","N2","N2O","NH3","O3","CH3CN","CH3CHO","CH3NO2","glycine","H2CCO","H2CO","H2O2","HCOOH","N2H4","NCCN"])
        g.set_xticklabels(rotation=90,fontsize=30)
        g.set_yticklabels(fontsize=40)
	sns.despine(left=True)
	#plt.legend(bbox_to_anchor=(1, 1), loc=2)
	plt.legend(bbox_to_anchor=(0, 0.4), loc=2,fontsize=25)
	plt.ylabel("Error in Formation Energy (eV)", fontsize=40)
	plt.xlabel("", fontsize=40)
	#g.legend( ["Model 1", "Model 2", "Model 3", "Model 4", "Model 5"],loc='upper right',fontsize=15)
	plt.tight_layout()
	plt.savefig("formation_factor_plot.png")




	plt.figure()
	sns.set(style="whitegrid", palette="pastel", color_codes=True)
	sns.set_context("poster")
	g = sns.factorplot(x="molecule_name", y="exc_error", hue="model_name",hue_order = order, data=data,
                   capsize=.2, palette="YlGnBu_d", size=12,aspect=1.5, legend=False, order = ["C2H2","C2H4","C2H6","CH3OH","CH4","CO","CO2","H2","H2O","HCN","HNC","N2","N2O","NH3","O3","CH3CN","CH3CHO","CH3NO2","glycine","H2CCO","H2CO","H2O2","HCOOH","N2H4","NCCN"])
        g.set_xticklabels(rotation=90,fontsize=30)
        g.set_yticklabels(fontsize=40)
	sns.despine(left=True)
	#plt.legend(bbox_to_anchor=(1, 1), loc=2)
	plt.legend(bbox_to_anchor=(0, 0.0), loc=2,fontsize=25)
	plt.ylabel("Error in Absolute Energy (eV)", fontsize=40)
	plt.xlabel("", fontsize=18)
	plt.tight_layout()
	plt.savefig("energy_factor_plot.png")
	
	
	
	


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

	with open(filename, 'rb') as handle:
		data = pickle.load(handle)

	print data




	sns.set(style="whitegrid", palette="pastel", color_codes=True)

	ax = sns.violinplot(x="model_name",y="formation_exc_error",hue="training_test",data=data,split=True,inner="quartile",palette={"training":"b","test":"y"})
	ax.set_xticklabels(rotation=45)
	sns.despine(left=True)
	
	plt.xlabel("Model", fontsize=18)
	plt.ylabel("Error in Formation Energy (eV)", fontsize=18)
	plt.tick_params('both',labelsize='15')

	handles, _ = ax.get_legend_handles_labels()
	ax.legend(handles, ["Training set", "Test set"],fontsize=15)
	plt.tight_layout()
	plt.savefig("formation_energy_grouped_violin_plot.png")



	plt.figure()
	
	sns.set(style="whitegrid", palette="pastel", color_codes=True)

	ax = sns.violinplot(x="model_name",y="exc_error",hue="training_test",data=data,split=True,inner="quartile",palette={"training":"b","test":"y"})
	ax.set_xticklabels(rotation=45)
	sns.despine(left=True)
	plt.xlabel("Model", fontsize=18)
	plt.ylabel("Error in Absolute Energy (eV)", fontsize=18)
	plt.tick_params('both',labelsize='15')

	handles, _ = ax.get_legend_handles_labels()
	ax.legend(handles, ["Training set", "Test set"],fontsize=15)
	plt.tight_layout()
	plt.savefig("energy_grouped_violin_plot.png")










	plt.figure()

	sns.set(style="whitegrid", palette="pastel", color_codes=True)

	ax = sns.violinplot(x="model_name",y="formation_exc_error",hue="training_test",data=data,split=True,inner="quartile",palette={"training":"b","test":"y"})
	ax.set_xticklabels(rotation=45)
	sns.despine(left=True)
	sns.swarmplot(x="model_name",y="formation_exc_error",data=data, color='k', alpha=0.7)
	plt.xlabel("Model", fontsize=18)
	plt.ylabel("Error in Formation Energy (eV)", fontsize=18)
	plt.tick_params('both',labelsize='15')

	handles, _ = ax.get_legend_handles_labels()
	ax.legend(handles, ["Training set", "Test set"],fontsize=15)
	plt.tight_layout()
	plt.savefig("formation_energy_violin_swarm_plot.png")



	plt.figure()

	sns.set(style="whitegrid", palette="pastel", color_codes=True)

	ax = sns.violinplot(x="model_name",y="exc_error",hue="training_test",data=data,split=True,inner="quartile",palette={"training":"b","test":"y"})
	ax.set_xticklabels(rotation=45)
	sns.despine(left=True)
	sns.swarmplot(x="model_name",y="exc_error",data=data, color='k', alpha=0.7)
	plt.xlabel("Model", fontsize=18)
	plt.ylabel("Error in Absolute Energy (eV)", fontsize=18)
	plt.tick_params('both',labelsize='15')

	handles, _ = ax.get_legend_handles_labels()
	ax.legend(handles, ["Training set", "Test set"],fontsize=15)
	plt.tight_layout()
	plt.savefig("energy_violin_swarm_plot.png")











	plt.figure(figsize=(10,10))
	ax = sns.swarmplot(x="model_name",y="exc_error",data=data, hue='molecule_name', split=True, palette=("Dark2"),hue_order = ["C2H2","C2H4","C2H6","CH3OH","CH4","CO","CO2","H2","H2O","HCN","HNC","N2","N2O","NH3","O3","CH3CN","CH3CHO","CH3NO2","glycine","H2CCO","H2CO","H2O2","HCOOH","N2H4","NCCN"]) # 3. Use Pokemon palette
 	ax.set_xticklabels(rotation=45)
	# 5. Place legend to the right
	plt.legend(bbox_to_anchor=(1, 1), loc=2,fontsize=15)
	plt.xlabel("Model", fontsize=18)
	plt.ylabel("Error in Formation Energy (eV)", fontsize=18)
	plt.tick_params('both',labelsize='15')

	plt.savefig("energy_swarm_plot.png")

	plt.figure(figsize=(10,10))
	ax = sns.swarmplot(x="model_name",y="formation_exc_error",data=data, hue='molecule_name', split=True, palette=("Dark2"),hue_order = ["C2H2","C2H4","C2H6","CH3OH","CH4","CO","CO2","H2","H2O","HCN","HNC","N2","N2O","NH3","O3","CH3CN","CH3CHO","CH3NO2","glycine","H2CCO","H2CO","H2O2","HCOOH","N2H4","NCCN"]) # 3. Use Pokemon palette
 	ax.set_xticklabels(rotation=45)
	# 5. Place legend to the right
	plt.legend(bbox_to_anchor=(1, 1), loc=2,fontsize=15)
	plt.xlabel("Model", fontsize=18)
	plt.ylabel("Error in Formation Energy (eV)", fontsize=18)
	plt.tick_params('both',labelsize='15')
	plt.savefig("formation_energy_swarm_plot.png")











	plt.figure()
	ax = sns.swarmplot(x="model_name",y="exc_error",data=data, hue='training_test', split=True, palette=("Dark2")) # 3. Use Pokemon palette
	ax.set_xticklabels(rotation=45)
	plt.xlabel("Model", fontsize=18)
	plt.ylabel("Error in Absolute Energy (eV)", fontsize=18)
	plt.tick_params('both',labelsize='15')

	handles, _ = ax.get_legend_handles_labels()
	ax.legend(handles, ["Training set", "Test set"],fontsize=15)
	# 5. Place legend to the right
	#plt.legend(bbox_to_anchor=(1, 1), loc=2)
	plt.savefig("energy_swarm_plot2.png")

	plt.figure()
	ax = sns.swarmplot(x="model_name",y="formation_exc_error",data=data, hue='training_test', split=True, palette=("Dark2")) # 3. Use Pokemon palette
	ax.set_xticklabels(rotation=45)
	plt.xlabel("Model", fontsize=18)
	plt.ylabel("Error in Formation Energy (eV)", fontsize=18)
	plt.tick_params('both',labelsize='15')

	handles, _ = ax.get_legend_handles_labels()
	ax.legend(handles, ["Training set", "Test set"],fontsize=15)
	# 5. Place legend to the right
	#plt.legend(bbox_to_anchor=(1, 1), loc=2)
	plt.savefig("formation_energy_swarm_plot2.png")













	plt.figure()

	sns.set(style="whitegrid", palette="pastel", color_codes=True)

	ax1 = sns.violinplot(x="model_name",y="formation_exc_error",hue="training_test",data=data,split=True,inner=None,palette={"training":"b","test":"y"})
	sns.despine(left=True)
	ax2 = sns.swarmplot(x="model_name",y="formation_exc_error",data=data, hue='training_test', split=True, color='k', alpha=0.7, palette=("Dark2"))
	ax1.set_xticklabels(rotation=45)
	ax2.set_xticklabels(rotation=45)
	plt.xlabel("Model", fontsize=18)
	plt.ylabel("Error in Absolute Energy (eV)", fontsize=18)
	plt.tick_params('both',labelsize='15')

	handles1, _ = ax1.get_legend_handles_labels()
	ax1.legend(handles1, ["Training set", "Test set"],fontsize=15)

	plt.tight_layout()
	plt.savefig("formation_energy_violin_swarm_plot2.png")


	plt.figure()

	sns.set(style="whitegrid", palette="pastel", color_codes=True)

	ax1 = sns.violinplot(x="model_name",y="exc_error",hue="training_test",data=data,split=True,inner=None,palette={"training":"b","test":"y"})
	sns.despine(left=True)
	ax2 = sns.swarmplot(x="model_name",y="exc_error",data=data, hue='training_test', split=True, color='k', alpha=0.7, palette=("Dark2"))
	ax1.set_xticklabels(rotation=45)
	ax2.set_xticklabels(rotation=45)
	plt.xlabel("Model", fontsize=18)
	plt.ylabel("Error in Absolute Energy (eV)", fontsize=18)
	plt.tick_params('both',labelsize='15')

	handles1, _ = ax1.get_legend_handles_labels()
	ax1.legend(handles1, ["Training set", "Test set"],fontsize=15)

	plt.tight_layout()
	plt.savefig("energy_violin_swarm_plot2.png")

















	plt.figure()
	sns.set(style="whitegrid", palette="pastel", color_codes=True)

	g = sns.factorplot(x="molecule_name", y="formation_exc_error", hue="model_name", data=data,
                   capsize=.2, palette="YlGnBu_d", size=12, legend=False, order = ["C2H2","C2H4","C2H6","CH3OH","CH4","CO","CO2","H2","H2O","HCN","HNC","N2","N2O","NH3","O3","CH3CN","CH3CHO","CH3NO2","glycine","H2CCO","H2CO","H2O2","HCOOH","N2H4","NCCN"])
        g.set_xticklabels(rotation=45,fontsize=15)
        g.set_yticklabels(fontsize=15)
	sns.despine(left=True)
	#plt.legend(bbox_to_anchor=(1, 1), loc=2)
	plt.ylabel("Error in Formation Energy (eV)", fontsize=18)
	plt.xlabel("", fontsize=18)
	#g.legend( ["Model 1", "Model 2", "Model 3", "Model 4", "Model 5"],loc='upper right',fontsize=15)
	plt.tight_layout()
	plt.savefig("formation_factor_plot.png")


	plt.figure()
	sns.set(style="whitegrid", palette="pastel", color_codes=True)

	g = sns.factorplot(x="molecule_name", y="exc_error", hue="model_name", data=data,
                   capsize=.2, palette="YlGnBu_d", size=12, legend=False, order = ["C2H2","C2H4","C2H6","CH3OH","CH4","CO","CO2","H2","H2O","HCN","HNC","N2","N2O","NH3","O3","CH3CN","CH3CHO","CH3NO2","glycine","H2CCO","H2CO","H2O2","HCOOH","N2H4","NCCN"])
        g.set_xticklabels(rotation=45,fontsize=15)
        g.set_yticklabels(fontsize=15)
	sns.despine(left=True)
	#plt.legend(bbox_to_anchor=(1, 1), loc=2)
	plt.ylabel("Error in Absolute Energy (eV)", fontsize=18)
	plt.xlabel("", fontsize=18)
	plt.tight_layout()
	plt.savefig("energy_factor_plot.png")
	
	
	
	
	
	
	
	
	
	plt.figure()
	sns.set(style="whitegrid", palette="pastel", color_codes=True)

	ax = sns.boxplot(x="model_name",y="formation_exc_error",hue="training_test",data=data,palette={"training":"b","test":"y"})
	ax.set_xticklabels(rotation=45)
	sns.despine(left=True)
	
	plt.xlabel("Model", fontsize=18)
	plt.ylabel("Error in Formation Energy (eV)", fontsize=18)
	plt.tick_params('both',labelsize='15')

	handles, _ = ax.get_legend_handles_labels()
	ax.legend(handles, ["Training set", "Test set"],fontsize=15)
	plt.tight_layout()
	plt.savefig("formation_energy_box_plot.png")



	plt.figure()
	
	sns.set(style="whitegrid", palette="pastel", color_codes=True)

	ax = sns.boxplot(x="model_name",y="exc_error",hue="training_test",data=data,palette={"training":"b","test":"y"})
	ax.set_xticklabels(rotation=45)
	sns.despine(left=True)
	plt.xlabel("Model", fontsize=18)
	plt.ylabel("Error in Absolute Energy (eV)", fontsize=18)
	plt.tick_params('both',labelsize='15')

	handles, _ = ax.get_legend_handles_labels()
	ax.legend(handles, ["Training set", "Test set"],fontsize=15)
	plt.tight_layout()
	plt.savefig("energy_box_plot.png")
	

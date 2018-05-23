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


	sns.set(style="whitegrid", palette="pastel", color_codes=True)

	sns.violinplot(x="model_name",y="formation_exc_error",hue="training_test",data=data,split=True,inner="quartile",palette={"training":"b","test":"y"})
	sns.despine(left=True)

	plt.savefig("formation_energy_grouped_violin_plot.png")

	plt.figure()

	sns.set(style="whitegrid", palette="pastel", color_codes=True)

	sns.violinplot(x="model_name",y="exc_error",hue="training_test",data=data,split=True,inner="quartile",palette={"training":"b","test":"y"})
	sns.despine(left=True)

	plt.savefig("energy_grouped_violin_plot.png")


	plt.figure()

	sns.set(style="whitegrid", palette="pastel", color_codes=True)

	sns.violinplot(x="model_name",y="formation_exc_error",hue="training_test",data=data,split=True,inner="quartile",palette={"training":"b","test":"y"})
	sns.despine(left=True)
	sns.swarmplot(x="model_name",y="formation_exc_error",data=data, color='k', alpha=0.7) 

	plt.savefig("formation_energy_violin_swarm_plot.png")



	plt.figure()

	sns.set(style="whitegrid", palette="pastel", color_codes=True)

	sns.violinplot(x="model_name",y="exc_error",hue="training_test",data=data,split=True,inner="quartile",palette={"training":"b","test":"y"})
	sns.despine(left=True)
	sns.swarmplot(x="model_name",y="exc_error",data=data, color='k', alpha=0.7) 

	plt.savefig("energy_violin_swarm_plot.png")


	plt.figure(figsize=(10,15))
	sns.swarmplot(x="model_name",y="exc_error",data=data, hue='molecule_name', split=True, palette=("Dark2")) # 3. Use Pokemon palette
 
	# 5. Place legend to the right
	plt.legend(bbox_to_anchor=(1, 1), loc=2)
	plt.savefig("energy_swarm_plot.png")

	plt.figure(figsize=(10,15))
	sns.swarmplot(x="model_name",y="formation_exc_error",data=data, hue='molecule_name', split=True, palette=("Dark2")) # 3. Use Pokemon palette
 
	# 5. Place legend to the right
	plt.legend(bbox_to_anchor=(1, 1), loc=2)
	plt.savefig("formation_energy_swarm_plot.png")


	plt.figure()
	sns.swarmplot(x="model_name",y="exc_error",data=data, hue='training_test', split=True, palette=("Dark2")) # 3. Use Pokemon palette
 
	# 5. Place legend to the right
	plt.legend(bbox_to_anchor=(1, 1), loc=2)
	plt.savefig("energy_swarm_plot2.png")

	plt.figure()
	sns.swarmplot(x="model_name",y="formation_exc_error",data=data, hue='training_test', split=True, palette=("Dark2")) # 3. Use Pokemon palette
 
	# 5. Place legend to the right
	plt.legend(bbox_to_anchor=(1, 1), loc=2)
	plt.savefig("formation_energy_swarm_plot2.png")


	plt.figure()

	sns.set(style="whitegrid", palette="pastel", color_codes=True)

	sns.violinplot(x="model_name",y="formation_exc_error",hue="training_test",data=data,split=True,inner="quartile",palette={"training":"b","test":"y"})
	sns.despine(left=True)
	sns.swarmplot(x="model_name",y="formation_exc_error",data=data, hue='training_test', split=True, color='k', alpha=0.7) 

	plt.savefig("formation_energy_violin_swarm_plot2.png")



	plt.figure()

	sns.set(style="whitegrid", palette="pastel", color_codes=True)

	sns.violinplot(x="model_name",y="exc_error",hue="training_test",data=data,split=True,inner="quartile",palette={"training":"b","test":"y"})
	sns.despine(left=True)
	sns.swarmplot(x="model_name",y="exc_error",data=data, hue='training_test', split=True, color='k', alpha=0.7) 

	plt.savefig("energy_violin_swarm_plot2.png")



	plt.figure(figsize=(20,10))

	sns.factorplot(x="molecule_name", y="formation_exc_error", hue="model_name", data=data,
                   capsize=.2, palette="YlGnBu_d", size=6)
	sns.despine(left=True)
	plt.legend(bbox_to_anchor=(1, 1), loc=2)

	plt.savefig("formation_factor_plot.png")



	plt.figure(figsize=(20,10))

	sns.factorplot(x="molecule_name", y="exc_error", hue="model_name", data=data,
                   capsize=.2, palette="YlGnBu_d", size=6)
	sns.despine(left=True)
	plt.legend(bbox_to_anchor=(1, 1), loc=2)

	plt.savefig("energy_factor_plot.png")
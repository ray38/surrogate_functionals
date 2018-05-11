from subsampling import subsampling_system_with_PCA, random_subsampling, subsampling_system
import random
import sklearn
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot(dataframe,filename):
	with sns.axes_style('white'):
		temp_plot = sns.jointplot('x', 'y', data=dataframe,size=10,s=1,stat_func=None,marker='o')
		#figure = temp_plot.get_figure()
		plt.savefig(filename)


n=100000

x_mu, x_sigma = 0, 0.1 
y_mu, y_sigma = 100, 20



x = np.random.normal(x_mu, x_sigma, n)
y = np.random.normal(y_mu, y_sigma, n)

df = pd.DataFrame({'x':x, 'y':y, 'group':np.repeat('original',n)})

plot(df,"original.png")
plt.figure()


temp = np.column_stack((x,y))
temp_subsampled = np.asarray(subsampling_system(temp,list_desc = [],cutoff_sig=0.2,rate=0.1))
x_subsampled = temp_subsampled[:,0]
y_subsampled = temp_subsampled[:,1]

df2 = pd.DataFrame({'x':x_subsampled, 'y':y_subsampled, 'group':np.repeat('subsampled',len(x_subsampled))})
df.append(df2)
plot(df2,"subsampled.png")


plt.figure()
with sns.axes_style('white'):
	temp_plot = sns.JointGrid(x='x', y='y', data=df,size=10)
	temp_plot.plot_joint(sns.kdeplot,shade=True,cmap="Reds")
	temp_plot.plot_marginals(sns.kdeplot,shade=True,color='r')
	
	temp_plot.x=x_subsampled
	temp_plot.y=y_subsampled
	temp_plot.plot_joint(plt.scatter,marker='o',c='b',s=20)
	temp_plot.plot_marginals(sns.kdeplot,shade=True,color='b')
	#temp_plot = sns.jointplot('x', 'y', data=df2,size=10,s=1,stat_func=None)
	#figure = temp_plot.get_figure()
	plt.savefig("test.png")


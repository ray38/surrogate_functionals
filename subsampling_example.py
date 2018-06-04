import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from subsampling import subsampling_system_with_PCA, random_subsampling, subsampling_system
import random
import sklearn
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot(dataframe,filename):
	with sns.axes_style('white'):
		#g = sns.jointplot('x', 'y', data=dataframe,size=10,s=2,stat_func=None,marker='o', space = 0)
		g = sns.JointGrid(x="x", y="y",data=dataframe,space=0)
		g = g.plot_joint(plt.scatter, color = "b", s=2)
		count = lambda a, b: len(a)
		g.annotate(count, template="{stat}: {val:.0f}",stat="Count",loc="upper left", fontsize = 12)
		
		_ = g.ax_marg_x.hist(dataframe["x"], color = "b", alpha = 0.6, bins = np.arange(-0.5, 0.5, 0.02))
		_ = g.ax_marg_y.hist(dataframe["y"], color = "b", alpha = 0.6, orientation = "horizontal",bins = np.arange(0, 200, 4))
		#figure = temp_plot.get_figure()
		plt.savefig(filename)


n=100000

x_mu, x_sigma = 0, 0.1 
y_mu, y_sigma = 100, 20



x = np.random.normal(x_mu, x_sigma, n)
y = np.random.normal(y_mu, y_sigma, n)

df = pd.DataFrame({'x':x, 'y':y, 'group':np.repeat('original',n)})

#plot(df,"original.png")
plt.figure()
plt.scatter(x, y,size=10)
plt.savefig("original.png", transparent=True)


temp = np.column_stack((x,y))
temp_subsampled = np.asarray(subsampling_system(temp,list_desc = [],cutoff_sig=0.2,rate=0.1))
x_subsampled = temp_subsampled[:,0]
y_subsampled = temp_subsampled[:,1]

plt.figure()
plt.scatter(x_subsampled, y_subsampled,size=10)
plt.savefig("subsampled.png", transparent=True)

#df2 = pd.DataFrame({'x':x_subsampled, 'y':y_subsampled, 'group':np.repeat('subsampled',len(x_subsampled))})
#df.append(df2)
#plot(df2,"subsampled.png")


#plt.figure()
#with sns.axes_style('white'):
#	temp_plot = sns.JointGrid(x='x', y='y', data=df,size=50, space = 0)
#	#temp_plot.plot_joint(sns.kdeplot,shade=True,cmap="Reds")
#	temp_plot.plot_joint(plt.scatter,marker='o',c='r',s=2)
#	temp_plot.plot_marginals(sns.distplot,color='r')
	
	
#	temp_plot.x=x_subsampled
#	temp_plot.y=y_subsampled
#	temp_plot.plot_joint(plt.scatter,marker='o',c='b',s=20)
#	temp_plot.plot_marginals(sns.distplot,color='b')
	
#	#temp_plot = sns.jointplot('x', 'y', data=df2,size=10,s=1,stat_func=None)
#	#figure = temp_plot.get_figure()
#	plt.savefig("test.png", transparent=True)


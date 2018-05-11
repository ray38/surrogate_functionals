from subsampling import subsampling_system_with_PCA, random_subsampling, subsampling_system
import random
import sklearn
import seaborn as sns

mean = [0, 0]
cov = [[1, 0], [0, 100]]
import matplotlib.pyplot as plt
x, y = np.random.multivariate_normal(mean, cov, 5000).T
plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()
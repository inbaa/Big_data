from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets.samples_generator import make_blobs
# creating datasets X containing n_samples
# Y containing two classes

# data = pd.read_csv('E:/Online courses/Data Science 5-day course/day 3/insurance.csv')
# data.loc[data["smoker"]=="no", "status"]=0
# data.loc[data["smoker"]=="yes", "status"]=1
# X = data.iloc[:,[0,6] ].values
# Y = data.iloc[:, -1].values #contains binary values

X, Y = make_blobs(n_samples=400, centers=2,random_state=0, cluster_std=0.40)
# print(X,Y)
# plotting scatters
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='spring');
plt.show()
# creating line space between -1 to 3.5
xfit = np.linspace(-1, 3.5)
# plotting scatter
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='spring')
# plot a line between the different sets of data
for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none', color='#AAAAAA', alpha=0.4)
plt.xlim(-1, 3.5);
plt.show()
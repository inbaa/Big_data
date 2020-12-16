import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('E:/mca/Big Data analytics/Practical/digest_2015_table_209_25.csv')
X = data.iloc[:20,2:6]  #independent columns (features)
y = data.iloc[:20,0]    #target column i.e States
#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="YlGnBu")
plt.show()
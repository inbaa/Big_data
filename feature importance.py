import pandas as pd
import numpy as np
data = pd.read_csv('E:/mca/Big Data analytics/Practical/digest_2015_table_209_25.csv')
X = data.iloc[:20,2:6]  #independent columns (features)
y = data.iloc[:20,0]    #target column i.e States
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
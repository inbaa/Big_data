import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
data = pd.read_csv('E:/mca/Big Data analytics/Practical/digest_2015_table_209_25.csv')
data.replace([np.inf, -np.inf], np.nan, inplace=True)
X = data.iloc[:20,2:6]  #independent columns (features)
y = data.iloc[:20,0]    #target column i.e States
# #apply SelectKBest class to extract top  4 best features k>=0, <=features
bestfeatures = SelectKBest(score_func=chi2, k=4)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Column','Value']  #naming the dataframe columns
print(featureScores.nlargest(10,'Value'))  #print 10 best features
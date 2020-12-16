import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score

data = pd.read_csv('E:/Online courses/Data Science 5-day course/day 4/Network_Ads.csv')
data=data.dropna() #drop nan value

x = data.iloc[:, 2].values.reshape(-1,1)
y = data.iloc[:, 4].values #contains binary values
# print(data.head())
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0) #test data is 25% train data is 75%
model=LogisticRegression().fit(x_train,y_train)

predictions = model.predict(x_test)
print(classification_report(y_test, predictions))

print("Confusion matrix : ",confusion_matrix(y_test, predictions))

import seaborn as sns
sns.heatmap(pd.DataFrame(confusion_matrix(y_test,predictions)))
plt.show()

print('Accuracy score ',accuracy_score(y_test, predictions))


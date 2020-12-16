import pandas as pd

dictionary1 = { 'salary' :
[45000,40000,35000,30000,42000,37000,43000,38000,41000,44000,90000,
80000,70000,60000,95000,85000,75000,65000,84000,92000],

'bike_or_car' : [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]
}

datatable1 = pd.DataFrame (dictionary1)

X = datatable1.iloc[:, :-1].values
y = datatable1.iloc[:, 1].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)




# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X, y)

# Predicting the Test set results
y_pred = classifier.predict(X)
print(y_pred)

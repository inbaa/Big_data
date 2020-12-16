import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
data = pd.read_csv('E:/mca/Big Data analytics/Practical/digest_2015_table_209_25.csv')
data=data.dropna() #drop nan value

x = data.iloc[:, 2].values.reshape(-1,1) #reshape is used to convert to 2D array with 1 elemnet
y = data.iloc[:, 4].values

model=LinearRegression().fit(x,y)

r_sq=model.score(x,y) #coefficient of determination R square using score
print('Coefficient of determination', r_sq)

print('Intercept', model.intercept_)
print('Slope', model.coef_)

y_predict=model.predict(x)
print('Predicted values',y_predict, sep='\n')

# Visualising the Training set results
plt.scatter(x, y, color = 'red')
plt.plot(x, model.predict(x), color = 'blue')
plt.title('Linear Regression')
plt.show()


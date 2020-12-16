from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.read_csv('E:/Online courses/Data Science 5-day course/day 3/insurance.csv')
data.loc[data["smoker"]=="no", "status"]=0
data.loc[data["smoker"]=="yes", "status"]=1
print(data.head())
X = data.iloc[:,[0,6] ].values
y = data.iloc[:, -1].values #contains binary values

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='spring')
plt.xlabel('Age')
plt.ylabel('Expenses')
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

from sklearn.svm import SVC
svm = SVC(kernel="linear")
svm.fit(X_train, y_train)
y_pred =svm.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
plt.scatter(X_train[:,0], X_train[:,1]) #Visualizing
plt.scatter(svm.support_vectors_[:,0], svm.support_vectors_[:,1], color='red') #data separated with red dots
plt.title('Linearly separable data with support vectors')
plt.xlabel('Age')
plt.ylabel('Expenses')
plt.show()

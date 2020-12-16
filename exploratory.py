import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('E:/mca/Big Data analytics/Practical/digest_2015_table_209_25.csv')
# print(data.head())
# print(data.dtypes)
data=data.drop(['Percent.1','Number.1','Percent.2','Number.2','Percent.3','Number.3'], axis=1)
# print(data.head())
# print(data.shape)
# duplicate=data[data.duplicated()] #checking for duplicate row
# print(duplicate.shape)
# print(data.isnull().sum()) # total no. of null value in dataframe
# dropping null rows
d=data.dropna()
# print(d.isnull().sum())
# d.to_csv('d:/test.csv') #exporting dataframe
# Outlier
# sns.boxplot(x=data['Number']) # ploting
# plt.show()
#quarantile
# q1=d.quantile(0.25)
# q3=d.quantile(0.75)
# iqr=q3-q1
# print(iqr)

# d.Number.value_counts().nlargest(100).plot(kind='bar', figsize=(10,5)) #bar garph
# plt.ylabel('State')
# plt.xlabel('Percent')
# plt.show()

# plt.figure(figsize=(10,10))
# sns.heatmap(d.corr(),annot=True,cmap="viridis")
# plt.show()

plt.subplots(figsize=(6,5))
plt.scatter(d['Number'],d['Percent'])
plt.ylabel('Percent')
plt.xlabel('Number')
plt.show()
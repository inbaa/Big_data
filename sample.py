import pandas as pd

data = pd.read_csv('E:/mca/Big Data analytics/Practical/digest_2015_table_209_25.csv')
# print(data.head())
# print(data.sample(n=7))
print(data.sample(frac=0.25, random_state=60))
import pandas as pd
dictionary1 = { 'salary' :
[45000,40000,35000,30000,42000,37000,43000,38000,41000,44000,90000,80000,70000,60000,95000,85000,75000,65000,84000,92000,190000,180000,170000,160000,195000,185000,175000,165000,184000,192000],
'car_type' :
[0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2]
 }

datatable1 = pd.DataFrame (dictionary1)


# Fitting K-Means to the dataset
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(datatable1)
print(y_kmeans)

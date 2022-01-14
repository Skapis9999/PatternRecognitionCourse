# from os import sep
# from numpy.core.fromnumeric import mean
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.cluster import KMeans

# Read the Data
cdata = pd.read_csv("./7. kMeans/quiz_data.csv")
cdata = cdata.loc[:, ["X1", "X2"]]

plt.scatter(cdata.X1, cdata.X2)
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

# question 1
# KMeans with 3 clusters and starting points [-4, 10], [0,0], [4,10]
initV = np.array([[-4, 10], [0,0], [4,10]], np.float64)
kmeans = KMeans(n_clusters=3, init=initV).fit(cdata)
print(kmeans.inertia_) #cohesion 637.8684021921762

# question 2
# Separation of KMeans
separation = 0
distance = lambda x1, x2: math.sqrt(((x1.X1 - x2.X1) ** 2) + ((x1.X2 - x2.X2) ** 2))
m = cdata.mean()
for i in list(set(kmeans.labels_)):
    mi = cdata.loc[kmeans.labels_ == i, :].mean()
    Ci = len(cdata.loc[kmeans.labels_ == i, :].index)
    separation += Ci * (distance(m, mi) ** 2)
print(separation) #separation 9496.20591189718

# spatial of 1st Kmeans
# question 4
cdata["cluster"] = kmeans.labels_
cdata = cdata.sort_values("cluster").drop("cluster", axis=1)
from scipy.spatial import distance_matrix
dist = distance_matrix(cdata, cdata)
plot1 = plt.figure(1)
plt.imshow(dist, cmap='hot')
plt.colorbar()


# question 3 
# silhouette 1
from sklearn.metrics import silhouette_samples, silhouette_score
print(silhouette_score(cdata, kmeans.labels_)) #silhouette 0.7816266818416537

#question 4
# KMeans with 3 clusters and starting points [-2, 0], [2,0], [0,10]
initV2 = np.array([[-2, 0], [2,0], [0,10]], np.float64)
kmeans = KMeans(n_clusters=3, init=initV2).fit(cdata)
print(kmeans.inertia_) #cohesion 3667.769665632974

# Separation of KMeans
for i in list(set(kmeans.labels_)):
    mi = cdata.loc[kmeans.labels_ == i, :].mean()
    Ci = len(cdata.loc[kmeans.labels_ == i, :].index)
    separation += Ci * (distance(m, mi) ** 2)
print(separation) #separation 15962.51056035356

# silhouette 2
print(silhouette_score(cdata, kmeans.labels_)) #silhouette 0.45532884123093065

# spatial of 2nd Kmeans
cdata["cluster"] = kmeans.labels_
cdata = cdata.sort_values("cluster").drop("cluster", axis=1)
from scipy.spatial import distance_matrix
dist = distance_matrix(cdata, cdata)
plot2 = plt.figure(2)
plt.imshow(dist, cmap='hot')
plt.colorbar()
plt.show()
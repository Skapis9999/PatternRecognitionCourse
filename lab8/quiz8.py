import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# Read the Data
data = pd.read_csv("./8. Hierarchical Clustering - DBSCAN/dcdata.txt")
target = data.loc[:, "Y"]
cdata = data.loc[:, ["X1", "X2"]]
labels = ["X1", "X2"]
labelList = range(200)

#plot the data
plt.scatter(data[(target == 1)].X1, data[(target == 1)].X2, c="red", marker="o")
plt.scatter(data[(target == 0)].X1, data[(target == 0)].X2, c="blue", marker="o")
plt.show()

# Hierarchical clustering with single linage
clustering = AgglomerativeClustering(n_clusters=2, linkage="single", distance_threshold=None, compute_distances=True).fit(cdata)
clusters = clustering.labels_
plt.scatter(data.X1, data.X2, c=clusters, cmap="spring")
plt.title("Hierarchical clustering with single linage")
plt.show()
print(accuracy_score(target, clustering.labels_)) #Question 2 accuracy = 0.495

# linkage_matrix = np.column_stack([clustering.children_, clustering.distances_, np.ones(len(cdata.index)-1)]).astype(float)
# dendrogram(linkage_matrix, labels=labelList)
# plt.show()

# print(silhouette_score(data, clustering.labels_))


# Hierarchical clustering with complete linage
clustering = AgglomerativeClustering(n_clusters=2, linkage="complete", distance_threshold=None, compute_distances=True).fit(cdata)
clusters = clustering.labels_
plt.scatter(data.X1, data.X2, c=clusters, cmap="spring")
plt.title("Hierarchical clustering with complete linage")
plt.show()
print(accuracy_score(target, clustering.labels_)) #Question 3 accuracy = 0.935

# linkage_matrix = np.column_stack([clustering.children_, clustering.distances_, np.ones(len(cdata.index)-1)]).astype(float)
# dendrogram(linkage_matrix, labels=labelList)
# plt.show()
# print(silhouette_score(data, clustering.labels_))

# DBSCAN eps=1.25, minPts=5
clustering = DBSCAN(eps=1.25, min_samples=5).fit(data)
clusters = clustering.labels_
plt.scatter(cdata.X1, cdata.X2, c=clusters, cmap="spring")
plt.title("DBSCAN(eps=1.25, minPts=5)")
plt.show()
# print(silhouette_score(data, clustering.labels_))

# DBSCAN eps=0.75, minPts=5
clustering = DBSCAN(eps=0.75, min_samples=5).fit(data)
clusters = clustering.labels_
plt.scatter(cdata.X1, cdata.X2, c=clusters, cmap="spring")
plt.title("DBSCAN(eps=0.75, minPts=5)")
plt.show()
# print(silhouette_score(data, clustering.labels_))

# DBSCAN eps=1.5, minPts=5
clustering = DBSCAN(eps=1.5, min_samples=5).fit(data)
clusters = clustering.labels_
plt.scatter(cdata.X1, cdata.X2, c=clusters, cmap="spring")
plt.title("DBSCAN(eps=1.5, minPts=5)")
plt.show()
# print(silhouette_score(data, clustering.labels_))

# DBSCAN eps=1, minPts=5
clustering = DBSCAN(eps=1, min_samples=5).fit(data)
clusters = clustering.labels_
plt.scatter(cdata.X1, cdata.X2, c=clusters, cmap="spring")
plt.title("DBSCAN(eps=1, minPts=5)")
plt.show()
# print(silhouette_score(data, clustering.labels_))

# KMeans with 2 clusters and random starting cluster centers
initV = np.array([[-4, 10], [3,3]], np.float64)
kmeans = KMeans(n_clusters=2, init=cdata.loc[0:1, :]).fit(cdata)
# plt.scatter(cdata.X1, cdata.X2, c=clusters, cmap="spring")
plt.scatter(cdata.X1, cdata.X2, c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="+", s=169, c=range(2))
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("KMeans(n_clusters=2)")
plt.show()
# print(silhouette_score(data, clustering.labels_))

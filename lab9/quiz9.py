import pandas as pd
import matplotlib.pyplot as plt


# Read the Data
gdata = pd.read_csv("./9. Expectation Maximization - GMMs/kmdata.txt")
x = gdata.loc[:, ["X1", "X2"]]
y = gdata.loc[:, "Y"]

#plot the data
plt.scatter(gdata[(y == 1)].X1, gdata[(y == 1)].X2, c="blue", marker="o")
plt.scatter(gdata[(y == 0)].X1, gdata[(y == 0)].X2, c="blue", marker="o")
plt.show()

plt.scatter(gdata[(y == 1)].X1, gdata[(y == 1)].X2, c="red", marker="o")
plt.scatter(gdata[(y == 2)].X1, gdata[(y == 2)].X2, c="blue", marker="o")
plt.scatter(gdata[(y == 3)].X1, gdata[(y == 3)].X2, c="yellow", marker="o")
plt.show()

import numpy as np
from sklearn.cluster import KMeans

# KMeans with 3 clusters and random starting cluster centers
initV = np.array([[-4, 10], [3,3], [1,1]], np.float64)
kmeans = KMeans(n_clusters=3, init=gdata.loc[0:2, :]).fit(gdata)
plt.scatter(gdata.X1, gdata.X2, c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], marker="o", c=range(3))
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], marker="+", c=range(3))
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("KMeans(n_clusters=3)")
plt.show()

from sklearn.metrics import accuracy_score
print(accuracy_score(kmeans.labels_, y))

epsilon = 1e-04

#Gaussian Mixture with 3 clusters
from sklearn.mixture import GaussianMixture
import seaborn as sns
gm = GaussianMixture(n_components=3, tol=epsilon).fit(x)
centers = gm.means_
clusters = gm.predict(x)

gdata["cluster"] = clusters
gdata = gdata.sort_values("cluster").drop("cluster", axis=1)
from scipy.spatial import distance_matrix
dist = distance_matrix(gdata, gdata)
plt.imshow(dist, cmap='hot')
plt.colorbar()
plt.show()

# plt.contour(X, Y, Z)
x1 = np.linspace(np.min(x.loc[:, "X1"]), np.max(x.loc[:, "X1"]))
y1 = np.linspace(np.min(x.loc[:, "X2"]), np.max(x.loc[:, "X2"]))
X, Y = np.meshgrid(x1, y1)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -gm.score_samples(XX)
Z = Z.reshape(X.shape)
plt.scatter(gdata.X1, gdata.X2, c=clusters)
plt.scatter(centers[:, 0], centers[:, 1], marker="+", s=169, c=range(3))
plt.contour(X, Y, Z)
plt.show()
p = accuracy_score(clusters, y)
if p<0.5 :
    p = 1 - p
print(p)

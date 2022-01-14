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

# faction mix pdf
from scipy.stats import norm
def mix_pdf(x, loc, scale, weights):
    d = np.zeros_like(x)
    for mu, sigma, pi in zip(loc, scale, weights):
        d += pi * norm.pdf(x, loc = mu, scale = sigma)
    return d


epsilon = 1e-04

#Gaussian Mixture with 3 clusters
from sklearn.mixture import GaussianMixture
import seaborn as sns
data = np.array(x.values.tolist()).reshape(-1,1)
gm = GaussianMixture(n_components=3).fit(data)
centers = gm.means_

# plt.contour(X, Y, Z)
plt.scatter(gdata[(y == 1)].X1, gdata[(y == 1)].X2, c="red", marker="+")
plt.scatter(gdata[(y == 2)].X1, gdata[(y == 2)].X2, c="blue", marker="o")
plt.scatter(gdata[(y == 3)].X1, gdata[(y == 3)].X2, c="black", marker="x")
plt.legend(loc = 'upper right')
plt.show()
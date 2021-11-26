import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import numpy as np

X1 = [2, 2, -2, -2, 1, 1, -1, -1]
X2 = [2, -2, -2, 2, 1, -1, -1, 1]
Y = [1, 1, 1, 1, 2, 2, 2, 2]
alldata = pd.DataFrame({"X1":X1, "X2":X2, "Y":Y})
X = alldata.loc[:, ["X1", "X2"]]
y = alldata.loc[:, "Y"]

clf = MLPRegressor(hidden_layer_sizes=(2), max_iter=10000)
clf = clf.fit(X, y)

pred = clf.predict(X)
trainingError = [(t - p) for (t, p) in zip(y, pred)]
MAE = np.mean(np.abs(trainingError))
print(MAE)


clf2 = MLPRegressor(hidden_layer_sizes=(20), max_iter=10000)
clf2 = clf2.fit(X, y)

pred2 = clf2.predict(X)
trainingError2 = [(t - p) for (t, p) in zip(y, pred2)]
MAE2 = np.mean(np.abs(trainingError2))
print(MAE2)

clf3 = MLPRegressor(hidden_layer_sizes=(20, 20), max_iter=10000)
clf3 = clf3.fit(X, y)

pred3 = clf3.predict(X)
trainingError3 = [(t - p) for (t, p) in zip(y, pred3)]
MAE3 = np.mean(np.abs(trainingError3))
print(MAE3)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# X1 = [-2.0, -2.0, -1.8, -1.4, -1.2, 1.2, 1.3, 1.3, 2.0, 2.0, -0.9, -0.5, -0.2, 0, 0, 0.3, 0.4, 0.5, 0.8, 1.0]
# X2 = [-2.0, 1.0, -1.0, 2.0, 1.2, 1.0, -1.0, 2.0, 0.0, -2.0, 0.0, -1.0, 1.5, 0.0, -0.5, 1.0, 0.0, -1.5, 1.5, 0.0]

# Y = [1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2]
# alldata = pd.DataFrame({"X1":X1, "X2":X2, "Y":Y})

# X = alldata.loc[:, ["X1", "X2"]]
# y = alldata.Y

# plt.scatter(X[(y == 2)].X1, X[(y == 2)].X2, c="red", marker="+")
# plt.scatter(X[(y == 1)].X1, X[(y == 1)].X2, c="blue", marker="o")
# plt.show()

from sklearn.neighbors import KNeighborsClassifier

# #print(X)
# clf = KNeighborsClassifier(n_neighbors=3)
# clf = clf.fit(X, y)
# print(clf.predict([[1.5, -0.5]]))
# print(clf.predict_proba([[1.5, -0.5]]))

# clf = KNeighborsClassifier(n_neighbors=5)
# clf = clf.fit(X, y)
# print(clf.predict([[-1, 1]]))
# print(clf.predict_proba([[-1, 1]]))

####################
X1 = [2, 2, -2, -2, 1, 1, -1,-1]
X2 = [2, -2, -2, 2, 1, -1, -1, 1]

Y = [1, 1, 1, 1, 2, 2, 2, 2]
alldata = pd.DataFrame({"X1":X1, "X2":X2, "Y":Y})

X = alldata.loc[:, ["X1", "X2"]]
y = alldata.Y

X1 = np.arange (min(X.X1.tolist()), max(X.X1.tolist()), 0.01)
X2 = np.arange (min(X.X2.tolist()), max(X.X2.tolist()), 0.01)
xx, yy = np.meshgrid(X1, X2)

from sklearn import svm
from sklearn.metrics import accuracy_score

clf = svm.SVC(kernel="rbf", gamma=1)
clf = clf.fit(X, y)
pred = clf.predict(X)

#print(accuracy_score(y, pred))


clf2 = svm.SVC(kernel="rbf", gamma=1000000)
clf2 = clf2.fit(X, y)
pred = clf2.predict([[-1, -1.9]])

gammavalues = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]
xx, yy = np.meshgrid(X1, X2)


plt.show()

for gamma in gammavalues:
    clf = svm.SVC(kernel="rbf", gamma=gamma)
    clf = clf.fit(X, Y)
    pred = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    pred = pred.reshape(xx.shape)
    X1 = np.arange (min(X.X1.tolist()), max(X.X1.tolist()), 0.1)
    X2 = np.arange (min(X.X2.tolist()), max(X.X2.tolist()), 0.1)
    plt.contour(xx, yy, pred, colors="green")
    plt.scatter(X[(y == 2)].X1, X[(y == 2)].X2, c="red", marker="+")
    plt.scatter(X[(y == 1)].X1, X[(y == 1)].X2, c="blue", marker="o")
    plt.show()


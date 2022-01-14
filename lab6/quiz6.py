import pandas as pd
from sklearn.preprocessing import StandardScaler

# Read the Data
data =pd.read_csv("./6. Preprocessing - PCA - ISOMAP/quiz_data.csv",sep=",")

# Define Training Set
trainingRange = list(range(0,50)) + list(range(90, 146))
training = data.loc[trainingRange, :]
trainingType = training.loc[:, "Type"]
training = training.drop(["Type"],axis=1)

# Define Testing Set
testingRange = list(range(50,90))
testing = data.loc[testingRange, :]
testingType = testing.loc[:,"Type"]
testing = testing.drop(["Type"],axis=1)

from sklearn.decomposition import PCA
scaler = StandardScaler()
scaler = scaler.fit(training)
transformedTrain = pd.DataFrame(scaler.transform(training), columns=training.columns)
transformedTesting = pd.DataFrame(scaler.transform(testing), columns=testing.columns)


# Perform PCA
pca = PCA()
pca = pca.fit(transformedTrain)
pca_transformed = pca.transform(transformedTrain)
eigenvalues = pca.explained_variance_
eigenvectors = pca.components_
print(eigenvalues)

pca_inverse = pd.DataFrame(pca.inverse_transform(pca_transformed), columns=transformedTrain.columns)

info1 = (eigenvalues[0]) / sum(eigenvalues)
print(info1)   #PC1 info percentage
info2 = (eigenvalues[0]+eigenvalues[1]+eigenvalues[2]+eigenvalues[3]) / sum(eigenvalues)
print (1-info2)   #first 4 PCs info loss

# question 3
# KNN to initial data
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf = clf.fit(training, trainingType)

from sklearn.metrics import accuracy_score, recall_score
pred = clf.predict(testing)
print(accuracy_score(testingType, pred))
print(recall_score(testingType, pred))

# question 4
# KNN to PCA data
import numpy as np
testingError = []
for i in range(len(eigenvalues)):
    a =[]
    pca = PCA(n_components= i + 1)
    pca = pca.fit(transformedTrain)
    pca_transformed = pca.transform(transformedTrain)
    pca_training = pca.transform(transformedTrain)
    pca_testing = pca.transform(transformedTesting)
    pca_inverse1 = pd.DataFrame(pca.inverse_transform(pca_training), columns=training.columns)
    pca_inverse2 = pd.DataFrame(pca.inverse_transform(pca_testing), columns=training.columns)
    clf = clf.fit(pca_inverse1, trainingType)
    for n in range(i+1):
        a.append(n+1)
    if a[-1] == 8:
        break
    # print(testing.iloc[:,a])
    pred = clf.predict(pca_inverse2)
    testingError.append(accuracy_score(testingType, pred))

print(testingError)

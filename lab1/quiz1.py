from sklearn import datasets
import pandas as pd
import statistics

iris = datasets.load_iris()

#featureIndex = iris.feature_names.index(featureName)
data = iris.data
#print(data)
featureIndex = iris.feature_names.index('petal length (cm)')
q1 = statistics.mean(data[:,featureIndex])

featureIndex = iris.feature_names.index('sepal width (cm)')
q2 = max(data[:,featureIndex])

featureIndex = iris.feature_names.index('sepal length (cm)')
q3 = statistics.variance(data[:,featureIndex])


print('Mean petal length (cm) is ', q1, ' while max sepal width (cm) is ',q2,' and the variance of epal length (cm) is ',q3)

names = [iris.feature_names.index('petal length (cm)'), iris.feature_names.index('sepal width (cm)'), iris.feature_names.index('petal width (cm)'), iris.feature_names.index('sepal length (cm)')]
n = sorted(names)
q = [statistics.mean(data[:,i]) for i in n]

print('Mean of each collunn is', q)
#[5.84, 3.06, 3.76, 1.20]
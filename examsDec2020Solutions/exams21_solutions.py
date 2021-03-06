import pandas as pd
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

alldata = pd.read_csv("./exams21/data.csv")
#print(alldata)

data1 = alldata.loc[(alldata.satisfaction < 0.6)&(alldata.department == 'technical')] # Access all columns of the rows that satisfy a condition
#print(data1)

#GINI with frequencies. Wrong but useful

# frequencies = {}

# for item in data1.left:
#     if item in frequencies:
#         frequencies[item] += 1
#     else:
#         frequencies[item] = 1

# #print(frequencies)
# a = len(data1.left)
# GINI= 1 - (273/a)**2 - (514/a)**2

absfreq = pd.crosstab(data1.left, data1.salary)
freq = pd.crosstab(data1.left, data1.salary, normalize='index')
freqSum = pd.crosstab(data1.left, data1.salary, normalize='all').sum(axis=1)
GINI_Yes = 1 - freq.loc["Yes", "high"]**2 - freq.loc["Yes", "medium"]**2 - freq.loc["Yes", "low"]**2
GINI_No = 1 - freq.loc["No", "high"]**2 - freq.loc["No", "medium"]**2 - freq.loc["No", "low"]**2
GINI_Outlook = freqSum.loc["Yes"] * GINI_Yes + freqSum["No"] * GINI_No
print(GINI_Outlook)
#Question 11
#0.5886122835974265
##########################################

encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
data2 = alldata.loc[(alldata.promotion == 'No')] # Access all columns of the rows that satisfy a condition

encoder.fit(data2.loc[:, ['left', 'promotion', 'department']])
transformed = encoder.transform(data2.loc[:, ['left', 'promotion', 'department']])
clf = tree.DecisionTreeClassifier()
clf = clf.fit(transformed, data2.loc[:, 'salary'])

new_data = pd.DataFrame({'left': ['No'], 'promotion': ['No'], 'department': ['sales']})
transformed_new_data = encoder.transform(new_data)
print(clf.predict(transformed_new_data))
print(clf.predict_proba(transformed_new_data), clf.classes_)
#Question 12
#0.46484698
##########################################

kappa = range(10, 41)

X = data2.loc[:, ['projects', 'hours']]
y = data2.loc[:, ['salary']]

counter = 0

from sklearn.neighbors import KNeighborsClassifier
for n in kappa:
    clf = KNeighborsClassifier(n_neighbors=n)
    clf = clf.fit(X, y)
    b = clf.predict([[0.0041, 0.0044]])
    if b == "medium":
        counter = counter + 1

print(counter)
#Question 13
#17 out of 31   
##########################################
data3 = alldata.loc[(alldata.accident == 'Yes')] # Access all columns of the rows that satisfy a condition

from sklearn.naive_bayes import CategoricalNB

X = data3.loc[:, ['left', 'department']]
y = data3.loc[:, ['salary']]

encoder3 = OneHotEncoder(handle_unknown="ignore", sparse=False)
encoder3 = encoder3.fit(X)
X = encoder3.transform(X)

clf3 = CategoricalNB(alpha=0)
clf3.fit(X, y)

new_data = pd.DataFrame({'left': ['No'], 'department': ['management']})
transformed_new_data = encoder3.transform(new_data)
print(clf3.predict(transformed_new_data))
print(clf3.predict_proba(transformed_new_data), clf3.classes_)
#Question 14
#0.04939939
##########################################

data = alldata.loc[(alldata.promotion == 'No')] # Access all columns of the rows that satisfy a condition

X = data.loc[:, ['projects', 'hours']]
y = data.loc[:, ['salary']]

from sklearn import svm
from sklearn.metrics import f1_score

clf = svm.SVC(kernel="rbf", gamma=100)
clf = clf.fit(X, y)
pred = clf.predict(X)
f = f1_score(y, pred, pos_label=1, average='micro')
f2 = f1_score(y, pred, pos_label=1, average='weighted')

print(f)
print(f2)
# Question 15
# 0.4933723196881092 for ???micro???-averaging 
# 0.3259953899309762 for "weighted"-averaging 
##########################################

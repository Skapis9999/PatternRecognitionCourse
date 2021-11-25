import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import CategoricalNB

dataV = pd.read_csv("./3. Naive Bayes/quiz_data.csv")
y = dataV.loc[:,"Class"]
probM1 = dataV.loc[:, "P_M1"]
probM2 = dataV.loc[:, "P_M2"]

from sklearn.metrics import roc_curve, f1_score, auc

fpr1, tpr1, thresholds1 = roc_curve(y, probM1)

for i in range(len(tpr1)):
    print("tpr is ", tpr1[i], "and threshold is ", thresholds1[i])


print("F1 Score: ", f1_score(y, round(probM2), pos_label=1))

import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic Model 1')
plt.plot(fpr1, tpr1, 'b', label = 'AUC = %0.2f' % auc(fpr1, tpr1))
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

fpr2, tpr2, thresholds2 = roc_curve(y, probM2)


plt.title('Receiver Operating Characteristic Model 2')
plt.plot(fpr2, tpr2, 'b', label = 'AUC = %0.2f' % auc(fpr2, tpr2))
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

print("AUC 1: ", auc(fpr1, tpr1))
print("AUC 2: ", auc(fpr2, tpr2))
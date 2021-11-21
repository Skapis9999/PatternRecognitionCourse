import pandas as pd
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

data = pd.read_csv("./2. Decision Trees/quiz_data.csv")

encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

absfreq = pd.crosstab(data.CarType, data.Insurance)
freq = pd.crosstab(data.CarType, data.Insurance, normalize='index')
freqSum = pd.crosstab(data.CarType, data.Insurance, normalize='all').sum(axis=1)
GINI_Sport = 1 - freq.loc["Sport", "No"]**2 - freq.loc["Sport", "Yes"]**2
GINI_Family = 1 - freq.loc["Family", "No"]**2 - freq.loc["Family", "Yes"]**2
GINI_Sedan = 1 - freq.loc["Sedan", "No"]**2 - freq.loc["Sedan", "Yes"]**2
GINI_CarType = freqSum.loc["Sport"] * GINI_Sport + freqSum["Family"] * GINI_Family + freqSum["Sedan"] * GINI_Sedan
print(GINI_CarType)

absfreq = pd.crosstab(data.Budget, data.Insurance)
freq = pd.crosstab(data.Budget, data.Insurance, normalize='index')
freqSum = pd.crosstab(data.Budget, data.Insurance, normalize='all').sum(axis=1)
GINI_Low = 1 - freq.loc["Low", "No"]**2 - freq.loc["Low", "Yes"]**2
GINI_Medium = 1 - freq.loc["Medium", "No"]**2 - freq.loc["Medium", "Yes"]**2
GINI_VeryHigh = 1 - freq.loc["VeryHigh", "No"]**2 - freq.loc["VeryHigh", "Yes"]**2
GINI_High = 1 - freq.loc["High", "No"]**2 - freq.loc["High", "Yes"]**2
GINI_Budget = freqSum.loc["Low"] * GINI_Low + freqSum["Medium"] * GINI_Medium + freqSum["High"] * GINI_High + freqSum["VeryHigh"] * GINI_VeryHigh
print(GINI_Budget)

absfreq = pd.crosstab(data.Sex, data.Insurance)
freq = pd.crosstab(data.Sex, data.Insurance, normalize='index')
freqSum = pd.crosstab(data.Sex, data.Insurance, normalize='all').sum(axis=1)
GINI_M = 1 - freq.loc["M", "No"]**2 - freq.loc["M", "Yes"]**2
GINI_F = 1 - freq.loc["F", "No"]**2 - freq.loc["F", "Yes"]**2
GINI_Sex = freqSum.loc["M"] * GINI_M + freqSum["F"] * GINI_F
print(GINI_Sex)

GINI_ID = 0 

GINI = [GINI_Sex,GINI_Sex, GINI_Budget, GINI_CarType,GINI_ID]
print(GINI)
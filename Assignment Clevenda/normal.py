# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 09:09:06 2018

@author: sunka
"""

import pandas as pd

import numpy as np

data = pd.read_csv('clevelanda.csv')

data.isnull().any()

type(data['age'][0])

list_i = []
list_j = []

for i in data.columns:
    for j in range(0,303):
        if data[i][j] == '?':
            data[i][j] = 0
            list_i.append(i)
            list_j.append(j)
        if type(data[i][j]) != np.int64:
            data[i][j] = int(data[i][j])

for i in list_i:
    for j in list_j:
        data[i][j] = data[i].mean()


x = data.iloc[:,:-1]
y = data.iloc[:,-1:]

from sklearn import model_selection

train_data, test_data, train_target, test_target = model_selection.train_test_split(x,y)

from sklearn import preprocessing

train_datas = preprocessing.normalize(train_data)
train_targets = preprocessing.normalize(train_target)
test_datas = preprocessing.normalize(test_data)
test_targets = preprocessing.normalize(test_target)

# DCTC - 100 - 55 -accu dtc1
# RFC - 10 -63 - accu dtc
# GBC - 2 - 57 - accu gbc
# LGR - - 63 - accu log
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(min_samples_split = 100) 

fitting = classifier.fit(train_data,train_target)

result = classifier.predict(test_data)

from sklearn import metrics

accu_dtc1 =  metrics.accuracy_score(result,test_target) 

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(min_samples_split = 10) 

fitting = classifier.fit(train_data,train_target)

result = classifier.predict(test_data)

from sklearn import metrics

accu_dtc =  metrics.accuracy_score(result,test_target)

from sklearn.ensemble import GradientBoostingClassifier

classifier = GradientBoostingClassifier(min_samples_split = 2) 

fitting = classifier.fit(train_data,train_target)

result = classifier.predict(test_data)

from sklearn import metrics

accu_gbc =  metrics.accuracy_score(result,test_target) 

from sklearn import linear_model

classifier = linear_model.LogisticRegression()
fitting = classifier.fit(train_data,train_target)
result = classifier.predict(test_data)

from sklearn import metrics

accu_log = metrics.accuracy_score(result,test_target)
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 12:17:50 2016

@author: Hanbin Seo
"""

import pandas as pd
dir_ = "../Data/"
data = pd.read_csv(dir_+"segmentation.data", sep=",")
data['target'] = data.index
data.index = list(range(len(data)))
feature_names = list(data.keys()[:19])
X, y = data[feature_names], data['target']

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100, stratify=y)

num_estimators = [1,5,10,20,30,50]
num_samples = 130
num_features = [10,15,19]


## library ===========================================

import random
def get_sample_set(T, N) :
    sample_set = []
    for i in range(T) :
       rand_num = random.sample(list(X_train.index), N)
       sample_set.append(rand_num)
    return sample_set
       
from pandas import DataFrame as df
def get_dataset(T, N) :
    sample_set = get_sample_set(T, N) 
    train_input, train_output = [], []
    for t in range(T) :
        tmp_data = df()
        for feature in feature_names :
            tmp_data[feature] = [X_train[feature][sample_set[t][i]] for i in range(N)]
        train_input.append(tmp_data)
        train_output.append([y_train[sample_set[t][i]] for i in range(N)])
    return train_input, train_output

from sklearn.tree import DecisionTreeClassifier
import numpy as np
from collections import Counter
def random_forest(input_, output_, F, T) :
    dtc = DecisionTreeClassifier(min_samples_split=10, max_features=F)
    pred_list = []
    for t in range(T) :
        dtc.fit(input_[t], output_[t])
        pred_ = dtc.predict(X_test)
        pred_list.append(pred_)
    tmp_pred = np.array(pred_list).T
    res_pred = []
    for i in range(len(tmp_pred)) :
        res_pred.append(Counter(tmp_pred[i]).most_common()[0][0])
    return res_pred, pred_list

def score_acc(pred_) :
    correct = 0
    for i in range(len(y_test)) :
        if pred_[i] == list(y_test)[i] :
            correct += 1
    return correct/len(y_test)

## library (end) ===========================================

for T in num_estimators :
    for F in num_features :
        input_, output_ = get_dataset(T, num_samples)
        res_pred, est_pred = random_forest(input_, output_, F, T)
        print("num_estimators: ", T, "& num_features: ", F, "\tScore_Acc: ", score_acc(res_pred))


""" Result (The number of samples in each subset = 130)

num_estimators:  1 & num_features:  10  Score_Acc:  0.8333333333333334
num_estimators:  1 & num_features:  15  Score_Acc:  0.9047619047619048
num_estimators:  1 & num_features:  19  Score_Acc:  0.9285714285714286
num_estimators:  5 & num_features:  10  Score_Acc:  0.9285714285714286
num_estimators:  5 & num_features:  15  Score_Acc:  0.9047619047619048
num_estimators:  5 & num_features:  19  Score_Acc:  0.9047619047619048
num_estimators:  10 & num_features:  10         Score_Acc:  0.8809523809523809
num_estimators:  10 & num_features:  15         Score_Acc:  0.8809523809523809
num_estimators:  10 & num_features:  19         Score_Acc:  0.9047619047619048
num_estimators:  20 & num_features:  10         Score_Acc:  0.9047619047619048
num_estimators:  20 & num_features:  15         Score_Acc:  0.9047619047619048
num_estimators:  20 & num_features:  19         Score_Acc:  0.9047619047619048
num_estimators:  30 & num_features:  10         Score_Acc:  0.9047619047619048
num_estimators:  30 & num_features:  15         Score_Acc:  0.9047619047619048
num_estimators:  30 & num_features:  19         Score_Acc:  0.9047619047619048
num_estimators:  50 & num_features:  10         Score_Acc:  0.9047619047619048
num_estimators:  50 & num_features:  15         Score_Acc:  0.9047619047619048
num_estimators:  50 & num_features:  19         Score_Acc:  0.9047619047619048

"""




# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 22:36:38 2016

@author: Hanbin Seo
"""

## data import 
import pandas as pd
dir_ = "../Data/"
data = pd.read_csv(dir_+"segmentation.data", sep=",")
data['target'] = data.index
data.index = list(range(len(data)))

feature_names = list(data.keys()[:19])
X, y = data[feature_names], data['target']
targets = set(data['target'])

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, random_state=100, stratify=y)
print(set(y_train), "\n", set(y_test))

## function =========================

def replace_label(obj, trg) :
    if obj == trg :
        return obj
    else :
        return 'the_rest'
        
def score_acc(pred_, y_true) :
    correct = 0
    for i in range(len(y_true)) :
        if pred_[i] == list(y_true)[i] :
            correct += 1
    return correct/len(y_true)

import random
def pick_label(items) :
    tmp_res = []
    for tg in targets :
        if tg in list(items) :
            tmp_res.append(tg)
    if len(set(tmp_res)) == 1 :
        return tmp_res[0]
    elif len(set(tmp_res)) == 0 :
        return None
    else :
        return random.choice(tmp_res)
            
## fuction(end) =========================


## one-versus-rest approach
y_train_ovr = [[replace_label(yt, lb) for yt in y_train] for lb in set(y_train)]
for rt in y_train_ovr :
    print(set(rt))

from sklearn import svm
pred_ovr = []
for idx in range(len(y_train_ovr)) :
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train_ovr[idx])
    pred = clf.predict(X_test)
    pred_ovr.append(pred)

from pandas import DataFrame as df
temp_df = df(pred_ovr)
predict_ovr = [pick_label(temp_df[idx]) for idx in range(len(temp_df.T))]
print(set(predict_ovr))
print("ACC: ", score_acc(predict_ovr, y_test))


## one-versus-one approach
pairs = []
for i in range(len(targets)) :
    for i0 in range(i+1, len(targets)) :
        print(i, i0, '\t', list(targets)[i], list(targets)[i0])
        pairs.append([list(targets)[i], list(targets)[i0]])

train_idx = [[idx for idx in y_train.index if y_train[idx] in pair] for pair in pairs]
X_train_df = [X_train.loc[idxs, :] for idxs in train_idx]
y_train_df = [y_train.loc[idxs] for idxs in train_idx]

from sklearn import svm
pred_ovo = []
for idx in range(len(pairs)) :
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train_df[idx], y_train_df[idx])
    pred = clf.predict(X_test)
    pred_ovo.append(list(pred))

from pandas import DataFrame as df
temp_df = df(pred_ovo)
from collections import Counter
predict_ovo = [Counter(temp_df[idx]).most_common()[0][0] for idx in range(len(temp_df.T))]
print("ACC: ", score_acc(predict_ovo, y_test))


## multi-classfication using package (부록) ============================
# (1) one-versus-rest approach
from sklearn import svm
import time
clf = svm.LinearSVC(multi_class='ovr')
t0 = time.time()
clf.fit(X_train, y_train)
t = time.time() - t0
pred = clf.predict(X_test)
print("fitting time: ", t, "\npredicted class: ", set(pred), "\nACC: ", score_acc(pred, y_test))

# (2) one-versus-one approach
clf = svm.LinearSVC(multi_class='crammer_singer')
t0 = time.time()
clf.fit(X_train, y_train)
t = time.time() - t0
pred = clf.predict(X_test)
print("fitting time: ", t, "\npredicted class: ", set(pred), "\nACC: ", score_acc(pred, y_test))



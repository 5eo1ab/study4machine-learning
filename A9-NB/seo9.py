# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 13:18:26 2016

@author: Hanbin Seo
"""

### data import and manipulation =================
import pandas as pd
from pandas import DataFrame as df
import numpy as np
dir = "../Data/"
meta_data = pd.read_csv(dir+'metadata-uci-mushroom.csv', names=['name', 'description'])
raw_data = pd.read_csv(dir+'mushroom_agaricus-lepiota.data', names=meta_data['name'])

def set_replace_dic(classes):
    classes, tmp_dic, r_num = set(classes), {}, 0
    for cl in classes :
        tmp_dic[cl] = r_num
        r_num += 1
    return tmp_dic
def get_replace_val(array, rep_dic):
    new_arr = []
    for a in array :
        new_arr.append(rep_dic[a])
    return new_arr

meta_data['replace_dic'] = [set_replace_dic(raw_data[feat]) for feat in meta_data['name']]
rep_data, num_outcome = df(), []
for idx in range(len(meta_data)) :
    feat_name, tmp_dic = meta_data['name'][idx], meta_data['replace_dic'][idx]   
    print("idx:", idx, "\tfeature:", feat_name, "\ndic:", tmp_dic, "\n")
    rep_data[feat_name] = get_replace_val(raw_data[feat_name], tmp_dic)
    num_outcome.append(len(tmp_dic))
meta_data['num_outcome'] = num_outcome

data = rep_data[meta_data['name'][1:]]
label = rep_data[meta_data['name'][0]]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(data, label, test_size=0.2, random_state=100, stratify=label)
print(len(y_train), ",\t", len(y_test))

### naive bayes classification using pure-implementation =================
priors = {}
for cl in set(y_train) :
        priors[cl] = len(y_train.loc[y_train==cl])/len(y_train)
 
X_train_c0, X_train_c1 = df(), df()
idx_c0 = y_train.loc[y_train==0].index.values
idx_c1 = y_train.loc[y_train==1].index.values
for idx in range(1, len(meta_data)):
    feat_nm = meta_data['name'][idx]    
    X_train_c0[feat_nm] = [X_train[feat_nm][idx] for idx in idx_c0]
    X_train_c1[feat_nm] = [X_train[feat_nm][idx] for idx in idx_c1]
X_train_c0 = X_train_c0.set_index(idx_c0)
X_train_c1 = X_train_c1.set_index(idx_c1)

def get_likelihood(array, meta_idx):
    tmp_dic = {}
    for key in range(meta_data['num_outcome'][meta_idx]):
        if len(array.loc[array==key]) > 1 :
            tmp_dic[key] = len(array.loc[array==key])/len(array)
        else :
            tmp_dic[key] = 0.01 # Laplace value for Nominal variable value that does not appear
    return tmp_dic

likelihoods = df()
for idx in range(1, len(meta_data)):
    feat_nm = meta_data['name'][idx]
    tgt_df = X_train_c0, X_train_c1 
    likelihoods[feat_nm] = [get_likelihood(t_df[feat_nm], idx) for t_df in tgt_df]

# predict class
idx_test = X_test.index.values
def naive_classifier(idx):
    likelihood_, prob0, prob1 = [], priors[0], priors[1]
    for j in range(1, len(meta_data)):
        feat_nm = meta_data['name'][j]
        val = X_test[feat_nm][idx]
        likelihood_.append((likelihoods[feat_nm][0][val], likelihoods[feat_nm][1][val]))
    for likelihood in likelihood_ :
        prob0 *= likelihood[0]
        prob1 *= likelihood[1]
    decision = 1
    if prob0 > prob1 :
        decision = 0
    return {'predict':decision, 'probability':(prob0, prob1)}

pred_df = df()
pred_df['predict'] = [naive_classifier(idx)['predict'] for idx in idx_test]
pred_df['probability'] = [naive_classifier(idx)['probability'] for idx in idx_test]
pred_df = pred_df.set_index(idx_test)

score = sum([1 for idx in idx_test if pred_df['predict'][idx] == y_test[idx]])/len(idx_test)
print("score: ", score)


### naive bayes classification using 'sklearn' package =================
from sklearn import naive_bayes
nb = naive_bayes.MultinomialNB()
nb.fit(X_train, y_train)
nb.score(X_test,y_test)



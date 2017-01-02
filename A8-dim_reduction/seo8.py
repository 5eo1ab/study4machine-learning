# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 15:16:50 2016

@author: Hanbin Seo
"""

### import data
import urllib.request
import numpy as np
X_train, y_train = None, None
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/arcene/ARCENE/arcene_train.data"
with urllib.request.urlopen(url) as respone :
    tmp_data = respone.read().split()
    tmp_data = [int(val) for val in tmp_data]
    print(sum(tmp_data), "\tCheck_sum value : 70726744.00")
    X_train = np.array(tmp_data).reshape(len(tmp_data)/10000, 10000) # Number of features: 10000
    print(len(X_train), "\tCount of Tot_ex : 100\n")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/arcene/ARCENE/arcene_train.labels"
with urllib.request.urlopen(url) as respone :
    tmp_data = respone.read().split()
    y_train = [int(val) for val in tmp_data]
    print(len(y_train), "\tcount of Tot_ex : 100")    
X_test, y_test = None, None
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/arcene/ARCENE/arcene_valid.data"
with urllib.request.urlopen(url) as respone :
    tmp_data = respone.read().split()
    tmp_data = [int(val) for val in tmp_data]
    print(sum(tmp_data), "\tCheck_sum value : 71410108.00")
    X_test = np.array(tmp_data).reshape(len(tmp_data)/10000, 10000) # Number of features: 10000
    print(len(X_test), "\tCount of Tot_ex : 100\n")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/arcene/arcene_valid.labels"
with urllib.request.urlopen(url) as respone :
    tmp_data = respone.read().split()
    y_test = [int(val) for val in tmp_data]
    print(len(y_test), "\tcount of Tot_ex : 100") 


### build k-neighClassifier
from sklearn.neighbors import KNeighborsClassifier
def neighClassifier(train_input, train_label, test_input, test_label) :
    neigh = KNeighborsClassifier(n_neighbors=5)  
    neigh.fit(train_input, train_label)
    res = {'acc_score':neigh.score(test_input, test_label), 'pred_val':neigh.predict(test_input)}
    res['dimension'] = len(train_input.T)
    return res

### apply k-NN classifier on overall data    
result_dic = {'origin_data':neighClassifier(X_train, y_train, X_test, y_test)}

### build dimension reducer
import pandas as pd
from pandas import DataFrame as df
data_input = pd.concat([df(X_train), df(X_test)], keys=['train', 'test'])

from sklearn import decomposition
from sklearn import manifold
def demension_reducer(type_, n_dimension) :
    reducer = None
    if type_ is 'pca' :
        reducer = decomposition.PCA(n_components=n_dimension)
    elif type_ is 'kernel_pca' :
        reducer = decomposition.KernelPCA(kernel='rbf', n_components=n_dimension) # because default kernel is 'linear'
    elif type_ is 'isomap' :
        reducer = manifold.Isomap(n_components=n_dimension)
    elif type_ is 'lle' :
        reducer = manifold.LocallyLinearEmbedding(n_components=n_dimension)
    else :
        print("you must define reducer and can choice one: ('pca', 'kernel_pca', 'isomap', 'lle')")
        return None
    try :
        reduced_data = reducer.fit_transform(data_input)
        reduced_train = df(reduced_data).loc[:len(X_test)-1,:]
        reduced_test = df(reduced_data).loc[len(X_test):,:]
    except ValueError as e :
        print("\n\nError Message: ", e,"\nError occur @", (type_,n_dimension),"\n")
    return reduced_train, reduced_test


### apply k-NN classigier on reduced data
dimensions = [10, 50, 100]
reducers = ['pca', 'kernel_pca', 'isomap', 'lle']
params_li = [('%s_%d'%(r,d) ,r,d) for r in reducers for d in dimensions]
for params in params_li :
    k, r, d = params
    t, v = demension_reducer(r,d)
    result_dic[k] = neighClassifier(t, y_train, v, y_test)
    result_dic[k]['reducer'] = r
result_df = df(result_dic).T    
#result_df.loc[result_df['reducer']=='pca']

### visuaize result
result_vis = result_df.drop('origin_data')
result_vis['log_dim'] = result_vis['dimension'].apply(lambda x:np.log(x))
result_vis['exp_acc'] = result_vis['acc_score'].apply(lambda x:np.exp(x))
def vis_df(type_) :
    tmp_df = result_vis.loc[result_vis['reducer']==type_].sort(['dimension'])
    return tmp_df    
x_min, x_max = result_vis['log_dim'].min()-0.1, result_vis['log_dim'].max()+0.1
y_min, y_max = result_vis['exp_acc'].min()-0.1, result_vis['exp_acc'].max()+0.1
xlin, ylin = np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200)

import matplotlib.pyplot as plt
plt.figure(figsize=(5,8))
plt.plot(xlin,np.exp([result_df['acc_score']['origin_data'] for i in range(200)]), '--', color = '0.65')
plt.scatter(result_vis['log_dim'], result_vis['exp_acc'])
for r in reducers :
    plt.plot(vis_df(r)['log_dim'], vis_df(r)['exp_acc'], 'b--')
    plt.text(vis_df(r)['log_dim'][0]+0.1, vis_df(r)['exp_acc'][0]+0.01, vis_df(r)['acc_score'][0])
    plt.text(vis_df(r)['log_dim'][1]+0.1, vis_df(r)['exp_acc'][1]+0.01, r)
plt.xlabel("log_dim")
plt.ylabel("exp_acc")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()




# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 16:32:38 2016

@author: Hanbin Seo
"""

## Creating dataset

import numpy as np
x = np.random.uniform(-4, 4, 100)   # train dataset
x0 = np.linspace(-4,4,100)          # test dataset
y = np.sin(x) + np.random.normal(size=100, scale=0.4)

## library ===========================================

import random
def get_sample_set(T, N) :
    sample_set = []
    for i in range(T) :
       rand_num = [int(random.uniform(0,N)) for i in range(N)]
       sample_set.append(rand_num)
    return sample_set

from pandas import DataFrame as df
def get_dataset(T, N, sample_set) :
    data = []
    for t in range(T) :
        tmp_data = df({
        'X' : {i : x[sample_set[t][i]] for i in range(N)},
        'Y' : {i : y[sample_set[t][i]] for i in range(N)}
        })
        data.append(tmp_data)
    return data

def knn_reg(k, df, x_test) :
    res_y = []
    for xt in x_test :
        dis = np.sqrt(abs(xt-df['X']))
        df['distance'] = dis
        tmp_df = df.sort_values('distance')
        #res_y.append(np.average(tmp_df['Y'][:k]))
        res_y.append(np.average(tmp_df['Y'][:k], weights=np.reciprocal(tmp_df['distance'][:k])))
        del df['distance']
    return res_y

def bagging_knn_reg(T, k, DF, x_test) :
    each_y = []
    for t in range(T) :
        tmp_y = knn_reg(k, DF[t], x_test)
        each_y.append(tmp_y)
    bagging_y = np.array(each_y)
    bagging_y = np.average(bagging_y.T, axis=1)
    return bagging_y, each_y

import matplotlib.pyplot as plt
def visualization_bagging(X, Y, X_test, T, bagging_fit, bagging_est, plt) :   
    plt.scatter(X,Y)
    for i in range(T) :
        plt.plot(X_test, bagging_est[i], 'r:')
    plt.plot(X_test, bagging_fit, 'b', lw=2)
    plt.plot(X_test, np.sin(X_test), 'g')
    plt.text(1.5, -1.7, "n_estimators: %d"%T, fontsize=12)
    return None

## library (end) ===========================================

n_estimators = [10, 30, 50]
N, k = len(x), 3
f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
f.set_size_inches(7,12)
axs = [ax1, ax2, ax3]
for i in range(len(n_estimators)) :
    T = n_estimators[i]
    sample_set = get_sample_set(T, N)
    data = get_dataset(T, N, sample_set)
    bagging = bagging_knn_reg(T, k, data, x0)
    bagging_fit = bagging[0]
    bagging_est = bagging[1]
    visualization_bagging(x, y, x0, T, bagging_fit, bagging_est, axs[i])
f.subplots_adjust(hspace=0.01)
plt.show()


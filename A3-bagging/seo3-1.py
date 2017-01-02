# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 00:21:40 2016

@author: Hanbin Seo
"""

from sklearn import datasets
iris = datasets.load_iris()
x = iris['data']
target_ = iris['target']


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
        'X' : {i : x.T[0][sample_set[t][i]] for i in range(N)},
        'Y' : {i : x.T[1][sample_set[t][i]] for i in range(N)},
        'target' : {i : target_[sample_set[t][i]] for i in range(N)}
        })
        data.append(tmp_data)
    return data
 
import numpy as np   
def get_meshgrid(DF, len_) :
    #arr_x, arr_y = df['X'], df['Y']
    #x_min, x_max = arr_x.min()-0.25, arr_x.max()+0.25
    #y_min, y_max = arr_y.min()-0.25, arr_y.max()+0.25
    x_min = min([DF[t]['X'].min()  for t in range(len(DF))])-0.25
    x_max = max([DF[t]['X'].max()  for t in range(len(DF))])+0.25
    y_min = min([DF[t]['Y'].min()  for t in range(len(DF))])-0.25
    y_max = max([DF[t]['Y'].max()  for t in range(len(DF))])+0.25
    xs, ys = np.linspace(x_min, x_max, len_), np.linspace(y_min, y_max, len_)
    XX, YY = np.meshgrid(xs, ys)
    return XX, YY
    
def knn_cl(XX, YY, df) :
    pred_ = []
    for i in range(len(XX)):
        pred_tmp = []
        for j in range(len(XX[0])) :
            dis = np.sqrt((XX[i][j]-df['X'])**2 + (YY[i][j]-df['Y'])**2)
            df['distance'] = dis
            tmp_df = df.sort_values('distance')
            tmp_df[:k]['target'].item()     # pandas series datatype
            pred_tmp.append(tmp_df['target'][:k].item())
            del df['distance'] 
        pred_.append(pred_tmp)
    return pred_

from collections import Counter
def bagging_knn_cl(T, XX, YY, sample_set, DF) :   
    each_zz = []
    for t in range(T) :
        tmp_zz = knn_cl(XX, YY, DF[t])
        np_zz = np.array(tmp_zz).ravel()      
        each_zz.append(np_zz)
    bagging_zz = []
    tmp_zz = np.array(each_zz).T
    for i in range(len(tmp_zz)) :
        if Counter(tmp_zz[0]).most_common()[0][1] != 1 :
            bagging_zz.append(Counter(tmp_zz[i]).most_common()[0][0])
        else :
            bagging_zz.append(random.choice([0, 1, 2]))
    bagging_zz = np.array(bagging_zz)
    return bagging_zz, each_zz

import matplotlib.pyplot as plt    
def visualization_bagging(XX, YY, x, T, bagging_fit, bagging_est) :
    f, (ax_fianl, ax_overlap, ax0, ax1, ax2) = plt.subplots(5, sharex=True, sharey=True)
    f.set_size_inches(7,7*5)
    ax_list = [ax_fianl, ax_overlap, ax0, ax1, ax2]
    ax_fianl.contourf(XX,YY ,bagging_fit, cmap=plt.cm.RdYlBu, alpha=0.8)
    ax_fianl.text(6.3, 4.5, "bagging final model", fontsize=15)
    for t in range(T) :
        ax_overlap.contourf(XX,YY ,bagging_est[t], cmap=plt.cm.RdYlBu, alpha=0.4)
        ax_list[t+2].contourf(XX,YY ,bagging_est[t], cmap=plt.cm.RdYlBu, alpha=0.7)
        ax_list[t+2].text(6.5, 4.5, "bagging model %d"%t, fontsize=15)
    ax_overlap.text(6.3, 4.5, "bagging overlapped", fontsize=15)
    for ax in ax_list :
        ax.scatter(x[:,0], x[:,1], c=target_, s=30)    
    f.subplots_adjust(hspace=0.01)
    plt.show()
    return None ;
    
## library (end) ===========================================

T, N, k, len_ = 3, len(x), 1, 100
sample_set = get_sample_set(T, N)
data = get_dataset(T, N, sample_set)
XX, YY = get_meshgrid(data, len_)  
bagging = bagging_knn_cl(T, XX, YY, sample_set, data)
bagging_fit = bagging[0].reshape(len_, len_)
bagging_est = [est.reshape(len_, len_) for est in bagging[1]]
visualization_bagging(XX, YY, x, T, bagging_fit, bagging_est)




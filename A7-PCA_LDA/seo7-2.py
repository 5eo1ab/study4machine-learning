# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 02:17:43 2016

@author: Hanbin Seo
"""
## LDA

# data import 
from sklearn import datasets
data = datasets.load_iris()
x_raw = data.data
y = data.target

# calculate mean of overall
import numpy as np
X = np.array(x_raw)
means_al = [np.mean(X.T[idx]) for idx in range(len(X.T))]
means_al = np.array(means_al)

# calculate mean of each class
from pandas import DataFrame as df
X_df = df(X)
idx_class = [[i for i in range(len(y)) if lb==y[i]] for lb in set(y)]
X_class = [X_df.loc[idxs, :] for idxs in idx_class]
means_cl = [np.mean(X_class[lb]) for lb in set(y)]
means_cl = [np.array(means_cl[idx]) for idx in range(len(means_cl))]

# calculate "within class scatter"
cov_mat = None
for cl in set(y) :
    mat = np.array(X_class[cl]-means_cl[cl]).T # column vectorizer
    if cov_mat is None :
        cov_mat = np.dot(mat, mat.T)
    else :
        cov_mat = np.add(cov_mat, np.dot(mat, mat.T))

# calculate "between class scatter"
between = None
for cl in set(y) :
    vec = np.array(means_cl[cl]-means_al).reshape(len(means_al),1)
    if between is None :
        between = len(X_class[cl])*np.dot(vec, vec.T)
    else :
        between = np.add(between, len(X_class[cl])*np.dot(vec, vec.T))

# calculate lamda(eigen-value)
cov_inv = np.linalg.inv(cov_mat)
matrix = np.dot(cov_inv, between)
eigen_val, eigen_vec = np.linalg.eig(matrix)
lamda = [val for val in eigen_val if val > 0 and abs(val)>1/(10**10)]
print("non-zero eigenvalues are :\n", lamda)

# Transform
n_component = len(lamda)
W_tmp = [eigen_vec[:,i] for i in range(n_component)]
W = np.array(W_tmp).T
T = np.dot(X, W)

# Visualization
from matplotlib import pyplot as plt
fig = plt.figure()
plt.scatter(X[:,0], X[:,1], s=100, c=y)
plt.show()
fig = plt.figure()
plt.scatter(T[:,0], T[:,1], s=100, c=y)
plt.show()



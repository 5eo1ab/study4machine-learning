# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 21:05:26 2016

@author: Hanbin Seo
"""
## PCA

# data import 
from sklearn import datasets
data = datasets.load_iris()
x_raw = data.data
y = data.target

# Standardization of variables
import numpy as np
means = [np.mean(x_raw.T[idx]) for idx in range(len(x_raw.T))]
sd = [np.std(x_raw.T[idx]) for idx in range(len(x_raw.T))]
X = [[(row[i]-means[i])/sd[i] for i in range(len(row))] for row in x_raw]
X = np.array(X)

# Calculate of eigenvalues and eigenvectors
cov_mat = np.dot(X.T, X)
eigen_val, eigen_vec = np.linalg.eigh(cov_mat) # the eigenvalues and eigenvectors of a symmetric matrix
sort_idx = np.argsort(eigen_val)[::-1] # order by DESC

# Transform
n_component = 2
W_tmp = [eigen_vec[:, sort_idx[i]] for i in range(n_component)]
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

## test =========
print([np.mean(X.T[i]) for i in range(len(X.T))], "~ almost equal 0")
print([np.var(X.T[i]) for i in range(len(X.T))], "~ almost equal 1")
print("dot product of w1.T and w1: ", np.dot(eigen_vec[0].T,eigen_vec[0]), "~ equal 1")
print("dot product of w2.T and w1: ", np.dot(eigen_vec[1].T,eigen_vec[0]), "~ almost equal 0")
print("np.dot(eigen_vec.T, eigen_vec):\n", np.dot(eigen_vec.T, eigen_vec), "\nnp.dot(W.T, W):\n", np.dot(W.T, W))
np.subtract(cov_mat,np.diag(eigen_val))
np.dot(np.subtract(cov_mat,np.diag(eigen_val)), eigen_vec)
np.identity(3)


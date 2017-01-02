# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 01:15:58 2016

@author: Hanbin Seo
"""

import pandas as pd
import numpy as np
from scipy import stats 
import matplotlib.pyplot as plt

dir_ = '../Data/'
data = pd.read_table(dir_+'kernel_smoothing_test.txt', names=['X', 'Y'])
input_ = data['X']
output_ = data['Y']

def kernel(p, s, h) :
    u = (p-s)/h
    if abs(u) <= 1 :
        return 3/4 * (1-u**2)
    else :
        return 0

def func(p, mode) :
    w0, w1, idx = 0, 0, 0
    for s in input_ :
       if mode == 0 :  # gaussian
            w0 += stats.norm.pdf(p, loc=s, scale=1)  
       elif mode == 1 :# epanechnikov
            w0 += kernel(p, s, 1)
    for s in input_ :
       if mode == 0 :   # gaussian
            w1 += stats.norm.pdf(p, loc=s, scale=1) * output_[idx]
       elif mode == 1 : # epanechnikov
            w1 += kernel(p, s, 1) * output_[idx]
       idx += 1
    return w1/w0
  
smooth_g = []
smooth_e = []
for x in input_ :
    smooth_g.append(func(x, 0))
    smooth_e.append(func(x, 1))


plt.plot(input_,smooth_g, color='green', linestyle='-', label="gaussian")
plt.plot(input_,smooth_e, color='red', linestyle='--', linewidth=2, label="epanechnikov")
plt.legend(loc='lower right', fontsize=10)
plt.scatter(input_,output_)
plt.show()

# the result :
# kernel smoothing with epanechnikov kernel is more effective than using gaussian kernel 
# from the viewpoint of explain data.




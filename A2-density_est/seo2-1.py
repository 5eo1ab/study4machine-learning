# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 21:34:11 2016

@author: Hanbin Seo
"""

import numpy as np
from scipy import stats
import math

sample = [2,3,4,8,10,11,12]
h = 1   # bandwidth

def kernel(p, h, data) :
    w = 0    
    for s in data :
        if abs((s-p)/h) <= 1 :      # "cosine" window function
            w += math.pi/4 * math.cos(math.pi/2*abs((s-p)/h))  
    w = w/(len(data)*h)
    return w
    
import matplotlib.pyplot as plt
x = np.linspace(0, 14, 200)
y = []
for i in x :
    y.append(kernel(i, 1, sample))
plt.plot(x,y)
plt.show()


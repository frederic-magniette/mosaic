# This function finds local minima of a given 2d array.
# It is a convenience function, used for visualisation purposes.

import numpy as np
import matplotlib.pyplot as plt

def find_local_minima(array):
    mins = 0*array
    ix, iy = [],[]
    for i in range(np.shape(array)[0]):
        for j in range(np.shape(array)[1]):
            val = array[i,j]

            n,s,e,w = np.max(array),np.max(array),np.max(array),np.max(array)

            if i > 0:
                n = array[i-1,j]
            if i != np.shape(array)[0]-1:
                s = array[i+1,j]
            if j > 0:
                w = array[i,j-1]
            if j != np.shape(array)[1]-1:
                e = array[i,j+1]

            if (val < n) and (val < s) and (val < e) and (val < w):
                mins[i,j]=1
                ix.append(j)
                iy.append(i)
    return mins, ix, iy

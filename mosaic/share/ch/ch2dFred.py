import numpy as np
import matplotlib.pyplot as plt

# This for the grid search of Fred's 2d classification challenges

def ch1_class1(x,y):
    x0, y0 = 0, 4 
    S  = np.array([[0.4,0.3],[0.3,0.4]])
    res = (x-x0)**2 * S[0,0] + (y-y0)**2 * S[1,1] \
            + (x-x0)*(y-y0) * (S[0,1] + S[1,0])
    return np.exp(-res)

def ch1_class2(x,y):
    x0, y0 = 3.4, 3.4 
    S  = np.array([[0.4,0.3],[0.3,0.4]])
    res = (x-x0)**2 * S[0,0] + (y-y0)**2 * S[1,1] \
            + (x-x0)*(y-y0) * (S[0,1] + S[1,0])
    return np.exp(-res)

x = np.linspace(-5,8,100)
y = np.linspace(0,8,100)
X, Y = np.meshgrid(x,y)
plt.imshow(ch1_class1(X,Y) + ch1_class2(X,Y))#, origin='lower')
plt.show()

V1 = ch1_class1(X,Y)
V2 = ch1_class2(X,Y)

cat = V2 / (V1 + V2) # <0.5 if in class1, >0.5 if in class 2
plt.imshow(cat)
plt.colorbar()
plt.show()

def find_boundary(vals1, vals2):
    diff = np.abs(vals1-vals2)
    nrows = np.shape(vals1)[1]
    idxs = []
    boundary = np.zeros(np.shape(vals1))
    for i in range(nrows):
        idx = np.where(diff[i,:] == np.min(diff[i,:]))[0][0]
        idxs.append(idx)
        boundary[i,idx] = 1
    return boundary, idxs
b, i = find_boundary(V1, V2)
plt.imshow(b)
plt.scatter(x,x,i)
plt.show()


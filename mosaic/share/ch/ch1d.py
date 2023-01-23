import numpy as np

# This file contains several one-dimensional functions that are useful and/or interesting for studying the performances of QML circuits

# Most are very simple and don't require to be declared in this specific file, but this is done to ensure exhaustivity and modularity
def normalise(res):
    ''' normalise values to [-1;1] '''
    b, a = np.max(res), np.min(res)
    y = (res - a) / (b-a) # normalise to [0;1]
    return 2*y - 1

def crenel(x):
    return np.heaviside(-x+0.5, 0.0) * np.heaviside(x+0.5, 0.0)

#def expFred(x): # should be plotted over [-4; 10]
#    res = np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2 / 10) + 1.0/(x**2 + 1.0)
#    return -1.0 * res
def expFred(x): # expFred rescaled to [-1,1]
    y = 7 * x + 3
    res = np.exp(-(y - 2)**2) + np.exp(-(y - 6)**2 / 10) + 1.0/(y**2 + 1.0)
    res = normalise(res)
    return -1.0 * res

def gauss(x, m=0, s=0.3): return np.exp(-(x-m)**2 / (2 * s**2))

def poly_PS(x): #7th order polynomial of Perez-Salinas et al., https://arxiv.org/pdf/2102.04032.pdf 
    return np.abs(3 * x**3 * (1 - x**4))
def reLU(x): return x * np.heaviside(x, 0.0)
def reLUreg(x): return x * np.heaviside(x, 0.0) / np.pi
def sigm(x, la=np.pi): return 1.0 / (1.0 + np.exp(-la*x))
def sin(x, la=np.pi): return np.sin(la * x)
def sphere(x): return x*x
def step(x): return np.heaviside(x, 0.0)
def tanh(x, la=np.pi): return np.tanh(x*la)


#c = [] #patch to generate a random poly 
#def poly20(x):
#    if c == []:
#        c = np.random.rand(21) * 2 - 1 #random coeffs in [-1, 1]
#    p = np.poly1d(c)
#    return p(x)

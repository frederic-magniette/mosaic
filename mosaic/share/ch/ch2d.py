import numpy as np
import ch1d # this to define some functions systematically from their 1d equivalent

# This file contains several two-dimensional functions that are useful and/or interesting for studying the performances of QML circuits

# Most are very simple and don't require to be declared in this specific file, but this is done to ensure exhaustivity and modularity
def normalise(res):
    ''' normalise values to [-1;1] '''
    b, a = np.max(res), np.min(res)
    y = (res - a) / (b-a) # normalise to [0;1]
    return 2*y - 1

def crenel(x,y): 
    return ch1d.crenel(x) * ch1d.crenel(y)
def expFred(x,y):
    return ch1d.expFred(x) * ch1d.expFred(y)
def gauss(x, y, mx=0, my=0, sx=1, sy=1): return np.exp(-((x-mx)**2 / (2*sx**2) + (y-my)**2 / (2*sy**2)) )

def poly_PS(x,y): #7th order polynomial of Perez-Salinas et al., https://arxiv.org/pdf/2102.04032.pdf 
    return ch1d.poly_PS(x) * ch1d.poly_PS(y)
def reLU(x, y): 
    return x * np.heaviside(x, 0.0) * y * np.heaviside(y, 0.0)
def sigm(x,y,la=1): 
    return ch1d.sigm(x) * ch1d.sigm(y)
def sin(x, y, la=np.pi): 
    return np.sin(la * (x+y))
def step(x,y): 
    return ch1d.step(x) * ch1d.step(y)#np.heaviside(x, 0.0) * np.heaviside(y, 0.0)
def tanh(x,y): 
    return ch1d.tanh(x) * ch1d.tanh(y)

# some standard test functions, see https://en.wikipedia.org/wiki/Test_functions_for_optimization
# the functions are properly rescaled to [-1;1] when possible
def rastrigin(x,y):
    x, y = 5.12*x, 5.12*y 
    A = 10
    return normalise(2 * A + x**2 - A * np.cos(2*np.pi*x) + y**2 - A * np.cos(2*np.pi*y))

def ackley(x,y):
    x, y = 5*x, 5*y
    res = -20 * np.exp(-0.2 * np.sqrt(0.5*(x**2 + y**2)))
    res -= np.exp(0.5 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y)))
    res += np.e + 20
    return normalise(res)

def sphere(x,y):
    return normalise(x**2 + y**2)

def rosenbrock(x,y):
    return normalise(100 * (y-x**2)**2 + (1-x)**2)

def himmelblau(x,y):
    x, y = 5*x, 5*y
    return normalise((x**2 + y - 11)**2 + (x + y**2 - 7)**2)

# Frederic Magniette's classification challenges, grid plot version (see 2Dchallenges/generation.py for comparison)
def ch1(x,y):
    gaussian = lambda x,y: np.exp(-0.5 * (x**2 * S[0,0] + y**2 * S[1,1]
        + x*y * S[0,1] + y*x * S[1,0])) 
    
    mean1 = [-0.4,0]
    cov1 = [[0.04,0.03],[0.03,0.04]]
    S = np.linalg.inv(cov1)
    X = x - mean1[0]
    Y = y - mean1[1]
    print(S)
    #z1 = np.exp(-0.5 * (X.T * X * S[0,0] + 0*X.T * Y * S[0,1]
    #    + Y.T * X * S[1,0] 
    #    + Y.T * Y * S[1,1]))
    z1 = gaussian(X,Y)

    mean2 = [.4,0]
    cov2 = [[0.04,0.03],[0.03,0.04]]
    S = np.linalg.inv(cov2)
    X = x - mean2[0]
    Y = y - mean2[1]
    z2 = gaussian(X,Y)
    which = float(z1 > z2)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,2)
    im = ax[0].imshow(z1+z2)
    fig.colorbar(im, ax=ax[0])
    im = ax[1].imshow(which)
    plt.show()

    return which     
    








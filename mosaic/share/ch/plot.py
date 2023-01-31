import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import plt_params
import seaborn as sns
cmap = sns.cubehelix_palette(start=1, rot=-1, hue=2.5, light=0, dark=1, gamma=.8, reverse=True, as_cmap=True)

name = "ackley2"
name = "ch1"
#if name == "ch1": func = ch.ch1
#elif name == "ackley2": func = ch2d.ackley
N = 50 # nbr of retained data points


#x = np.linspace(-1,1,100)
#X, Y = np.meshgrid(x,x)
#fgrid = func(X,Y)

# get a subset of the data
cwd = os.getcwd() + "/"
f = h5py.File(cwd+name+".h5", "r")
in_data, out_data = f.get('input')[:50],f.get('output')[:50]
f.close()

# fetch the data computed on the grid
print(cwd+name+"_grid.h5")
fgrid = h5py.File(cwd+name+"_grid.h5", "r")
in_grid, out_grid = fgrid.get('input')[:],fgrid.get('output')[:]
fgrid.close()

x = np.array([d[0] for d in in_data])
y = np.array([d[1] for d in in_data])
xg = np.array([d[0] for d in in_grid])
yg = np.array([d[1] for d in in_grid])

plt.scatter(x, y, marker='X', color='k', edgecolors='w', label='sampled')
plt.imshow(out_grid, extent=[-1,1,-1,1], origin="lower", cmap=cmap)

plt.show()

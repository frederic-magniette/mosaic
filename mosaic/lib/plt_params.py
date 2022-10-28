# See https://matplotlib.org/stable/tutorials/introductory/customizing.html for much more tunable parameters

import matplotlib.pyplot as plt
from cycler import cycler
import seaborn as sns # for the best cmap ever
# Simply changing the fig size is often enough
plt.rcParams['figure.figsize'] = (6.4, 4.8)#(4.48, 3.34) # default (6.4, 4.8)

#plt.rc('text', usetex=True) #FIXME does not work
#plt.rcParams['text.latex.preamble'] = [ r'\usepackage{tgheros}',
#        r'\usepackage{sansmath}',
#        r'\sansmath',
#        r'\usepackage{siunitx}',
#        r'\sisetup{detect-all}']

default_cycler = (
        cycler(color=['tab:blue','tab:orange','tab:green','tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive']) +
        cycler(linestyle=['-', '--', '-.', ':', '-', '--', '-.', ':']) +
        cycler(marker=['o', 'v', '^', 's', 'p', 'P', 'X', 'D']))
plt.rc('axes', prop_cycle=default_cycler)

# Legends NOTE: shadow colour overlaps with facecolor, making the legend ugly if the former is darker. NOTE 2: one can use shadow by setting framealpha to none
plt.rcParams['legend.frameon'] = True # default True
plt.rcParams['legend.fancybox'] = False # rounded corners, default True
plt.rcParams['legend.shadow'] = False # default False
plt.rcParams['legend.framealpha'] = 0.6 # default 0.8
plt.rcParams['legend.facecolor'] = 'inherit' # default 'inherit'
plt.rcParams['legend.edgecolor'] = 'black' # default 'inherit'

# Colormap
plt.rcParams['image.cmap'] = sns.cubehelix_palette(start=1, rot=-1, hue=2.5, light=0, dark=1, gamma=.8, reverse=True, as_cmap=True) 

# Axes
plt.rcParams['axes.linewidth'] = 1.6 # default 0.8
plt.rcParams['axes.spines.left']   = True # default True
plt.rcParams['axes.spines.bottom'] = True # default True
plt.rcParams['axes.spines.top']    = True # default True
plt.rcParams['axes.spines.right']  = True # default True

# Ticks
plt.rcParams['xtick.major.size'] = 7 # default 3.5 
plt.rcParams['xtick.minor.size'] = 4 # default 2
plt.rcParams['xtick.major.width'] = 1.6 # default 0.8
plt.rcParams['xtick.minor.width'] = 1.2 # default 0.6

plt.rcParams['ytick.major.size'] = 7 # default 3.5 
plt.rcParams['ytick.minor.size'] = 4 # default 2
plt.rcParams['ytick.major.width'] = 1.6 # default 0.8
plt.rcParams['ytick.minor.width'] = 1.2 # default 0.6

# Lines
plt.rcParams['lines.linewidth'] = 2.5 # default 1.5
plt.rcParams['lines.markeredgewidth'] = 1.5 # default  1
plt.rcParams['lines.markersize'] = 9 # default  6, I often use 9
plt.rcParams['lines.markeredgecolor'] = 'k'

# Scatters
plt.rcParams['scatter.edgecolors'] = 'k' # default face

# Font
plt.rcParams['font.size'] = 14 # default 10, 18 is fine but big 
plt.rcParams['font.family'] = 'sans-serif' # default sans-serif
#plt.rcParams['font.sans-serif'] = 'Lucid' # default DejaVu Sans? none are found anyway

# Error bars
plt.rcParams['errorbar.capsize'] = 10 # default 0
elw = 2 * plt.rcParams['errorbar.capsize'] #line width for error bars

# Markers
'''
plt.rcParams[''] = # default  
plt.rcParams[''] = # default  
plt.rcParams[''] = # default  
plt.rcParams[''] = # default  
plt.rcParams[''] = # default  
plt.rcParams[''] = # default  
plt.rcParams[''] = # default  
plt.rcParams[''] = # default  
plt.rcParams[''] = # default  
plt.rcParams[''] = # default  
plt.rcParams[''] = # default  
plt.rcParams[''] = # default  
plt.rcParams[''] = # default  
plt.rcParams[''] = # default  
plt.rcParams[''] = # default  
plt.rcParams[''] = # default  
plt.rcParams[''] = # default  
plt.rcParams[''] = # default  
plt.rcParams[''] = # default  
'''

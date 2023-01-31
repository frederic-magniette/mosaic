# Generate .h5 files for the 1d challenges
# for each challenge, generate (i) a file with {input, output}, so that the sizes of the training and testing sets can be chosen by the user, (ii) a file with fixed {input_train, output_train, input_test, output_test} for reproducibility
# TODO: exhaust all the possibilities for the data generation (rand vs grid vs split vs whatever...)

import sys
import numpy as np
import h5py
import ch1d, ch2d
import challenges
<<<<<<< HEAD
from find_local import find_local_minima
=======
try: # optional module
    from find_local import find_local_minima
except:
    print("Module find_local not found")
>>>>>>> 9a111c7 (removing useless outputs)
import matplotlib.pyplot as plt # for visual checks of the generated data
import plt_params
import seaborn as sns
import mkdir_pdfs
#sns.set_theme('talk')
import seaborn as sns
#cmap = sns.cubehelix_palette(start=.5, rot=-.5, hue=2., light=0, dark=1, gamma=.8, reverse=True, as_cmap=True)
cmap = sns.cubehelix_palette(start=1, rot=-1, hue=2.5, light=0, dark=1, gamma=.8, reverse=True, as_cmap=True)
#cmap = 'inferno'

interp = 'gaussian'

bounds = [-1,1] # bounds w.r.t which the data are normalised
lvls = np.arange(bounds[0], bounds[1], 0.2)

n = 1000 # number of training points
NN = [1000,100,10] # 'exact grid' nbr of points


print(sys.argv)
if len(sys.argv)<2:
        print("usage: %s challenge_name"%(sys.argv[0]))
        sys.exit(1)
name=sys.argv[1]

is_classif = lambda funcname: True if 'ch' in funcname else False #because these need some reshaping to match my convention

func = None
# One dimensional functions
if   name=='crenel1':  func = ch1d.crenel
elif name=='expFred1': func = ch1d.expFred
elif name=='gauss1':   func = ch1d.gauss
<<<<<<< HEAD
=======
elif name=='gauss_tight1':   func = ch1d.gauss_tight
>>>>>>> 9a111c7 (removing useless outputs)
elif name=='poly_PS1': func = ch1d.poly_PS
elif name=='reLU1':    func = ch1d.reLU
elif name=='reLUreg1': func = ch1d.reLUreg
elif name=='sigm1':    func = ch1d.sigm
elif name=='sin1':     func = ch1d.sin
elif name=='sphere1':  func = ch1d.sphere
elif name=='step1':    func = ch1d.step
elif name=='tanh1':    func = ch1d.tanh
# Two dimensional functions
elif name=='crenel2':  func = ch2d.crenel
elif name=='expFred2': func = ch2d.expFred
elif name=='gauss2':   func = ch2d.gauss
elif name=='poly_PS2': func = ch2d.poly_PS
elif name=='reLU2':    func = ch2d.reLU
elif name=='reLUreg2': func = ch2d.reLUreg
elif name=='sigm2':    func = ch2d.sigm
elif name=='sin2':     func = ch2d.sin
elif name=='step2':    func = ch2d.step
elif name=='tanh2':    func = ch2d.tanh
elif name=='rastrigin2':    func = ch2d.rastrigin
elif name=='ackley2':    func = ch2d.ackley
elif name=='sphere2':    func = ch2d.sphere
elif name=='rosenbrock2':    func = ch2d.rosenbrock
elif name=='himmelblau2':    func = ch2d.himmelblau
elif 'ch' in name: func = ch2d.ch1
else:
    print("function %s not found"%name)
    sys.exit(1)


# Generate data
<<<<<<< HEAD
def gen(func, name, seed=False, gen='rand'):
=======
def gen(func, name, seed=True, gen='rand'):
>>>>>>> 9a111c7 (removing useless outputs)
    if seed == True:
         np.random.seed(123456789)
    # name is used only to determine the nbr of dimensions
    isOne = '1' in name and 'ch' not in name
    isTwo = '2' in name or 'ch' in name
    isThr = '3' in name and 'ch' not in name
    dim = 0
    if isOne: dim = 1
    if isTwo: dim = 2
    if isThr: dim = 3
    # assert the dim has been unambiguously read
    assert([isOne, isTwo, isThr].count(True) == 1)
    
    # generate input and output data
    x,y,z = np.random.rand(n), np.random.rand(n), np.random.rand(n)
    x,y,z = 2 * x - 1, 2*y-1, 2*z-1

    if gen == 'grid':
        x, y, z = np.linspace(-1,1,n), np.linspace(-1,1,n), np.linspace(-1,1,n)


    N = NN[dim-1]
    xgrid = np.linspace(-1,1, N) #for plots here and in the QML code
    ygrid = xgrid
    zgrid = xgrid
    fgrid = []

    f, fgrid = None, None
    if dim == 1:
        f = func(x)
        fgrid = func(xgrid)
    elif dim == 2:
        print(is_classif(name))
        # regression problems
        f = func(x,y) #do not eval on meshgrid here!
        import matplotlib.pyplot as plt
        #fgrid = np.zeros(shape=(len(xgrid), len(ygrid)))
        #for i in range(len(xgrid)):
        #    for j in range(len(ygrid)):
        #        fgrid[i,j] = func(xgrid[i],ygrid[j]) #!!! meshgrid messes up the evaluated funcs
        Xgrid, Ygrid = np.meshgrid(xgrid,ygrid)
        fgrid = func(Xgrid, Ygrid)

        # classification problems
        if is_classif(name):
            print('%s is a classification challenge' %name)
            f = h5py.File(name + ".h5", "r")
            data, result = f.get('input')[:],f.get('output')[:]
            f.close()
            # match the names used in the rest of this script, reshaping is done later on
            x = np.array([d[0] for d in data])
            y = np.array([d[1] for d in data])
            f = result
            #datx = np.array([d[0] for d in data])
            #daty = np.array([d[1] for d in data])
            #data = [[x,y] fpr x, y in zip(datx, daty)]

    elif dim == 3: 
        f = func(x,y,z) #do not eval on meshgrid here!
        Xgrid, Ygrid, Zgrid = np.meshgrid(xgrid, ygrid, zgrid)
        fgrid = func(Xgrid,Ygrid,Zgrid)
    
    # normalise f(x) to [-1;1]
    # NOTE: do NOT do this, it will apply different normalisation to the samples and the grid
    #f = challenges.normalise([-1,1], f, fgrid)
    #fgrid = challenges.normalise([-1,1], fgrid, fgrid)


    # plots for visual checks
    import matplotlib.pyplot as plt
    if dim == 1:
        plt.plot(xgrid, fgrid)
        plt.plot(x, f, ls='', marker='x', color='k', label='sampled')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.title('f='+name)
        plt.tight_layout()
        plt.savefig(name + '.pdf')
    if dim == 2:

        #if not(is_classif(name)):
        # find local minima, for visualisation purposes
        #gridMinima, x_mins, y_mins = find_local_minima(fgrid)
        #xMins = [xgrid[i] for i in x_mins]
        #yMins = [ygrid[i] for i in y_mins]

        # plots
        for show_samples in [True, False]:
            samples = '_with_samples' if show_samples else '_without_samples'
            fig, ax = plt.subplots()
            # CAVEAT: scatter must come before grid plot, otherwise messes up cbar
            plt.contour(fgrid, origin='lower', extent=[-1,1,-1,1], linewidths=1, linestyles='--', levels=lvls, cmap='binary', zorder=1)
            if show_samples: plt.scatter(x, y, marker='X', color='k', edgecolors='w', label='sampled')
            #plt.scatter(xMins, yMins, marker='o', color='k', s=2, edgecolors='w', label='local minima') #not 100% accurate + makes plots too charged
            plt.imshow(fgrid, origin='lower', extent=[-1,1,-1,1], interpolation=interp, cmap=cmap)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.colorbar()
            plt.title('f='+name) #TODO: not optimal, e.g. sin2 while sin(x+y) is better
            #plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig(name + samples + '.pdf')

    # tranform to lists to remove ambiguities (no ambiguities if nbr of dims is given, but this to have a well-defined structure)
    if dim == 1:
        newx = [[val] for val in x]    
        newf = [[val] for val in f]
        newxgrid = [[val] for val in xgrid]
        newfgrid = [[val] for val in fgrid]
    elif dim == 2:
        newx = [[vx, vy] for vx, vy in zip(x,y)]
        newf = [[val] for val in f]
        newxgrid = [[vx, vy] for vx, vy in zip(xgrid,ygrid)]
        newfgrid = [val for val in fgrid]

    if seed:
        name += '_seed'
    if gen == 'even':
        name += '_even'
    #challenges.save_challenge(newx, newf, name)#+'_mm')
    #challenges.save_challenge(newxgrid, newfgrid, name+'_mm_grid')
    #challenges.save_challenge(newxgrid, newfgrid, name+'_grid')

    save_challenge(newx, newf, name)#+'_mm')
    save_challenge(newxgrid, newfgrid, name+'_grid')

# stolen from Fred's challenges.py, with minor mods
def save_challenge(x,y,name):
    f=h5py.File(name+".h5","w")
    f.create_dataset('input',data=x)
    f.create_dataset('output',data=y)
    
    # Check that the data is correctly normalised. 
    # If not, raise a warning because that's not what we expect.
    is_x_normalised = (np.min(x) >= -1) and (np.max(x) <= 1)
    is_y_normalised = (np.min(y) >= -1) and (np.max(y) <= 1)
    if not (is_x_normalised):
        print('Input data for %s is not normalised' %name)
        print('Extremal values for x axis: ', np.min(x), '\t', np.max(x))
    if not (is_y_normalised):
        print('Output data for %s is not normalised' %name)
        print('Extremal values for y axis: ', np.min(y), '\t', np.max(y))

    f.create_dataset('input normalised',  data=is_x_normalised)
    f.create_dataset('output normalised', data=is_y_normalised)
    
    f.close()



gen(func, name)
mkdir_pdfs.move()

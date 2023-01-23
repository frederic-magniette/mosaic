# Generate .h5 files for the 1d challenges
# for each challenge, generate (i) a file with {input, output}, so that the sizes of the training and testing sets can be chosen by the user, (ii) a file with fixed {input_train, output_train, input_test, output_test} for reproducibility
# TODO: exhaust all the possibilities for the data generation (rand vs grid vs split vs whatever...)

import sys
import numpy as np
import h5py
import ch1d, ch2d
import challenges
import matplotlib.pyplot as plt # for visual checks of the generated data

n = 30 # number of training points
N = 100 # 'exact grid' nbr of points


print(sys.argv)
if len(sys.argv)<2:
        print("usage: %s challenge_name"%(sys.argv[0]))
        sys.exit(1)
name=sys.argv[1]

func = None
# One dimensional functions
if   name=='crenel1':  func = ch1d.crenel
#elif name=='expFred': func = ch1d.expFred
elif name=='expYann1': func = ch1d.expYann
elif name=='gauss1':   func = ch1d.gauss
elif name=='poly_PS1': func = ch1d.poly_PS
elif name=='reLU1':    func = ch1d.reLU
elif name=='reLUreg1': func = ch1d.reLUreg
elif name=='sigm1':    func = ch1d.sigm
elif name=='sin1':     func = ch1d.sin
elif name=='step1':    func = ch1d.step
elif name=='tanh1':    func = ch1d.tanh
# Two dimensional functions
elif name=='sin2': func = ch2d.sin
else:
    print("function %s not found"%name)
    sys.exit(1)

#x = np.linspace(-np.pi, np.pi, 100)
x = np.linspace(-1, 1, 100)
if name=='expFred': x = np.linspace(-4,10,100)

def gen(func, seedTrain=None, seeTest=None):
    input_train = np.random.rand(n) * (x[1]-x[0]) + x[0]
    output_train = func(input_train)

    # train = random, test = grid
    f1 = h5py.File(name + '_1.h5', 'w')
    f1.create_dataset('input_train',  data=input_train,  dtype=float)
    f1.create_dataset('input_test',   data=x,            dtype=float)
    f1.create_dataset('output_train', data=output_train, dtype=float) 
    f1.create_dataset('output_test',  data=func(x),      dtype=float)
    f1.close()

    # train = fixed rand, test = grid
    np.random.seed(123456789)
    input_train = np.random.rand(n)
    np.random.seed()
    output_train = func(input_train)
    f2 = h5py.File(name + '_2.h5', 'w')
    f2.create_dataset('input_train',  data=input_train,  dtype=float)
    f2.create_dataset('input_test',   data=x,          dtype=float)
    f2.create_dataset('output_train', data=output_train, dtype=float) 
    f2.create_dataset('output_test',  data=func(x),      dtype=float)
    f2.close()
   
    # train & test random, not split
    np.random.seed(987654321)
    xx = np.random.rand(n*10) * (x[1]-x[0]) + x[0]
    np.random.seed()
    ff = func(xx)
    f3 = h5py.File(name + '_3.h5', 'w')
    f3.create_dataset('input',  data=xx, dtype=float)
    f3.create_dataset('output', data=ff, dtype=float)
    f3.close()

   
    fread = h5py.File(name + '_3.h5', 'r')
    fread['input']

#gen(func)



def newgen(func, train_mode, test_mode):
    input_train, input_test, output_train, output_test = [], [], [], []
    input_mono, output_mono = [], []

    if train_mode == 'rand':
        input_train = np.random.rand(n) * (x[-1]-x[0]) + x[0]
    elif train_mode == 'seeded':
        np.random.seed(123456789)
        input_train = np.random.rand(n) * (x[-1]-x[0]) + x[0]
    elif train_mode == 'grid':
        input_train = np.linspace(x[0], x[-1], n)


    if test_mode == 'rand':
        output_train = np.random.rand(n) * (x[-1]-x[0]) + x[0]
    elif test_mode == 'seeded':
        np.random.seed(123456789)
        output_train = np.random.rand(n) * (x[-1]-x[0]) + x[0]
    elif train_mode == 'grid':
        output_train = np.linspace(x[0], x[-1], n)
   
    if train_mode == 'mono' and test_mode == 'mono':
        input_mono  = np.random.rand(n) * (x[-1]-x[0]) + x[0]
        output_mono = func(input_mono)
        
        f2 = h5py.File(name + '_' + modes + '.h5', 'w')
        f2.create_dataset('input',  data=input_mono,   dtype=float)
        f2.create_dataset('output', data=output_mono,   dtype=float)


    modes = train_mode[0]+test_mode[0]

    # train = random, test = grid
    f1 = h5py.File(name + '_' + modes + '.h5', 'w')
    f1.create_dataset('input_train',  data=input_train,  dtype=float)
    f1.create_dataset('input_test',   data=input_test,   dtype=float)
    f1.create_dataset('output_train', data=output_train, dtype=float) 
    f1.create_dataset('output_test',  data=output_test,      dtype=float)
    f1.close()

    
    fread = h5py.File(name + '_rr.h5', 'r')
    fread['input_train']

# Generate data where train and test are not specified
def newgen_mono(func): 
    #input_mono, output_mono = [], []

    # generate random inputs
    inputs  = np.random.rand(n) * (x[-1]-x[0]) + x[0]
    # calc corresponding outputs
    outputs = func(inputs)
    
    # convert to lists (=Fred's convention?)
    inputs  = list(inputs)
    outputs = list(outputs)

    # normalise and save
    # FIXME: change normalize -> normalise
    challenges.save_challenge(challenges.normalize_coords(inputs), challenges.normalize_coords(outputs), 'sin_mm')

    #list_input_mono = [[k] for k in input_mono]
    #list_output_mono = [[k] for k in output_mono] #old version, inconsistent with Fred's layout
    #list_output_mono = [k for k in output_mono]
    
    #f2 = h5py.File(name + '_mm' + '.h5', 'w')
    #f2.create_dataset('input',  data=list_input_mono,   dtype=float)
    #f2.create_dataset('output', data=list_output_mono,   dtype=float)

def genFred(func, name):
    # name is used only to determine the nbr of dimensions
    isOne = '1' in name
    isTwo = '2' in name
    isThr = '3' in name
    dim = 0
    if isOne: dim = 1
    if isTwo: dim = 2
    if isThr: dim = 3
    print(isOne, isTwo, isThr)
    # assert the dim has been unambiguously read
    assert([isOne, isTwo, isThr].count(True) == 1)
    # generate input and output data
    x,y,z = np.random.rand(n), np.random.rand(n), np.random.rand(n)
    x,y,z = 2 * x - 1, 2*y-1, 2*z-1
    #xyz = [x, y, z][:dim]
    
    xgrid = np.linspace(-1,1, N) #for plots
    ygrid = xgrid
    zgrid = xgrid

    f, fgrid = None, None
    if dim == 1:
        f = func(x)
        fgrid = func(xgrid)
    elif dim == 2: 
        x,y = np.meshgrid(x,y)
        xgrid, ygrid = np.meshgrid(xgrid,ygrid)
        f = func(x,y)
        fgrid = func(xgrid,ygrid)
    elif dim == 3: 
        x,y,z = np.meshgrid(x,y,z)
        xgrid, ygrid, zgrid = np.meshgrid(xgrid,ygrid, zgrid)
        f = func(x,y,z)
        fgrid = func(xgrid,ygrid,zgrid)

    #f = func(*xyz)
    #fgrid = func(*xyzgrid)
   
    if dim == 2:
        import matplotlib.pyplot as plt
        plt.imshow(fgrid)
        plt.show()

    # tranform to lists to remove ambiguities (no ambiguities if nbr of dims is given, but this to have a well-defined structure)
    newx = [[val] for val in x]    
    newf = [[val] for val in f]
    newxgrid = [[val] for val in xgrid]
    newfgrid = [[val] for val in fgrid]

    challenges.save_challenge(newx, newf, name+'_mm')
    challenges.save_challenge(newxgrid, newfgrid, name+'_mm_grid')

#for train in ['random', 'seeded', 'grid']:
#    for test in ['random', 'seeded', 'grid']:
#        newgen(func, train, test)

#newgen_mono(func)
genFred(func, name)

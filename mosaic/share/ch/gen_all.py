import os
import sys
import glob

f1d = ['crenel1', 'expFred1', 'gauss1', 'poly_PS1', 'reLU1', 'sigm1', 'sin1', 'step1', 'tanh1'] 
f2d = ['crenel2', 'expFred2', 'gauss2', 'poly_PS2', 'reLU2', 'sigm2', 'sin2', 'step2', 'tanh2', 'rastrigin2', 'ackley2', 'sphere2', 'rosenbrock2', 'himmelblau2'] 
f3d = []
funcs   = f1d + f2d + f3d

print(sys.argv)
#if len(sys.argv)<2:
#        print("usage: %s regen"%(sys.argv[0]))
#        sys.exit(1)
#regen=sys.argv[1].lower()
try: conf =sys.argv[1].lower()
except: conf=[]
#print(conf)

#if regen=='true':
valid = conf
print('All data is about to be regen. Are you sure?')
if valid == []:
    valid = input('Regenerate all data? [Y/n]')
if valid.lower() not in ['y', 'yes']:
    print('Exiting')
    sys.exit(1)

for func in funcs:
    os.system('rm -f ' + func + '*.h5')
    os.system('python3 generation.py ' + func)


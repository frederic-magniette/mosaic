import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils import data
import h5py
import os

def plot_challenge(x,y,name):
    plt.rcParams['figure.dpi']=200

    px=[]
    py=[]
    c=[]

    for i in range(len(x)):
        px.append(x[i][0])
        py.append(x[i][1])
        c.append(y[i])
                             
    plt.scatter(px,py,c=c,s=1)
    plt.title(name)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.show()
    
def save_challenge(x,y,name):
    f=h5py.File(name+".h5","w")
    f.create_dataset('input',data=x)
    f.create_dataset('output',data=y)
    f.close()  
    
def load_challenge_data(name):
    cwd = os.getcwd() # this is now the path to where MOSAIC is ran from
    path = cwd + "/mosaic/share/ch/"
    f=h5py.File(path + name +".h5","r")
    x,y=f.get('input')[:],f.get('output')[:]
    f.close()
    return x,y

def load_challenge(name,train_prop=0.9,batch_size=64, data_size=100):
    x,y=load_challenge_data(name)
    data_size=int(data_size) # parsed as a string in the config
    x = x[:data_size]
    y = y[:data_size]
    tensor_x=torch.Tensor(x)
    tensor_y=torch.Tensor(y)
    dset=torch.utils.data.TensorDataset(tensor_x,tensor_y)
    train_size=int(train_prop*len(dset))
    test_size=len(dset)-train_size
    train_dset,test_dset=data.random_split(dset,[train_size,test_size])
    
    train_loader=data.DataLoader(train_dset,batch_size=batch_size)
    test_loader=data.DataLoader(test_dset,batch_size=batch_size)
    return x,y,train_loader,test_loader

def load_challenge_exact(name): 
    # addendum by YBT, dense grid for plots
    # deliberatly mimicking load_challenge_data
    f=h5py.File(PATH_CHALLENGES + name + "_grid.h5","r")
    x,y=f.get('input')[:],f.get('output')[:]
    f.close()
    return x,y

def normalise(bounds, vals, Vals):
    '''
    Input: bounds=list sorted by dimension, e.g. [xmin, xmax, ymin, ymax]
    Output: vals normalised to within bounds
    '''
    assert(type(bounds)==list)

    bmin, bmax = bounds[0], bounds[1]
    vmin = np.min(Vals)
    vmax = np.max(Vals)
    
    vals = vals * (bmax-bmin)/(vmax-vmin) + bmin - vmin * (bmax-bmin)/(vmax-vmin)
    return vals

def train(train_loader,test_loader,model,optimizer,criterion,nb_epoch=60):
    ptr_loss=[]
    pte_loss=[]
    pepoch=[]

    print("training for %d epochs"%(nb_epoch))
    for epoch in range(nb_epoch):
        train_cum_loss=0.0
        model.train()
        for x_train,y_train in train_loader:
            optimizer.zero_grad()
            y_pred=model.forward(x_train)
            batch_loss=criterion(y_pred.squeeze(),y_train)
            train_cum_loss+=batch_loss.item()
            batch_loss.backward()
            optimizer.step()
        
        test_cum_loss=0.0
        model.eval()
        for x_test,y_test in test_loader:
            y_pred=model.forward(x_test)
            batch_loss=criterion(y_pred.squeeze(),y_test) 
            test_cum_loss+=batch_loss.item()
        
        tr_loss=train_cum_loss/len(train_loader)
        te_loss=test_cum_loss/len(test_loader)
        print("Epoch %d/%d: train loss : %f test loss: %f"%(epoch,nb_epoch,tr_loss,te_loss))
        ptr_loss.append(tr_loss)
        pte_loss.append(te_loss)
        pepoch.append(epoch)
    return pepoch,ptr_loss,pte_loss

def plot_loss(pepoch,ptr_loss,pte_loss):
    plt.plot(pepoch,ptr_loss,label="training loss")
    plt.plot(pepoch,pte_loss,label="test loss")
    plt.legend()
    plt.show()


def plot_classif(x,y,model):
    
    print("plot classification")
    
    px=[]
    py=[]
    c=[]

    for i in range(len(x)):
        px.append(x[i][0])
        py.append(x[i][1])
        c.append(y[i])

    model.eval()
    tensor_x=torch.Tensor(x)
    bad_class=0
    for i  in range(len(x)):
        y_pred=model.forward(tensor_x[i]).detach().numpy()[0] 
        if abs(y_pred-y[i])>0.4:
            px.append(x[i][0])
            py.append(x[i][1])
            c.append(2)
            bad_class+=1

    print("nb of bad classification : %d"%(bad_class))

    plt.scatter(px,py,c=c,s=1)
    plt.title("Neural network classification result")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.show()
    return px,py,c


def grid_search(x,y,model,plot_data=False):
    print("grid search")
    minx=np.min(x.T[0])
    maxx=np.max(x.T[0])
    miny=np.min(x.T[1])
    maxy=np.max(x.T[1])
    
    mpx=[]
    mpy=[]
    mc=[]
    
    #compute the model estimation for all the points on a grid
    model.eval()
    for i in np.arange(minx,maxx,(maxx-minx)/100):
        for j in np.arange(miny,maxy,(maxy-miny)/50):
            #print("i=",i," j=",j)
            xl=torch.tensor([i,j],dtype=torch.float)
            y_pred=model.forward(xl).detach().numpy()[0]
            mpx.append(i)
            mpy.append(j)
            if y_pred<0.5:
                mc.append(3)
            else:
                mc.append(4)
            
    #add the original points and bad classification from previous section
    if plot_data:
        mpx+=x.T[0].tolist()
        mpy+=x.T[1].tolist()
        mc+=y.tolist()

    plt.clf()
    plt.scatter(mpx,mpy,c=mc,s=1)
    plt.title("Grid search")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.show()


def normalize_coords(coords):
    coords=np.array(coords)
    minx=np.min(coords.T[0])
    maxx=np.max(coords.T[0])
    miny=np.min(coords.T[1])
    maxy=np.max(coords.T[1])
    normx=coords.T[0]/(maxx-minx)
    normx=normx-np.min(normx)
    normy=coords.T[1]/(maxy-miny)
    normy=normy-np.min(normy)
    normalized_coords=np.array([normx,normy]).T
    return normalized_coords

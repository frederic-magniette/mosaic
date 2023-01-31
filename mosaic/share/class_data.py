import sys
import Global # contains paths
from challenges import load_challenge
import pandas as pd
from torch.utils import data as torchData
import torch

# Class that contains the datasets and all related relevant information

class Challenges:

    def __init__(self):
        pass
  
    def prepare(self,data,local_dict,params_dict):
        self.data_name=params_dict["data_name"]
        self.batch_size=int(params_dict["batch_size"])
        self.train_prop=float(params_dict["train_prop"])
        self.data_inputs, self.data_outputs, self.train_loader, self.test_loader = load_challenge(self.data_name, train_prop=self.train_prop, batch_size=self.batch_size, data_size=params_dict["data_size"])
        self.nb_data     = len(self.data_inputs)
        self.nb_traindata = int(self.train_prop * self.nb_data)
        self.nb_testdata = self.nb_data - self.nb_traindata
        self.input_size   = len(self.data_inputs[0]) 
        return self.train_loader,self.test_loader

    def info(self):
        return { "input_size" : self.input_size, "batch_size" : self.batch_size}


class Varint:

    def __init__(self, path, nb_varint):
        self.path = path
        self.nb_varint = int(nb_varint)

    def prepare(self, data, local_dict, params_dict):
        df=pd.read_csv(self.path,sep=" ")
        #print(df)
        pertinence_list=["energ_perc_90","prop_energ_hcal","meanz","energ_perc_10","std_R","energ_perc_50","hit_perc_67","std_phi","core_shower_length","first_layer","max_layer","hit_perc_90","energ_max_layer","std_eta","shower_length","energ_tot energ_ecal","energ_hcal","prop_energ_ecal","layer_sigma_eta_max","layer_sigma_phi_max","layer_sigma_R_max","last_layer"]


        param_list=pertinence_list[0:self.nb_varint]

        varints=df.loc[:,param_list]

        labels=df["label"]

        self.train_prop=float(params_dict["train_prop"])
        self.batch_size=int(params_dict["batch_size"])

        tensor_x=torch.Tensor(varints.values)
        tensor_y=torch.Tensor(labels.values)
        dset=torch.utils.data.TensorDataset(tensor_x,tensor_y)
        train_size=int(self.train_prop*len(dset))
        test_size=len(dset)-train_size
        train_dset,test_dset=torchData.random_split(dset,[train_size,test_size])
        train_loader=torchData.DataLoader(train_dset,batch_size=self.batch_size)
        test_loader=torchData.DataLoader(test_dset,batch_size=self.batch_size)
        return train_loader,test_loader

    def infos(self):
        return { "input_size" : self.nb_varint, "batch_size" : self.batch_size}


#v=Varint()
#a,b=v.prepare("/data/DATA/data.qml/pred2/varint_4000.txt",5,{},{"train_prop" : 0.9,"batch_size" : 2})

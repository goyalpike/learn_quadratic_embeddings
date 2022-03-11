"""
Copyright (c) 2022 Pawan Goyal

All rights reserved.

This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.

@author: Pawan Goyal

"""

import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset

###################################
## TUBULAR MODEL #################
##################################
class Data_Tubular_OpInf(Dataset):
    def __init__(self, noise_level = 0.0):
        super().__init__()
        self.data = sio.loadmat('../Datasets/Tabular_reactor.mat')
        # Extract data
        self.u = np.real(self.data['Xfom']).T

        self.t = np.real(self.data['tspan']).T
        # Normalize data 
        self.t_max = self.t.max()
        self.tnor = self.t/self.t_max
        
        self.tnor = 2*(self.tnor - 0.5)
        
        # Data 
        self.coords = torch.tensor(self.tnor.reshape(-1,1)).float()

        self.u_vec = torch.tensor(self.u).float().reshape(1201,2,99)
        
        M1 = self.u_vec.min(0, keepdim=True)[0]
        self.u_min = M1.min(2, keepdim=True)[0]
        
        M2 = self.u_vec.max(0, keepdim=True)[0]
        self.u_max = M2.max(2, keepdim=True)[0]

        
        self.u_vec[...,0,:99] = (self.u_vec[...,0,:99] - self.u_min[0,0,0] )/ (self.u_max[0,0,0] - self.u_min[0,0,0])
        self.u_vec[...,1,:99] = (self.u_vec[...,1,:99] - self.u_min[0,1,0] )/ (self.u_max[0,1,0] - self.u_min[0,1,0])

        
        # self.noise = torch.tensor( np.random.randn(*self.u_vec.shape)) 
        self.u_vec += torch.randn(*self.u_vec.shape)*noise_level

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {'coords': self.coords}, {'func': self.u_vec} 
    
###################################
## 2D BURGERS EQUATION ############
##################################  
class Data_Burgers2D_OpInf(Dataset):
    def __init__(self, noise_level = 0.0):
        super().__init__()
        self.data = sio.loadmat('../Datasets/burgers2D_shock.mat')
        # Extract data
        self.u = self.data['snapmat'].T.reshape(-1,1,512,512)
        self.t = self.data['t_samples'].T

        # Normalize data 
        self.t_max = self.t.max()
        self.tnor = self.t/self.t_max
        self.tnor = 2*(self.tnor - 0.5)
        
        # Data 
        self.coords = torch.tensor(self.tnor.reshape(-1,1)).float()
        self.u_vec = torch.tensor(self.u).float()

        # self.noise = torch.tensor( np.random.randn(*self.u_vec.shape)) 
        self.u_vec += torch.randn(*self.u_vec.shape)*noise_level

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {'coords': self.coords}, {'func': self.u_vec} 
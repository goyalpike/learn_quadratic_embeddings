"""
Copyright (c) 2022 Pawan Goyal

All rights reserved.

This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.

@author: Pawan Goyal

"""

import torch
import os

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
                
    
def vectorize_mesh(grid_x, grid_t):
    coords = torch.cat((grid_x.reshape(-1,1), grid_t.reshape(-1,1)),dim= -1).float()
    return {'coords': coords}
    
    
    ## Simple RK model    
def rk4th_onestep(model,x,t=0,timestep = 1e-2, num_steps = 1):
    
    h = timestep
    k1 = model(x)
    k2 = model(x + 0.5*h*k1)
    k3 = model(x + 0.5*h*k2)
    k4 = model(x + 1.0*h*k3)
    y = x + (1/6)*(k1+2*k2+2*k3+k4)*h
        
    return y


def function_mse(model_output, gt):
    return {'func_loss': ((model_output['model_out'] - gt['func']) ** 2).mean()}

def function_l1(model_output, gt):
    return {'func_loss': ((model_output['model_out'] - gt['func']).abs()).mean()}

def simple_mse(y, y_):
    return {'func_loss': ((y - y_) ** 2).mean()}

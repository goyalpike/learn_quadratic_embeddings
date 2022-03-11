"""
Copyright (c) 2022 Pawan Goyal

All rights reserved.

This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.

@author: Pawan Goyal

"""

import torch
from torch import nn
import torch.nn.functional as F
import os
import Functions.utils as utils
from tqdm import tqdm
import time

###############################
##### QUADRATIC MODEL #########
###############################
class QuadModel(nn.Module):
    def __init__(self, dim_x, zero_init = True, print_model = True):
        super().__init__()
        self.dim_x = dim_x

        self.fc = nn.Linear(dim_x + dim_x**2 , dim_x)

        if zero_init:
            with torch.no_grad():
                self.fc.weight.zero_()
                self.fc.bias.zero_()
                
        if print_model: 
            print(self)

    def forward(self , x):
        x_kron = kron_einsum_batched_1D(x,x)
        x_total = torch.cat((x,x_kron), dim = -1)
        return self.fc(x_total)
    

##############################################################
##### PERFORMING KRONECKER PRODUCT #########
##############################################################
def kron_einsum_batched_1D(A: torch.Tensor, B: torch.Tensor): 
    """ 
    Batched Version of Kronecker Products 
    :param A: has shape (b, c, a) 
    :param B: has shape (b, c, k) 
    :return: (b, c, ak) 
    """   
    res = torch.einsum('ba,bk->bak', A, B).view(A.size(0), A.size(1)*B.size(1) ) 
#     res = torch.einsum('bca,bck->bcak', A, B).view(A.size(0), A.size(1), A.size(2)*B.size(2) ) 
    return res 


########################################################
## ENCODER AND DECODER DESIGN FOR Tubular model
########################################################

class encoder_NN_Tubular(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 16, dim_afterconv = 1024, dim_reduced = 10, only_Linear = False, kernel_size = 3, poly_order = 3, print_model = True):
        super(encoder_NN_Tubular, self).__init__()
        self.only_linear = only_Linear
        self.in_channels = in_channels
        if not only_Linear:
            self.conv1 = nn.Conv1d(in_channels = self.in_channels, out_channels = 8, kernel_size = 5, stride = 2, padding = int((kernel_size - 1)/2), padding_mode = 'reflect')
            self.conv2 = nn.Conv1d(in_channels = 8, out_channels = 16, kernel_size = 5, stride = 2, padding = int((kernel_size - 1)/2), padding_mode = 'reflect')
            self.conv3 = nn.Conv1d(in_channels = 16, out_channels = 32, kernel_size = 5, stride = 2, padding = int((kernel_size - 1)/2), padding_mode = 'reflect')
            self.conv4 = nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 5, stride = 2, padding = int((kernel_size - 1)/2), padding_mode = 'reflect')
        
        self.linear = nn.Linear(dim_afterconv, 32)
        self.linear2 = nn.Linear(32, dim_reduced)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(dim_reduced)

        if print_model:
            print(self)

    def forward(self, x):
        if not self.only_linear:
            out = F.elu(self.bn1(self.conv1(x)))
            out = F.elu(self.bn2(self.conv2(out)))
            out = F.elu(self.bn3(self.conv3(out)))
            out = F.elu(self.bn4(self.conv4(out)))
            out = out.reshape(out.shape[0],out.shape[1]*out.shape[2])
            out = self.linear(out)
            out = self.bn5(self.linear2(out))
        else:
            out = self.linear(x)
        return out
        
    
class decoder_NN_Tubular(nn.Module):
    def __init__(self, in_channels = 16, dim_afterconv = 1024, dim_reduced = 10, out_channels = 1, kernel_size = 3, num_vars = 1, only_Linear = False, print_model = True):
        super(decoder_NN_Tubular, self).__init__()
        self.only_Linear = only_Linear
        self.num_vars = num_vars
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = nn.Linear(dim_reduced, 32)
        self.linear2 = nn.Linear(32, dim_afterconv)

        if not self.only_Linear:
            self.convT1 = nn.ConvTranspose1d(in_channels = 64, out_channels = 32, kernel_size = 5, padding = 1, stride = 2, output_padding = 0)
            self.convT2 = nn.ConvTranspose1d(in_channels = 32, out_channels = 16, kernel_size = 5, padding = 1, stride = 2, output_padding = 1)
            self.convT3 = nn.ConvTranspose1d(in_channels = 16, out_channels = 8, kernel_size = 5, padding = 1, stride = 2, output_padding = 0)
            self.convT4 = nn.ConvTranspose1d(in_channels = 8, out_channels = self.out_channels, kernel_size = 5, padding = 1, stride = 2, output_padding = 0)
        
            self.bn1 = nn.BatchNorm1d(8)
            self.bn2 = nn.BatchNorm1d(16)
            self.bn3 = nn.BatchNorm1d(32)
            self.bn4 = nn.BatchNorm1d(64)
            
        if print_model:
            print(self)

    def forward(self, x):
        if self.only_Linear:
            out = self.linear(x)
            out = out.reshape(out.shape[0], self.num_vars, -1)
        else:
            out = F.elu_(self.linear(x))
            out = F.elu_(self.linear2(out))
            out = self.bn4(out.reshape(out.shape[0], 64, -1))
            out = F.elu_(self.bn3(self.convT1(out)))
            out = F.elu_(self.bn2(self.convT2(out)))
            out = F.elu_(self.bn1(self.convT3(out)))
            out = self.convT4(out)
        return out

########################################################
## ENCODER AND DECODER DESIGN FOR 2D-Burgers model
########################################################

class encoder_NN_Burgers2D(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 16, dim_afterconv = 1024, dim_reduced = 10, only_Linear = False, kernel_size = 5, poly_order = 3, print_model = True):
        super(encoder_NN_Burgers2D, self).__init__()
        self.only_linear = only_Linear
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        if not only_Linear:
            self.conv1 = nn.Conv2d(in_channels = self.in_channels, out_channels = 4, kernel_size = kernel_size, stride = 2, padding = int((kernel_size - 1)/2), padding_mode = 'reflect')
            self.conv2 = nn.Conv2d(in_channels = 4, out_channels = 8, kernel_size = kernel_size, stride = 2, padding = int((kernel_size - 1)/2), padding_mode = 'reflect')
            self.conv3 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = kernel_size, stride = 2, padding = int((kernel_size - 1)/2), padding_mode = 'reflect')
            self.conv4 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = kernel_size, stride = 2, padding = int((kernel_size - 1)/2), padding_mode = 'reflect')
            self.conv5 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = kernel_size, stride = 2, padding = int((kernel_size - 1)/2), padding_mode = 'reflect')
            self.conv6 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = kernel_size, stride = 2, padding = int((kernel_size - 1)/2), padding_mode = 'reflect')

        self.linear = nn.Linear(dim_afterconv, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, dim_reduced)

        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(dim_reduced)

        if print_model:
            print(self)

    def forward(self, x):
        if not self.only_linear:
            out = F.elu(self.bn1(self.conv1(x)))
            out = F.elu(self.bn2(self.conv2(out)))
            out = F.elu(self.bn3(self.conv3(out)))
            out = F.elu(self.bn4(self.conv4(out)))
            out = F.elu(self.bn5(self.conv5(out)))
            out = out.reshape(out.shape[0],-1)
            out = F.elu(self.linear(out))
            out = F.elu(self.linear2(out))
            out = self.bn6(self.linear3(out))
        else:
            out = self.linear(x)
        return out
        
    
class decoder_NN_Burgers2D(nn.Module):
    def __init__(self, in_channels = 16, dim_afterconv = 1024, dim_reduced = 10, out_channels = 1, kernel_size = 3, num_vars = 1, only_Linear = False, print_model = True):
        super(decoder_NN_Burgers2D, self).__init__()
        self.only_Linear = only_Linear
        self.num_vars = num_vars
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = nn.Linear(dim_reduced, 64)
        self.linear2 = nn.Linear(64, 256)
        self.linear3 = nn.Linear(256, dim_afterconv)

        if not self.only_Linear:
            self.convT1 = nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 5, padding = 2, stride = 2, output_padding = 1)
            self.convT2 = nn.ConvTranspose2d(in_channels = 32, out_channels = 16, kernel_size = 5, padding = 2, stride = 2, output_padding = 1)
            self.convT3 = nn.ConvTranspose2d(in_channels = 16, out_channels = 8, kernel_size = 5, padding = 2, stride = 2, output_padding = 1)
            self.convT4 = nn.ConvTranspose2d(in_channels = 8, out_channels = 4, kernel_size = 5, padding = 2, stride = 2, output_padding = 1)
            self.convT5 = nn.ConvTranspose2d(in_channels = 4, out_channels = self.out_channels, kernel_size = 5, padding = 2, stride = 2, output_padding = 1)
        
            self.bn1 = nn.BatchNorm2d(4)
            self.bn2 = nn.BatchNorm2d(8)
            self.bn3 = nn.BatchNorm2d(16)
            self.bn4 = nn.BatchNorm2d(32)
            self.bn5 = nn.BatchNorm2d(64)
            
        if print_model:
            print(self)

    def forward(self, x):
        if self.only_Linear:
            out = self.linear(x)
            out = out.reshape(out.shape[0], self.num_vars, -1)
        else:
            out = F.elu_(self.linear(x))
            out = F.elu_(self.linear2(out))
            out = F.elu_(self.linear3(out))
            out = self.bn5(out.reshape(out.shape[0], 64,  16, 16))
            out = F.elu_(self.bn4(self.convT1(out)))
            out = F.elu_(self.bn3(self.convT2(out)))
            out = F.elu_(self.bn2(self.convT3(out)))
            out = F.elu_(self.bn1(self.convT4(out)))
            out = self.convT5(out)
        return out

    
######################################
## TRAINING ENCODER AND DECONDER
######################################
def train_opInf_autoenc(models, train_dataloader, epochs, steps_til_summary, 
                          model_dir, loss_fn, optim, grid_info, decay_scheduler = None, epochs_decaylr = 5000, RK_params = 1.0):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tlen = grid_info['tlen']
    dt = grid_info['dt']
    
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=2000, gamma= 0.8)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):    
    
            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()
    
                model_input = {key: value.to(device) for key, value in model_input.items()}
                gt = {key: value.to(device) for key, value in gt.items()}
                
                enc_output = models['encoder'](gt['func'][0])
                dec_output = models['decoder'](enc_output)
                
                # Autoencoder loss 
                loss_solpred = loss_fn['quad_model'](dec_output, gt['func'][0])
                
                # One step forward prediction by fusing the RK4 scheme 
                u_pred = utils.rk4th_onestep(models['quad_model'],
                                             enc_output.reshape(tlen,-1)[:-1],
                                             timestep = dt)
                loss_RKforw = loss_fn['quad_model'](u_pred, enc_output.reshape(tlen,-1)[1:])
            
                # One step backward prediction by fusing the RK4 scheme 
                u_back = utils.rk4th_onestep(models['quad_model'],
                                                 enc_output.reshape(tlen,-1)[1:],
                                                 timestep = -dt)
                
                loss_RKback = loss_fn['quad_model'](u_back, enc_output.reshape(tlen,-1)[:-1])
                
                #################################
                ## Assembling losses ############
                #################################
                train_loss = 0.
                # Prediction loss
                for loss_name, loss in loss_solpred.items():
                    single_loss = loss.mean()
    
                    train_loss += single_loss

                # RKf loss
                for loss_name, loss in loss_RKforw.items():
                    single_loss = loss.mean()
    
                    train_loss += RK_params*(1/dt)*single_loss
            
                # RKb loss
                for loss_name, loss in loss_RKback.items():
                    single_loss = RK_params*(1/dt)*loss.mean()
    
                    train_loss += single_loss
                
                train_loss += 1e-5*(models['quad_model'].fc.weight.abs().sum())
                
                        
                train_losses.append(train_loss.item())
                
                if not total_steps % steps_til_summary:
                    torch.save(models['encoder'].state_dict(),
                                os.path.join(checkpoints_dir, 'model_encoder.pth'))
                    torch.save(models['decoder'].state_dict(),
                                os.path.join(checkpoints_dir, 'model_decoder.pth'))
                    torch.save(models['quad_model'].state_dict(),
                                os.path.join(checkpoints_dir, 'quadModel.pth'))

                optim.zero_grad()
                train_loss.backward()
                optim.step()
                scheduler.step()
                pbar.update(1)
    
                if not total_steps % steps_til_summary:
                    tqdm.write("Epoch %d, Total loss %0.6e, sol loss %0.6e, RKf loss %0.6e, RKb loss %0.6e, iteration time %0.6f, lr %0.3e %0.3e" % (
                        epoch, train_loss, loss_solpred['func_loss'].item(), loss_RKforw['func_loss'].item(),loss_RKback['func_loss'].item(),
                        time.time() - start_time, optim.param_groups[0]['lr'],optim.param_groups[1]['lr'])
                              )     
                total_steps += 1

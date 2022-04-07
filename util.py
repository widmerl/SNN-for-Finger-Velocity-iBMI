'''
=====================================================================
Project:      An Energy-Efficient Spiking Neural Network for Finger Velocity Decoding for Implantable Brain-Machine Interface
File:         util.py
Description:  Python code with utility functions

Date:        10. April 2022

=====================================================================

Copyright (C) 2022 ETH Zurich.

Author: Lars Widmer

SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the License); you may
not use this file except in compliance with the License.
You may obtain a copy of the License at
www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an AS IS BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and
limitations under the License.

Notice: The work in this file is partially based on 
"STBP-train-and-compression" by liukuang, which is licensed 
under the MIT License.

Please see the File "LICENCE.md" for the full licensing information.

=====================================================================
'''

from os import X_OK
from layers import LI_no_Spike, LIFSpike, tdBatchNorm2d, tdBatchNorm1d, tdBatchNorm0d
from torch.utils.data import Dataset, DataLoader
import scipy.io
import numpy as np
import torch
import torch.nn as nn
import logging
import matplotlib.pyplot as plt

#define loss function
L1loss = nn.L1Loss()
L2loss = nn.MSELoss()
def corr_loss(x,y):
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    pearson = cos(x - x.mean(dim=2, keepdim=True), y - y.mean(dim=2, keepdim=True))
    loss = torch.mean(1-pearson) #because pytorch minimizes loss and we want to maximise correlation
    return loss


#Dataset loading utility
class BMI_Dataset(Dataset):
    def __init__(self, hyperparam, set='train', fold = 'full' , single_timeseries= False):
        self.hyperparam = hyperparam
        #load dataset
        dataset = scipy.io.loadmat(hyperparam['dataset_file'])

        #Extract
        if(set == 'train' or set == 'eval'):
            X = np.array(dataset['X_train'])
            Y = np.array(dataset['Y_train'])
        else:
            X = np.array(dataset['X_val'])
            Y = np.array(dataset['Y_val'])

        totalsamplecount = X.shape[0]

        if fold != 'full':
            if set == 'eval':
                X = X[int(totalsamplecount*0.2*fold):int(totalsamplecount*0.2*(fold+1))]
                Y = Y[int(totalsamplecount*0.2*fold):int(totalsamplecount*0.2*(fold+1))]
            elif set == 'train':
                X = np.concatenate((X[:int(totalsamplecount*0.2*fold)],X[int(totalsamplecount*0.2*(fold+1)):]), axis = 0)
                Y = np.concatenate((Y[:int(totalsamplecount*0.2*fold)],Y[int(totalsamplecount*0.2*(fold+1)):]), axis = 0)

        if(single_timeseries):
            steps_per_sample = 'max'
        else:
            steps_per_sample = hyperparam['steps']

        #only use position/velocity/acceleration: 
        if hyperparam['output_type'] == 'pos':
            Y = Y[:,[0,3]]
        elif hyperparam['output_type'] == 'vel':
            Y = Y[:,[1,4]]
        elif hyperparam['output_type'] == 'acc':
            Y = Y[:,[2,5]] 
        

        #put in bins of the given length
        if(steps_per_sample == 'max'):
            steps_per_sample = Y.shape[0]
        self.X_binned = np.zeros((np.shape(X)[0]-steps_per_sample+1, np.shape(X)[1], steps_per_sample))
        self.Y_binned = np.zeros((np.shape(Y)[0]-steps_per_sample+1, np.shape(Y)[1], steps_per_sample))
        for i in range(0, np.shape(X)[0]-steps_per_sample+1):
            self.X_binned[i,:,:] = np.swapaxes(X[i:i+steps_per_sample,:], 0,1)
            self.Y_binned[i,:,:] = np.swapaxes(Y[i:i+steps_per_sample,:], 0, 1)
        

        self.X_binned = torch.from_numpy(self.X_binned).type(torch.FloatTensor)
        self.Y_binned = torch.from_numpy(self.Y_binned).type(torch.FloatTensor)

    def __len__(self):
        return (np.shape(self.X_binned)[0])

    def __getitem__(self, idx):
        return self.X_binned[idx,:,:], self.Y_binned[idx,:]

#Dataset normalisation parameter calculation function
def calc_normpar(train_set, hyperparam):
    #calculate mean, std for each electrode
    loader = DataLoader(train_set, batch_size=len(train_set), num_workers=1)
    for X,Y in loader:
        Xmean = X.mean(dim = (0,2), keepdims = True).to(hyperparam['device'])
        Xstd = X.std(dim = (0,2), keepdims = True).to(hyperparam['device'])
        Ymean = Y.mean(dim = (0,2), keepdims = True).to(hyperparam['device'])
        Ystd = Y.std(dim = (0,2), keepdims = True).to(hyperparam['device'])
    return(Xmean, Xstd, Ymean, Ystd)

#Data normalisation
def normalize(x, y, normpar):
    xmean, xstd, ymean, ystd = normpar
    x = (x - xmean)/xstd
    y = (y - ymean)/ystd
    return x, y

def denormalize(y, normpar):
    xmean, xstd, ymean, ystd = normpar
    ymean, ystd = ymean.to('cpu'), ystd.to('cpu')
    y = torch.from_numpy(y)
    y = (y * ystd.view(1,2)) + ymean.view(1,2)
    y = y.numpy()
    return y

#Metrics for eval: RMSE
def rmse(y, y_hat):
    if(y.shape[1] == 2):
        rmse1 = np.linalg.norm(y_hat[:,0] - y[:,0]) / np.sqrt(len(y[:,0]))
        rmse2 = np.linalg.norm(y_hat[:,1] - y[:,1]) / np.sqrt(len(y[:,1]))
        return (rmse1, rmse2, (rmse1+rmse2)/2)
    else:
        rmse = np.linalg.norm(y_hat - y) / np.sqrt(len(y))
        return rmse
#Metrics for eval: Correlation
def corr(y, y_hat):
    if(y.shape[1] == 2):
        corr1 = np.corrcoef(y[:,0],y_hat[:,0])[0,1]
        corr2 = np.corrcoef(y[:,1],y_hat[:,1])[0,1]
        return (corr1,corr2, (corr1+corr2)/2)
    else:
        corr = np.corrcoef(y,y_hat)[0,1]
        return corr



#plot L2, L1, correlation and rmse
def lossplot(results, trial, hyperparam, name = None):

    fig, axs = plt.subplots(figsize = (9,9))
    #fig.subplots_adjust(right=0.75)
    ax = axs
    ax.grid()
    ln1 = ax.plot(results[1,trial,:,2], linewidth=1, color = 'b', label = 'holdout eval loss L1') 
    ln2 = ax.plot(results[1,trial,:,1], linewidth = 1, color = 'b', linestyle = '--', label = 'training eval loss L1')
    ln3 = ax.plot(results[1,trial,:,0], linewidth = 1, color = 'b', linestyle = ':', label = 'training loss L1')
    ln4 = ax.plot(results[0,trial,:,2], linewidth=1, color = 'g', label = 'holdout eval loss L2')
    ln5 = ax.plot(results[0,trial,:,1], linewidth = 1, color = 'g', linestyle = '--', label = 'training eval loss L2')
    ln6 = ax.plot(results[0,trial,:,0], linewidth = 1, color = 'g', linestyle = ':', label = 'training loss L2')
    #ax.legend(loc = 'best')
    ax.set_yscale('log')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss (L2, L1)")

    ax1 = ax.twinx()
    #ax.plot(train_corr, linewidth = 1, label = 'training correlation')
    ln7 = ax1.plot(results[4,trial,:,2], linewidth=1, color = 'r', label = 'holdout eval correlation')
    ln8 = ax1.plot(results[4,trial,:,1], linewidth=1, color = 'r', linestyle = '--', label = 'training eval correlation')
    ln9 = ax1.plot(results[4,trial,:,0], linewidth=1, color = 'r', linestyle = ':', label = 'training correlation')

    #ax.legend(loc = 'best')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Correlation")

    ax2 = ax.twinx()
    ax2.spines.right.set_position(("axes", 1.2))
    ln10 = ax2.plot(results[7,trial,:,2], linewidth=1, color = 'orange', label = 'holdout eval RMSE')
    ln11 = ax2.plot(results[7,trial,:,1], linewidth=1, linestyle = '--', color = 'orange', label = 'training eval RMSE')
    ln12 = ax2.plot(results[7,trial,:,0], linewidth=1, linestyle = ':', color = 'orange', label = 'training RMSE')
    ax2.set_yscale('log')
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("RMSE")

    lns = ln1+ln2+ln3+ln4+ln5+ln6+ln7+ln8+ln9+ln10+ln11+ln12
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='best')
    if(name == None):
        plt.savefig(hyperparam['output_plots']+'training_plot_'+str(hyperparam['id'])+'.png', transparent= True, dpi=600,bbox_inches='tight' )
    else:
        plt.savefig(hyperparam['output_plots']+name+'_'+str(hyperparam['id'])+'.png', transparent= True, dpi=600,bbox_inches='tight' )
        

def training_plot(results, hyperparam, name = None):

    #mean over trials: 
    res_mean = np.mean(results, axis = 1)
    res_min = res_mean-np.std(results,axis = 1)
    res_max = res_mean+np.std(results,axis = 1)
    epochs = list(range(np.shape(results)[2]))

    fig, ax = plt.subplots(figsize = (9,9))
    ax.grid()
    #ax.plot(train_corr, linewidth = 1, label = 'training correlation')
    lns = ax.plot(res_mean[4,:,2], linewidth=1, color = 'r', label = 'eval correlation')
    ax.fill_between(epochs, res_min[4,:,2], res_max[4,:,2], color= 'r', alpha = 0.3)
    lns += ax.plot(res_mean[4,:,1], linewidth=1, color = 'r', linestyle = '--', label = 'training correlation')
    ax.fill_between(epochs, res_min[4,:,1], res_max[4,:,1], color= 'r', alpha = 0.3)
    #ln9 = ax1.plot(res_mean[4,:,0], linewidth=1, color = 'r', linestyle = ':', label = 'training correlation')
    #ax.fill_between(res_min[4,:,0], res_max[4,:,2], color= 'r')

    #ax.legend(loc = 'best')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Correlation")

    ax2 = ax.twinx()
    #ax2.spines.right.set_position(("axes", 1.2))
    lns += ax2.plot(res_mean[7,:,2], linewidth=1, color = 'orange', label = 'eval RMSE')
    ax2.fill_between(epochs, res_min[7,:,2], res_max[7,:,2], color= 'orange', alpha = 0.3)
    lns += ax2.plot(res_mean[7,:,1], linewidth=1, linestyle = '--', color = 'orange', label = 'training RMSE')
    ax2.fill_between(epochs, res_min[7,:,1], res_max[7,:,1], color= 'orange', alpha = 0.3)
    #ln12 = ax2.plot(results[7,trial,:,0], linewidth=1, linestyle = ':', color = 'orange', label = 'training RMSE')
    ax2.set_yscale('log')
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("RMSE")

    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='best')
    if(name == None):
        plt.savefig(hyperparam['output_plots']+'shaded_training_plot_'+str(hyperparam['id'])+'.png', transparent= True, dpi=600,bbox_inches='tight' )
    else:
        plt.savefig(hyperparam['output_plots']+name+'_'+str(hyperparam['id'])+'.png', transparent= True, dpi=600,bbox_inches='tight' )
        


def velplot(Y_whole, Y_hat_whole, finger, hyperparam, name=None):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30,10))
    ln1 = ax1.plot(Y_whole[:,finger], label = 'true')
    ln2 = ax1.plot(Y_hat_whole[:,finger], label = 'pred')
    ax1.set_title("whole plot")
    ax1.set_xlabel("Time (50ms bins")
    ax1.set_ylabel("Finger " + str(finger)+" velocity")
    ax1.legend()

    ln1 = ax2.plot(Y_whole[0:100,finger], label = 'true')
    ln2 = ax2.plot(Y_hat_whole[0:100,finger], label = 'pred')
    ax2.set_title("zoomed into beginning")
    ax2.set_xlabel("Time (50ms bins")
    ax2.set_ylabel("Finger " + str(finger)+" velocity")

    ln1 = ax3.plot(Y_whole[0:600,finger], label = 'true')
    ln2 = ax3.plot(Y_hat_whole[0:600,finger], label = 'pred')
    ax3.set_title("zoomed a little closer")
    ax3.set_xlabel("Time (50ms bins")
    ax3.set_ylabel("Finger " + str(finger)+" velocity")
    if(name == None):
        plt.savefig(hyperparam['output_plots']+'Velocity_plot_finger_'+str(finger)+'_'+str(hyperparam['id'])+'.png', transparent= True, dpi=600,bbox_inches='tight' )
    else:
        plt.savefig(hyperparam['output_plots']+name+'_'+str(hyperparam['id'])+'.png', transparent= True, dpi=600,bbox_inches='tight' )
        
def spikeplot(spike, layers, hyperparam):
    spikes = []
    for i in range(layers) :
        spikes.append(spike[i][:,:].detach().to('cpu').numpy()) 
        
    #plot Spike count
    logging.info('Spike counts per Inference Step and Layer:')
    logging.info('Layer 0: ' + 'Not Spiking')
    for i in range(layers):
        logging.info('Layer '+str(i+1)+': '+str(np.sum(spikes[i][:]))+' never spiked: '+str(spikes[i].shape[0]*spikes[i].shape[1] - np.count_nonzero(spikes[i])))

    fig, axs = plt.subplots(1,layers,figsize = (21,7))
    for i, ax in enumerate(axs):
        ax.hist(spikes[i][0], density=False, bins=40)  
        ax.set_ylabel('Count of Neurons')
        ax.set_xlabel('Spike Rate')
        ax.set_title('layer '+str(i+1))
    plt.savefig(hyperparam['output_plots']+'Spikerate_plot_'+str(hyperparam['id'])+'.png', transparent= True, dpi=600,bbox_inches='tight' )

    #plot spike distribution
def neuron_hist_plot(net, hyperparam):
    fig, axs = plt.subplots(1,6,figsize = (21,7))
    LIF_layer = (net.sp1, net.sp2, net.sp3)
    
    for i in range(3):
        axs[i*2].hist(LIF_layer[i].Vth.to('cpu').data.numpy(), density=False, bins=np.linspace(-0.1,1.5,31)) 
        axs[i*2].set_ylabel('Count of Neurons')
        axs[i*2].set_xlabel('Layer '+str(i)+' Vth')
        axs[i*2+1].hist(LIF_layer[i].tau.to('cpu').data.numpy(), density=False, bins=np.linspace(-0.1,1.5,31)) 
        axs[i*2+1].set_ylabel('Count of Neurons')
        axs[i*2+1].set_xlabel('Layer '+str(i)+' tau')
    plt.savefig(hyperparam['output_plots']+'Neuron_distribution_plot_'+str(hyperparam['id'])+'.png', transparent= True, dpi=600,bbox_inches='tight' )

def weight_hist_plot(net, hyperparam):
    fig, axs = plt.subplots(1,8,figsize = (28,7))
    layer = (net.fc1.layer, net.fc2.layer, net.fc3.layer, net.fc4.layer)
    for i in range(4):
        
        axs[i*2].hist(layer[i].weight.to('cpu').data.numpy().flatten(), density=False, bins=50) 
        axs[i*2].set_ylabel('Count of Weights')
        axs[i*2].set_xlabel('Layer '+str(i)+' Weights')
        axs[i*2+1].hist(layer[i].bias.to('cpu').data.numpy().flatten(), density=False, bins=50) 
        axs[i*2+1].set_ylabel('Count of Biases')
        axs[i*2+1].set_xlabel('Layer '+str(i)+' Bias')
    plt.savefig(hyperparam['output_plots']+'Weight_distribution_plot_'+str(hyperparam['id'])+'.png', transparent= True, dpi=600,bbox_inches='tight' )



#Network training function
def train(dataloader, model, hyperparam, optimizer, normpar):
    if(hyperparam['loss'] == 'L2'):
        loss_fn = L2loss
    elif(hyperparam['loss']=='L1'):
        loss_fn = L1loss
    elif(hyperparam['loss']=='corr'):
        loss_fn = corr_loss

    model.train()
    
    #variables for calculationg and saving results
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    Y_whole = np.zeros((size,2))
    Y_hat_whole = np.zeros((size,2))
    res_L2, res_L1 = 0,0
    
    #Run through Batches
    for batch, (X, Y) in enumerate(dataloader):
        X = X.to(hyperparam['device'])
        Y = Y.to(hyperparam['device'])

        #normalize inputs, expected outputs
        X,Y = normalize(X, Y, normpar)
            
        # Compute prediction error
        pred, spikeCount = model(X)

        #calculate the loss
        loss = loss_fn(pred[...,hyperparam['warmup_steps']:], Y[...,hyperparam['warmup_steps']:]) 

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #save metrics
        with torch.no_grad():
            Y_whole[batch*hyperparam['batch_size']:(batch+1)*hyperparam['batch_size']] = Y[:,:,-1].to('cpu')
            Y_hat_whole[batch*hyperparam['batch_size']:(batch+1)*hyperparam['batch_size']] = pred[:,:,-1].to('cpu')
            res_L2 += L2loss(pred[...,-1], Y[...,-1]).item()
            res_L1 += L1loss(pred[...,-1], Y[...,-1]).item()

        #print loss if needed
        if batch % hyperparam['loss_steps'] == 0:
            loss, current = loss.item(), batch * len(X)
            logging.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    #calculate metrics
    res_L2 = res_L2/num_batches
    res_L1 = res_L1/num_batches
    Y_whole = denormalize (Y_whole, normpar)
    Y_hat_whole = denormalize (Y_hat_whole, normpar)
    res_corr = corr(Y_whole,Y_hat_whole)
    res_rmse = rmse(Y_whole,Y_hat_whole)

    return (res_L2, res_L1, *res_corr, *res_rmse)




def evaluate(dataloader, model, hyperparam, normpar, plot):

    model.eval()

    #variables for calculationg and saving results
    size = len(dataloader.dataset)

    Y_whole = np.zeros((1,2))
    Y_hat_whole = np.zeros((1,2))
    res_L2, res_L1 = 0,0
    
    #Run through Batches
    for batch, (X, Y) in enumerate(dataloader):

        X = X.to(hyperparam['device'])
        Y = Y.to(hyperparam['device'])

        #normalize inputs, expected outputs
        X,Y = normalize(X, Y, normpar)

        # Compute forward pass
        pred, spikeCount = model(X)

        #save metrics
        with torch.no_grad():
            Y_whole = np.array(np.swapaxes(Y[0,:,:].to('cpu'), 0,1))
            Y_hat_whole = np.array(np.swapaxes(pred[0,:,:].to('cpu'), 0,1))

            res_L2 += L2loss(pred, Y).item()
            res_L1 += L1loss(pred, Y).item()

    #calculate metrics
    Y_whole = denormalize (Y_whole, normpar)
    Y_hat_whole = denormalize (Y_hat_whole, normpar)
    res_corr = corr(Y_whole,Y_hat_whole)
    res_rmse = rmse(Y_whole,Y_hat_whole)

    #plots
    if(plot):
        velplot(Y_whole, Y_hat_whole, 0, hyperparam)
        velplot(Y_whole, Y_hat_whole, 1, hyperparam)
        spikeplot(spikeCount, 3, hyperparam)

    return (res_L2, res_L1, *res_corr, *res_rmse)


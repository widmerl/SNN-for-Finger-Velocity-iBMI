'''
=====================================================================
Project:      An Energy-Efficient Spiking Neural Network for Finger Velocity Decoding for Implantable Brain-Machine Interface
File:         main.py
Description:  Python code describing the main loop of Training and Evaluation of the SNN

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

import yaml
from yaml.loader import SafeLoader
import logging
import torch
import numpy as np
import sys

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
'''


#todo train vth give to tdbn in network file


from util import BMI_Dataset, calc_normpar, train, evaluate, lossplot, neuron_hist_plot, weight_hist_plot, training_plot

from network import Net

#load Hyperparameters
hyperparameter_file = str(sys.argv[1])
with open(hyperparameter_file) as f:
    hyperparam = yaml.load(f, Loader=SafeLoader)

#prepare logger
logging.basicConfig(filename=hyperparam['output_report'],
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=15)
#Add an identifier for the current experiment
hyperparam['id'] = np.random.randint(100000,1000000)

#Start Log
logging.info(' ')
logging.info("=========================================================================")
logging.info("New run started with id {}".format(hyperparam['id']))
print("ID: {}".format(hyperparam['id']), end=' ')




#write Hyperparams to Output textfile
logging.info("Loading Hyperparameters : " + str(hyperparam))

#Select device
device = hyperparam['device'] if torch.cuda.is_available() else "cpu"
logging.info("Using {} device".format(device))

#load net from memory if asked
if(hyperparam['load_model']):
    net = Net(hyperparam)
    net.to(hyperparam['device'])
    net.load_state_dict(torch.load(hyperparam['load_model_dir']))
    hyperparam['epochs'] = 0
    hyperparam['5-fold'] = False
    hyperparam['trials'] = 1


#prepare results variables
                # (L2, L1, corr_finger0, corr_finger1, corr_mean, rmse_finger0, rmse_finger1, rmse_mean) (trials*folds)(epochs)(train, train_eval, train_holdout, eval)
results = np.zeros([8,hyperparam['trials']*(max(1,hyperparam['5-fold']*5)), hyperparam['epochs'], 4])

#multiple initialisations
for trial in range(hyperparam['trials']):
    print("\nTrial: {}".format(trial), end = ' ')
    for fold in range(max(1,hyperparam['5-fold']*5)):
        print("Fold: {}".format(fold), end = '')
        print("\tEpoch: ", end='')
        if(hyperparam['seed']=='random'):
            torch.manual_seed(np.random.randint(0,1000000))
        else:
            torch.manual_seed((trial+hyperparam['seed']))
        logging.info(' ')
        logging.info('Starting Trial {}, Fold {}:'.format(trial, fold))

    #prepare dataset
        logging.info("Loading Dataset")
        if(hyperparam['5-fold']):
            train_set = BMI_Dataset(hyperparam, set = 'train', fold = fold, single_timeseries= False)
            train_eval_set = BMI_Dataset(hyperparam, set = 'train', fold = fold , single_timeseries= True)
            train_holdout_set = BMI_Dataset(hyperparam, set = 'eval', fold = fold , single_timeseries= True)
            test_set = BMI_Dataset(hyperparam, set = 'test', fold = False , single_timeseries= True)
        else:
            train_set = BMI_Dataset(hyperparam, set = 'train', fold = 'full', single_timeseries= False)
            train_eval_set = BMI_Dataset(hyperparam, set = 'train', fold = 'full', single_timeseries= True)
            #train_holdout_set = BMI_Dataset(hyperparam, set = 'eval', fold = False, single_timeseries= True)
            test_set = BMI_Dataset(hyperparam, set = 'test', fold = 'full', single_timeseries= True)


        logging.info("Training set: "+str(train_set.__len__())+" samples, Validation set: "+str(train_eval_set.__len__())+" samples")


    #make network
        net = Net(hyperparam)
        net.to(hyperparam['device'])
        logging.info("Network with {} parameters initialized.".format(net.count_parameters()))

    #Do normalisation
        normpar = calc_normpar(train_set, hyperparam) #calc parameters for normalisation

    #prepare for training
        optimizer = torch.optim.AdamW(net.parameters(), lr = float(hyperparam['learning_rate']), weight_decay = hyperparam['weight_decay']) #init optimizer
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=hyperparam['batch_size'],shuffle=True, num_workers=1, pin_memory=True) #load dataset training
        #validation dataset has only a single item to begin with, but this item contains the entire timeseries
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,shuffle=False, num_workers=1, pin_memory=True) #load dataset validation
        train_eval_loader = torch.utils.data.DataLoader(train_eval_set, batch_size=1,shuffle=False, num_workers=1, pin_memory=True) #load dataset training_eval
        if hyperparam['5-fold']:
            train_holdout_loader = torch.utils.data.DataLoader(train_holdout_set, batch_size=1,shuffle=False, num_workers=1, pin_memory=True) #load dataset training_holdout

    #run for as many epochs as specified
        best_epoch = 0
        patience = hyperparam['early_stopping_patience']
        for t in range(hyperparam['epochs']): 
            logging.info("Epoch {} ".format(t))
            print("{}, ".format(t), end='')

            #decide if plots should be made
            plot = False
            if(trial == 0 and fold == 0 and t == hyperparam['epochs']-1):
                plot = True

    #save initial net to memory if asked
            if(hyperparam['save_model'] and t==0):
                torch.save(net.state_dict(), hyperparam['save_model_dir']+"model_trial_"+str(trial)+"_fold_"+str(fold)+"_epoch_"+str(t-1)+"_id_"+str(hyperparam['id'])+".pt")

    #train network, eval network
            if(not hyperparam['load_model']): #no training when loading model 
                results[:,trial*max(1,hyperparam['5-fold']*5)+fold,t,0] = train(train_loader, net, hyperparam, optimizer, normpar) #train
            results[:,trial*max(1,hyperparam['5-fold']*5)+fold,t,1] = evaluate(train_eval_loader, net, hyperparam, normpar, False) #eval training
            if hyperparam['5-fold']:
                results[:,trial*max(1,hyperparam['5-fold']*5)+fold,t,2] = evaluate(train_holdout_loader, net, hyperparam, normpar, False) #eval holdout
            results[:,trial*max(1,hyperparam['5-fold']*5)+fold,t,3] = evaluate(test_loader, net, hyperparam, normpar, plot) #eval test


            

    #log results
            logging.info("Training:   Loss L2: {:3.4}   Loss L1: {:3.4}   Corr f0: {:3.4}  Corr f1: {:3.4}  Corr: {:3.4}  RMSE f0: {:3.4}  RMSE f1: {:3.4}  RMSE: {:3.4}".format(*results[:,trial*max(1,hyperparam['5-fold']*5)+fold,t,0]))
            logging.info("Train eval: Loss L2: {:3.4}   Loss L1: {:3.4}   Corr f0: {:3.4}  Corr f1: {:3.4}  Corr: {:3.4}  RMSE f0: {:3.4}  RMSE f1: {:3.4}  RMSE: {:3.4}".format(*results[:,trial*max(1,hyperparam['5-fold']*5)+fold,t,1]))
            logging.info("Eval:       Loss L2: {:3.4}   Loss L1: {:3.4}   Corr f0: {:3.4}  Corr f1: {:3.4}  Corr: {:3.4}  RMSE f0: {:3.4}  RMSE f1: {:3.4}  RMSE: {:3.4}".format(*results[:,trial*max(1,hyperparam['5-fold']*5)+fold,t,2]))
            logging.info("Test:       Loss L2: {:3.4}   Loss L1: {:3.4}   Corr f0: {:3.4}  Corr f1: {:3.4}  Corr: {:3.4}  RMSE f0: {:3.4}  RMSE f1: {:3.4}  RMSE: {:3.4}".format(*results[:,trial*max(1,hyperparam['5-fold']*5)+fold,t,3]))

    #do plots, save them
            if(plot):
                lossplot(results, 0, hyperparam)
                neuron_hist_plot(net, hyperparam)
                weight_hist_plot(net, hyperparam)
    #save net to memory if asked
            if(hyperparam['save_model']):
                torch.save(net.state_dict(), hyperparam['save_model_dir']+"model_trial_"+str(trial)+"_fold_"+str(fold)+"_epoch_"+str(t)+"_id_"+str(hyperparam['id'])+".pt")
            
    #early stopping of training process
            if(results[4,trial*max(1,hyperparam['5-fold']*5)+fold,best_epoch,2] > results[4,trial*max(1,hyperparam['5-fold']*5)+fold,t,2] and hyperparam['early_stopping']):
                patience -= 1
                if patience == 0 :
                    if(trial*max(1,hyperparam['5-fold']*5)+fold == 0):
                        lossplot(results[:,:,:t,:], 0, hyperparam) 
                        neuron_hist_plot(net, hyperparam)
                        weight_hist_plot(net, hyperparam)
                    print(' early_stop ',end='')
                    logging.info("early stopping")
                    break  
            else:
                best_epoch = t
                patience = hyperparam['early_stopping_patience']


if(hyperparam['early_stopping']):
    best_epoch = np.argmax(np.mean(results[4,:,:,2],axis=0), axis=0)
    logging.info("Best epoch (mean): {}".format(best_epoch+1))
else:
    best_epoch = hyperparam['epochs'] -1
    logging.info("Fixed epoch (mean): {}".format(best_epoch+1))
#calculate overall results
res = np.zeros([16,4])
for i in range(8):
    res[i*2,:] = np.mean(results[i,:,best_epoch,:],axis = 0)
    res[i*2+1,:] = np.std(results[i,:,best_epoch,:],axis = 0)
#log overall results
logging.info("Mean res")
logging.info("Training:   Loss L2: {:3.4} +/- {:3.4}  Loss L1: {:3.4} +/- {:3.4}  Corr f0: {:3.4} +/- {:3.4}  Corr f1: {:3.4} +/- {:3.4}  Corr mean: {:3.4} +/- {:3.4}  RMSE f0: {:3.4} +/- {:3.4}  RMSE f1: {:3.4} +/- {:3.4}  RMSE mean: {:3.4} +/- {:3.4}".format(*res[:,0]))
logging.info("Train eval: Loss L2: {:3.4} +/- {:3.4}  Loss L1: {:3.4} +/- {:3.4}  Corr f0: {:3.4} +/- {:3.4}  Corr f1: {:3.4} +/- {:3.4}  Corr mean: {:3.4} +/- {:3.4}  RMSE f0: {:3.4} +/- {:3.4}  RMSE f1: {:3.4} +/- {:3.4}  RMSE mean: {:3.4} +/- {:3.4}".format(*res[:,1]))
logging.info("Eval:       Loss L2: {:3.4} +/- {:3.4}  Loss L1: {:3.4} +/- {:3.4}  Corr f0: {:3.4} +/- {:3.4}  Corr f1: {:3.4} +/- {:3.4}  Corr mean: {:3.4} +/- {:3.4}  RMSE f0: {:3.4} +/- {:3.4}  RMSE f1: {:3.4} +/- {:3.4}  RMSE mean: {:3.4} +/- {:3.4}".format(*res[:,2]))
logging.info("Test:       Loss L2: {:3.4} +/- {:3.4}  Loss L1: {:3.4} +/- {:3.4}  Corr f0: {:3.4} +/- {:3.4}  Corr f1: {:3.4} +/- {:3.4}  Corr mean: {:3.4} +/- {:3.4}  RMSE f0: {:3.4} +/- {:3.4}  RMSE f1: {:3.4} +/- {:3.4}  RMSE mean: {:3.4} +/- {:3.4}\n".format(*res[:,3]))


#make shaded training plot
training_plot(results, hyperparam)

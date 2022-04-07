'''
=====================================================================
Project:      An Energy-Efficient Spiking Neural Network for Finger Velocity Decoding for Implantable Brain-Machine Interface
File:         network.py
Description:  Python code describing the network architecture

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

Please see the File "LICENCE.md" for the full licensing information.

=====================================================================
'''

import torch
import torch.nn as nn
import numpy as np
from layers import SpikeAct,LIFSpike,LI_no_Spike, state_update, state_update_no_spike, tdLayer, tdBatchNorm0d, init_surrogate_gradient

class Net(nn.Module):
    def __init__(self, hyperparam):
        self.hyperparam = hyperparam
        init_surrogate_gradient(hyperparam)
        super(Net, self).__init__()
        #self.sp0 = LIFSpike([96],hyperparam)

        self.fc1 = tdLayer(nn.Linear(96, hyperparam['neuron_count'], bias = hyperparam['use_bias']), hyperparam = hyperparam)  # 5*5 from image dimension
        if(hyperparam['batchnorm'] == 'none'):
            self.dr1 = tdLayer(nn.Dropout(p=hyperparam['dropout']), hyperparam = hyperparam)
        elif(hyperparam['batchnorm'] == 'tdBN'):
            self.dr1 = tdLayer(nn.Dropout(p=hyperparam['dropout']), tdBatchNorm0d(hyperparam['neuron_count'], Vth = hyperparam['Vth']), hyperparam = hyperparam)
        self.sp1 = LIFSpike([hyperparam['neuron_count']], hyperparam = hyperparam)

        self.fc2 = tdLayer(nn.Linear(hyperparam['neuron_count'], hyperparam['neuron_count'], bias = hyperparam['use_bias']), hyperparam = hyperparam)
        if(hyperparam['batchnorm'] == 'none'):
            self.dr2 = tdLayer(nn.Dropout(p=hyperparam['dropout']), hyperparam = hyperparam)
        elif(hyperparam['batchnorm'] == 'tdBN'):
            self.dr2 = tdLayer(nn.Dropout(p=hyperparam['dropout']), tdBatchNorm0d(hyperparam['neuron_count'], Vth = hyperparam['Vth']), hyperparam = hyperparam)
        self.sp2 = LIFSpike([hyperparam['neuron_count']], hyperparam = hyperparam)

        self.fc3 = tdLayer(nn.Linear(hyperparam['neuron_count'], hyperparam['neuron_count'], bias = hyperparam['use_bias']), hyperparam = hyperparam)
        if(hyperparam['batchnorm'] == 'none'):
            self.dr3 = tdLayer(nn.Dropout(p=hyperparam['dropout']), hyperparam = hyperparam)
        elif(hyperparam['batchnorm'] == 'tdBN'):
            self.dr3 = tdLayer(nn.Dropout(p=hyperparam['dropout']), tdBatchNorm0d(hyperparam['neuron_count'], Vth = hyperparam['Vth']), hyperparam = hyperparam)
        self.sp3 = LIFSpike([hyperparam['neuron_count']], hyperparam = hyperparam)
        
        self.fc4 = tdLayer(nn.Linear(hyperparam['neuron_count'], 2, bias = hyperparam['use_bias']), hyperparam = hyperparam)
        self.nospike = LI_no_Spike([2], hyperparam = hyperparam)
        self.dequant = torch.quantization.DeQuantStub()



    def count_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

 
    def forward(self, x):
        x = self.fc1(x)
        x = self.sp1(self.dr1(x))
        spikeCount1 = torch.mean(x, dim=(-1)).to(self.hyperparam['device'])
        x = self.fc2(x)
        x = self.sp2(self.dr2(x))
        spikeCount2 = torch.mean(x, dim=(-1)).to(self.hyperparam['device'])
        x = self.fc3(x)
        x = self.sp3(self.dr3(x))
        spikeCount3 = torch.mean(x, dim=(-1)).to(self.hyperparam['device'])
        x = self.fc4(x)
        x = self.nospike(x)
        return x,(spikeCount1,spikeCount2,spikeCount3)
    
    def constrain(self, hyperparam):
        if(hyperparam['constrain_method']=='eval'):
            with torch.no_grad():
                self.sp1.Vth.data = torch.clamp(self.sp1.Vth,min=0)
                self.sp1.tau.data = torch.clamp(self.sp1.tau,min=0, max=1)

                self.sp2.Vth.data = torch.clamp(self.sp2.Vth,min=0)
                self.sp2.tau.data = torch.clamp(self.sp2.tau,min=0, max=1)

                self.sp3.Vth.data = torch.clamp(self.sp3.Vth,min=0)
                self.sp3.tau.data = torch.clamp(self.sp3.tau,min=0, max=1)

                self.nospike.tau.data = torch.clamp(self.nospike.tau,min=0, max=1)
        
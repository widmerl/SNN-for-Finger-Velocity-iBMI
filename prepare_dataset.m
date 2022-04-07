%=====================================================================
%Project:      An Energy-Efficient Spiking Neural Network for Finger Velocity Decoding for Implantable Brain-Machine Interface
%File:         prepare_dataset.m
%Description:  Matlab code to prepare the dataset
%
%Date:        10. April 2022
%
%=====================================================================
%
%Copyright (C) 2022 ETH Zurich.
%
%Author: Lars Widmer
%
%SPDX-License-Identifier: Apache-2.0
%
%Licensed under the Apache License, Version 2.0 (the License); you may
%not use this file except in compliance with the License.
%You may obtain a copy of the License at
%www.apache.org/licenses/LICENSE-2.0
%
%Unless required by applicable law or agreed to in writing, software
%distributed under the License is distributed on an AS IS BASIS, WITHOUT
%WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
%
%See the License for the specific language governing permissions and
%limitations under the License.
%
%Please see the File "LICENCE.md" for the full licensing information.
%
%=====================================================================%}

split = 0.8

%Load data
z = load('/scratch/msc21h12/datasets/MonkeyN_MC.mat');


data_len = size(z.X,1);
train_len = split*data_len;

X_train = z.X(1:train_len,1:4:384);
X_val = z.X(train_len:data_len,1:4:384);

Y_train = [z.y(1:train_len,1),z.y(1:train_len,3) ,z.y(1:train_len,3),z.y(1:train_len,2),z.y(1:train_len,4) ,z.y(1:train_len,4)];
Y_val = [z.y(train_len:data_len,1),z.y(train_len:data_len,3) ,z.y(train_len:data_len,3),z.y(train_len:data_len,2),z.y(train_len:data_len,4) ,z.y(train_len:data_len,4)];

save("/scratch/msc21h12/dataset_processed_2021.mat",'X_train','Y_train','X_val','Y_val','-v7')
Copyright (C) 2022 ETH Zurich, Switzerland. Refer to the file LICENCE.md for information on the license.

An Energy-Efficient Spiking Neural Network for Finger Velocity Decoding for Implantable Brain-Machine Interface
=============================

Install needed packages
--------------------------
I Recommend to use conda, you need the following packages installed: 

- pytorch
- torchvision
- torchaudio
- cudatoolkit
- matplotlib
- numpy
- scikit-learn
- scipy
- pyyaml


Prepare the dataset:
----------------------
If you want to use the open source dataset you can simply adapt the prepare_dataset.m file to fit your datastructure and run it. Otherwise you need to Prepare a .mat file containing the dataset for the SNN to use: It should contain 4 matrices named "X_train", "Y_train", "X_test", "Y_test". The matrices with "..train" in the name should contain the training set data, the matrices with "..test" in the name should countain the test set data. The "X.." meaning input data matrices should have the following shape: [timesteps,electrodes]. For the code to work without modification the number of electrodes has to equal 96. The "Y.." meaning expected output data matrices should have the following shape: [timesteps,(pos_0,vel_0,acc_0,pos_1,vel_1,acc_1)], with pos=position, vel=velocity, acc=acceleration of the two fingers.

Adapt hyperparameter yaml file:
----------------------------------
Most parameters are already set to good values, but you probably need to change the "dataset_file" field to the location of the prepared dataset, and choose the locations for the output report and the corresponding plots. You need to make sure that the specified path exists.
If you want to test a hyperparameter configuration you should set 5-fold to true. This means that 5-fold validation is used. To test the network 5-fold should be set to false to use the entire training set (including the holdout validation set) for training.

Train and Test
-----------------
Run the network by calling the main file with python and using the hyperparameter file as the first argument. 

    $ python main.py hyperparameters.yaml

Look at Results
---------------
After python has concluded its operation you can open the file specified at the "output_report" field to look at the results. In this file you also find the "id", which is used to identify the corresponding plots at the location specified at "output_plots"

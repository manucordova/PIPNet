#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import sys

import torch
from torch import nn
from torchsummary import summary

import importlib

import matplotlib.pyplot as plt

sys.path.insert(0, "../src/")
import data
import model
import train


# In[5]:


data_pars = dict(
                 # General parameters
                 td = 512, # Number of points
                 Fs = 10, # Sampling frequency
                 debug = False, # Print data generation details

                 # Peak parameters
                 pmin = 1, # Minimum number of Gaussians in a peak
                 pmax = 10, # Maximum number of Gaussians in a peak
                 ds = 0.03, # Spread of chemical shift values for each peak
                 lw = [1e-2, 1e-1], # Linewidth range for Gaussians
                 phase = 0., # Spread of phase

                 # Isotropic parameters
                 nmin = 1, # Minimum number of peaks
                 nmax = 10, # Maximum number of peaks
                 shift_range = [1., 9.], # Chemical shift range
                 positive = True, # Force the spectrum to be positive

                 # MAS-dependent parameters
                 mas_g_range = [1e4, 1e5], # MAS-dependent Gaussian broadening range
                 mas_l_range = [1e4, 1e5], # MAS-dependent Lorentzian broadening range
                 mas_s_range = [-1e4, 1e4], # MAS-dependent shift range
                 mas_phase = 0.1, # Random phase range for MAS spectra
                 peakwise_phase = True, # Whether the phase should be peak-wise or spectrum-wise
                 encode_imag = False, # Encode the imaginary part of the MAS spectra
                 nw = 8, # Number of MAS rates
                 mas_w_range = [30000, 100000], # MAS rate range
                 random_mas = False,
                 encode_w = False, # Encode the MAS rate of the spectra

                 # Post-processing parameters
                 noise = 0., # Noise level
                 smooth_end_len = 10, # Smooth ends of spectra
                 scale_iso = 0.8, # Scale isotropic spectra
                 offset = 0., # Baseline offset
                 norm_wr = True, # Normalize MAS rate values
                 wr_inv = False # Encode inverse of MAS rate instead of MAS rate
                )

train_pars = dict(batch_size = 16, # Dataset batch size
                  num_workers = 20, # Number of parallel processes to generate data
                  checkpoint = 1000, # Perform evaluation after that many batches
                  n_eval = 100, # Number of batches in the evaluation
                  max_checkpoints = 100, # Maximum number of checkpoints before finishing training
                  out_dir = "../data/Ensemble_PIPNet_2021_10_14_lin_noise_multiscale/", # Output directory
                  change_factor = {50: 100., 90: 10.}, # Checkpoints where
                  avg_models = False,
                  device = "cuda",
                  monitor_end = "\n"
                 )

model_pars = dict(n_models = 15,
                  input_dim = 1,
                  hidden_dim = 64,
                  kernel_size = [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3], [5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5], [7, 7, 7, 7, 7, 7], [7, 7, 7, 7, 7, 7], [7, 7, 7, 7, 7, 7], [9, 9, 9, 9, 9, 9], [9, 9, 9, 9, 9, 9], [9, 9, 9, 9, 9, 9]],
                  num_layers = 6,
                  final_kernel_size = [1, 5, 9, 1, 5, 9, 1, 5, 9, 1, 5, 9, 1, 5, 9],
                  batch_input = 4,
                  bias = True,
                  final_bias = True,
                  return_all_layers = False,
                  final_act = "linear",
                  noise = 1.e-4,
                 )

if not os.path.exists(train_pars["out_dir"]):
    os.mkdir(train_pars["out_dir"])


# In[6]:


dataset = data.PIPDataset(**data_pars)

net = model.ConvLSTMEnsemble(**model_pars).to(train_pars["device"])

opt = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# L1 loss
#loss = model.CustomLoss(exp=1., offset=1., factor=0., out_factor=0.)
# L2 loss
#loss = model.CustomLoss(exp=2., offset=1., factor=0., out_factor=0.)
# Custom loss
loss = model.CustomLoss(exp=1., offset=1., factor=1000., out_factor=0.)

sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)


# # Train the model

# In[ ]:


train.train(dataset, net, opt, loss, sch, train_pars)


# In[ ]:
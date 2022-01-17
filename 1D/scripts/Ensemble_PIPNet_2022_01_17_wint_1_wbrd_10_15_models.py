#!/usr/bin/env python
# coding: utf-8

### Implemented different broadening scales

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
                 td = 256, # Number of points
                 Fs = 12800, # Sampling frequency
                 debug = False, # Print data generation details

                 # Peak parameters
                 pmin = 1, # Minimum number of Gaussians in a peak
                 pmax = 1, # Maximum number of Gaussians in a peak
                 ds = 0.03, # Spread of chemical shift values for each peak
                 lw = [[5e1, 2e2], [1e2, 1e3]], # Linewidth range for Gaussians
                 iso_p = [0.8, 0.2],
                 iso_p_peakwise = True,
                 iso_int = [0.5, 1.], # Intensity
                 phase = 0., # Spread of phase

                 # Isotropic parameters
                 nmin = 1, # Minimum number of peaks
                 nmax = 15, # Maximum number of peaks
                 shift_range = [2000., 10000.], # Chemical shift range
                 positive = True, # Force the spectrum to be positive

                 # MAS-dependent parameters
                 mas_g_range = [[1e10, 1e11], [1e10, 5e11]], # MAS-dependent Gaussian broadening range
                 mas_l_range = [[1e7, 1e8], [1e7, 5e8]], # MAS-dependent Lorentzian broadening range
                 mas_s_range = [[-1e7, 1e7], [-1e7, 1e7]], # MAS-dependent shift range
                 mas_p = [0.8, 0.2],
                 mas_phase = 0.1, # Random phase range for MAS spectra
                 peakwise_phase = True, # Whether the phase should be peak-wise or spectrum-wise
                 encode_imag = False, # Encode the imaginary part of the MAS spectra
                 nw = 24, # Number of MAS rates
                 mas_w_range = [30000, 100000], # MAS rate range
                 random_mas = True,
                 encode_w = True, # Encode the MAS rate of the spectra

                 # Post-processing parameters
                 noise = 0., # Noise level
                 smooth_end_len = 10, # Smooth ends of spectra
                 iso_norm = 256., # Normalization factor for peaks
                 brd_norm = 64., # Normalization factor for MAS spectra
                 offset = 0., # Baseline offset
                 norm_wr = True, # Normalize MAS rate values
                 wr_inv = False # Encode inverse of MAS rate instead of MAS rate
                )

loss_pars = dict(srp_w = 1.,
                 srp_exp = 2.,
                 srp_offset = 1.,
                 srp_fac = 100.,

                 brd_w = 10.,
                 brd_sig = 3.,
                 brd_len = 15,
                 brd_exp = 2.,
                 brd_offset = 1.,
                 brd_fac = 0.,

                 int_w = 1.,
                 int_exp = 2.,

                 return_components = True,
                )

train_pars = dict(batch_size = 16, # Dataset batch size
                  num_workers = 40, # Number of parallel processes to generate data
                  checkpoint = 1000, # Perform evaluation after that many batches
                  n_eval = 100, # Number of batches in the evaluation
                  max_checkpoints = 150, # Maximum number of checkpoints before finishing training
                  out_dir = "../data/Ensemble_PIPNet_2022_01_17_wint_1_wbrd_10_15_models/", # Output directory
                  change_factor = {50: 0.}, # Checkpoints where
                  avg_models = False,
                  device = "cuda" if torch.cuda.is_available() else "cpu",
                  monitor_end = "\n"
                 )

model_pars = dict(n_models = 15,
                  input_dim = 2,
                  hidden_dim = 64,
                  kernel_size = [5, 15, 25, 35, 45, 55],
                  num_layers = 6,
                  final_kernel_size = 1,
                  batch_input = 4,
                  bias = True,
                  final_bias = True,
                  independent = True,
                  return_all_layers = True,
                  final_act = "linear",
                  noise = 2.e-4,
                 )

if not os.path.exists(train_pars["out_dir"]):
    os.mkdir(train_pars["out_dir"])


# In[6]:


dataset = data.PIPDataset(**data_pars)

net = model.ConvLSTMEnsemble(**model_pars)

class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

net = nn.MyDataParallel(net).to(train_pars["device"])

opt = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

loss = model.CustomLoss(**loss_pars, device=train_pars["device"])

sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)


# # Train the model

# In[ ]:


train.train(dataset, net, opt, loss, sch, train_pars)


# In[ ]:

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

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

sys.path.insert(0, "../src/")
import data
import model

torch.set_num_threads(8)


# In[2]:


data_pars = dict(nmin = 1, # Minimum number of peaks
                 nmax = 10, # Maximum number of peaks
                 shift_range = [1., 9.], # Chemical shift range
                 lw1_range = [0., 0.005], # Isotropic broadening range
                 gen_lw2 = True,
                 lw2_range = [0.1, 0.5], # Second isotropic broadening range
                 asym = False, # Generate asymmetric shifts
                 mas_g_range = [0., 1e5], # MAS-dependent Gaussian broadening range
                 mas_l_range = [0., 5e4], # MAS-dependent Lorentzian broadening range
                 mas_s_range = [-1e4, 1e4], # MAS-dependent shift range
                 nw = 9, # Number of MAS rates
                 mas_w_range = [30000, 100000], # MAS rate range
                 random_mas = False,
                 encode_w = False,
                 td = 512,
                 Fs = 10,
                 noise = 0.,
                 smooth_end_len = 10,
                 scale_iso = 0.8,
                 offset = 0.,
                 norm_wr = True,
                 wr_inv = False
                )

load_pars = dict(batch_size = 64,
                 num_workers = 8
                )

model_pars = dict(input_dim = 1,
                  hidden_dim = 64,
                  kernel_size = [1, 3, 5],
                  num_layers = 3,
                  final_kernel_size = 1,
                  batch_input = 3,
                  bias = True,
                  final_bias = True,
                  return_all_layers = False,
                 )


# In[3]:


dataset = data.PIPDataset(**data_pars)

data_loader = torch.utils.data.DataLoader(dataset,
                                          **load_pars)


# In[4]:

print("Generating training data...")


n_samples = 16
X = []
y = []

for i in range(n_samples):
    xi, _, yi = dataset.__getitem__(i)
    X.append(torch.unsqueeze(xi, 0))
    y.append(torch.unsqueeze(yi, 0))

X = torch.cat(X)
y = torch.cat(y)


# In[5]:


net = model.ConvLSTM(**model_pars)


L2 = torch.nn.MSELoss()
L1 = torch.nn.L1Loss()
BCE = torch.nn.BCELoss()


# In[6]:


L = model.CustomLoss(exp=1., offset=0.1, factor=10., out_factor=0.)


# In[7]:


opt = torch.optim.Adam(net.parameters(), lr=0.001)
sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)


# In[8]:


def live_plot(y_out, y_trg, step, n_steps, loss, lr, figsize=(16,6), n_row=1, n_col=2, n_show=2, fig=None, axs=None):
    
    if fig is None:
        fig = plt.figure(figsize=figsize)
        axs = []
        
        for i in range(n_row):
            for j in range(n_col):
                axs.append(fig.add_subplot(n_row, n_col, (i * n_col) + j + 1))
        
        for i in range(n_show):
            axs[i].plot(y_trg[i])
            axs[i].plot(y_out[i])
        
    else:
        for i in range(n_show):
            axs[i].lines[1].set_ydata(y_out[i])
        
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    return fig, axs


# In[ ]:


print("Training...")

n_train = 1000

net.train()

ymax = torch.max(y)

plt.ion()
fig = None
axs = None

for i in range(n_train):

    print("Step {}/{}".format(i+1, n_train))
    
    net.zero_grad()
    
    output, _, _ = net(X)
    
    loss = L(output, y)
    
    loss.backward()
    
    opt.step()
    
    sch.step(loss)
    
    lr = opt.param_groups[0]["lr"]
    fig, axs = live_plot(output.detach(), y, i, n_train, float(loss.detach()), lr, figsize=(12,6), n_row=4, n_col=4, n_show=16, fig=fig, axs=axs)


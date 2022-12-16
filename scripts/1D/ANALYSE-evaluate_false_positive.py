#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import sys
import copy

import torch
torch.set_num_threads(os.cpu_count())
from torch import nn

import json

import matplotlib as mpl
import matplotlib.pyplot as plt

from pipnet import data
from pipnet import model
from pipnet import utils

device = "cuda" if torch.cuda.is_available() else "cpu"


# In[2]:


mod = sys.argv[1]

in_dir = f"../../trained_models/{mod}/"
fig_dir = f"../../figures/1D/{mod}/"

batch_size = 16
n_batch = 64

eval_all_steps = False

noise_levels = [0., 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]

signal_thresh = 1e-2

signal_extend = 10

iso_pars = dict(
    td = 512,
    Fs = 12_800,
    nmin = 1,
    nmax = 15,
    freq_range = [2_000., 10_000.],
    gmin = 1,
    gmax = 1,
    spread = 5.,
    lw_range = [[5e1, 2e2], [1e2, 5e2], [1e2, 1e3]],
    lw_probs = [0.7, 0.2, 0.1],
    int_range = [0.5, 1.], # Intensity
    phase = 0.,
    debug = False,
)

mas_pars = dict(
    nw = 24,
    mas_w_range = [30_000., 100_000.],
    random_mas = True,
    mas_phase_p = 0.5,
    mas_phase_scale = 0.05,
    
    # First-order MAS-dependent parameters
    mas1_lw_range = [[1e7, 5e7], [5e7, 1e8]],
    mas1_lw_probs = [0.8, 0.2],
    mas1_m_range = [[0., 0.], [0., 1e4], [1e4, 5e4]],
    mas1_m_probs = [0.1, 0.1, 0.8],
    mas1_s_range = [[-1e7, 1e7]],
    mas1_s_probs = [1.],

    # Second-order MAS-dependent parameters
    mas2_prob = 1.,
    mas2_lw_range = [[0., 0.], [1e11, 5e11]],
    mas2_lw_probs = [0.5, 0.5],
    mas2_m_range = [[0., 0.], [1e8, 5e8]],
    mas2_m_probs = [0.8, 0.2],
    mas2_s_range = [[0., 0.], [-2e10, 2e10]],
    mas2_s_probs = [0.8, 0.2],
    
    # Other MAS-dependent parameters
    non_mas_p = 0.5,
    non_mas_m_trends = ["constant", "increase", "decrease"],
    non_mas_m_probs = [0.34, 0.33, 0.33],
    non_mas_m_range = [0., 1.],
    
    int_decrease_p = 0.1,
    int_decrease_scale =[0.3, 0.7],
    debug = False,
)

with open(f"{in_dir}data_pars.json", "r") as F:
    data_pars = json.load(F)

data_pars["iso_pars"] = iso_pars
data_pars["mas_pars"] = mas_pars
data_pars["noise"] = 0.
data_pars["mas_l_noise"] = 0.05
data_pars["mas_s_noise"] = 25.
data_pars["gen_mas_shifts"] = False

def get_false_positive_rate(ys_pred, ys_trg, thresh, extend, fdir=None, nshow=0):
    
    all_pos = 0
    all_fps = 0
    
    for k, (y_pred, y_trg) in enumerate(zip(ys_pred, ys_trg)):
        
        # Normalize spectra
        y_pred /= np.max(y_pred)
        y_trg /= np.max(y_trg)
        
        # Get signal rate
        all_pos += np.sum(y_pred > thresh)
        
        # Select spectral regions without signal
        mask0 = y_trg < thresh
        
        mask = mask0.copy()
        
        # Extend spectral regions with signal to avoid false false positive
        for i in range(extend, mask0.shape[0]-extend):
            if not mask0[i]:
                mask[i-extend:i+extend] = False
        
        all_fps += np.sum(y_pred[mask] > thresh)
        
        this_fp = np.logical_and(mask, y_pred > thresh)
        
        if fdir is not None and k < nshow:
            fig = plt.figure(figsize=(4,3))
            ax = fig.add_subplot(1,1,1)
            ax.plot(y_trg, "k")
            ax.plot(y_pred, "r")
            ax.fill_between(np.arange(mask.shape[0]), mask, alpha=0.2, color="C0", zorder=-2, linewidth=0.)
            ax.fill_between(np.arange(mask.shape[0]), this_fp, alpha=0.2, color="r", zorder=-1, linewidth=0.)
            fig.tight_layout()
            fig.savefig(f"{fdir}sample_{k+1}.pdf")
            plt.close()
    
    return all_fps / all_pos


# In[3]:


if not os.path.exists(in_dir):
    raise ValueError(f"Unknown model: {mod}")
    
if not os.path.exists(fig_dir):
    os.mkdir(fig_dir)

fdir = fig_dir + "eval_false_positive/"

if not os.path.exists(fdir):
    os.mkdir(fdir)


# In[6]:


with open(in_dir + "model_pars.json", "r") as F:
    model_pars = json.load(F)
model_pars["noise"] = 0.
model_pars["return_all_layers"] = eval_all_steps

net = model.ConvLSTMEnsemble(**model_pars).to(device)
net.load_state_dict(torch.load(in_dir + f"network", map_location=device))
net = net.eval()

fp_rates = []

for noise in noise_levels:

    np.random.seed(1)

    print(f"Generating noise level {noise}")

    # Update data parameters
    data_pars2 = copy.deepcopy(data_pars)
    data_pars2["noise"] = noise
    dataset = data.Dataset(**data_pars2)

    X = []
    y = []
    y_pred = []
    y_std = []
    ys_pred = []
    for ibatch in range(n_batch):
        # Generate dataset
        Xi, yi = dataset.generate_batch(size=batch_size)
        Xi = Xi.to(device)
        yi = yi.to(device)

        print(f"  Batch {ibatch + 1}/{n_batch}")

        # Make predictions
        with torch.no_grad():
            yi_pred, yi_std, yis_pred = net(Xi)

        if net.return_all_layers:
            if net.ndim == 1:
                yi = yi.repeat((1, yi_pred.shape[1], 1))
            elif net.ndim == 2:
                yi = yi.repeat((1, yi_pred.shape[1], 1, 1))

        X.append(Xi)
        y.append(yi[:, 0])
        y_pred.append(yi_pred[:, 0])

    X = torch.cat(X).cpu()
    y = torch.cat(y).cpu()
    y_pred = torch.cat(y_pred).cpu()
    
    fdir2 = fdir + f"noise_level_{noise}/"
    if not os.path.exists(fdir2):
        os.mkdir(fdir2)
    
    fp_rate = get_false_positive_rate(y_pred.numpy(), y.numpy(), signal_thresh, signal_extend, fdir=fdir2, nshow=64)
    
    fp_rates.append(fp_rate*100)
    
    print(fp_rates)

fig = plt.figure(figsize=(4,3))
ax = fig.add_subplot(1,1,1)
ax.semilogx(noise_levels, fp_rates)
ax.set_xlabel("Noise level")
ax.set_ylabel("False positive signal rate [%]")
fig.tight_layout()
fig.savefig(f"{fdir}fp_rates.pdf")
plt.close()

pp = ""
for n, f in zip(noise_levels, fp_rates):
    pp += f"{n: 6.4f} -> {f: 6.4f}%\n"

with open(f"{fdir}fp_rates.txt", "w") as F:
    F.write(pp)
import numpy as np
import os
import sys
import copy

import torch
from torch import nn
import scipy as sp

import importlib
import pickle as pk

sys.path.insert(0, "../src/")
import data
import model
import train

if __name__ == "__main__":

    data_pars = dict(
        # General parameters
        td=256,  # Number of points
        Fs=12800,  # Sampling frequency
        debug=False,  #  Print data generation details
        # Peak parameters
        pmin=1,  # Minimum number of Gaussians in a peak
        pmax=1,  # Maximum number of Gaussians in a peak
        ds=0.03,  # Spread of chemical shift values for each peak
        lw=[[5e1, 2e2], [1e2, 1e3]],  # Linewidth range for Gaussians
        iso_p=[0.8, 0.2],
        iso_p_peakwise=True,
        iso_int=[0.5, 1.0],  # Intensity
        phase=0.0,  # Spread of phase
        # Isotropic parameters
        nmin=1,  # Minimum number of peaks
        nmax=15,  # Maximum number of peaks
        shift_range=[2000.0, 10000.0],  # Chemical shift range
        positive=True,  # Force the spectrum to be positive
        # MAS-independent parameters
        # MAS-independent GLS broadening range
        mas0_lw_range=[0.0, 0.0],
        mas0_lw_p=[1.0],
        # MAS-independent GLS mixing paramter
        mas0_m_range=[[0.0, 0.1], [0.0, 1.0]],
        mas0_m_p=[0.8, 0.2],
        mas0_s_range=[0.0, 0.0],  # MAS-independent shift range
        mas0_s_p=[1.0],
        # MAS-dependent parameters
        # MAS-dependent GLS broadening range
        mas_lw_range=[[1e7, 1e8], [5e7, 2e8]],
        mas_lw_p=[0.8, 0.2],
        mas_m_range=[[1e4, 5e4]],  # MAS-dependent GLS mixing paramter
        mas_m_p=[1.0],
        mas_s_range=[-1.5e7, 1.5e7],  # MAS-dependent shift range
        mas_s_p=[1.0],
        # Second-order MAS-dependent parameters
        mas_w2=True,
        mas_w2_p=1.0,
        # Second-order MAS-dependent Gaussian broadening range
        mas2_lw_range=[[1e2, 1e4], [1e8, 1e9]],
        mas2_lw_p=[0.5, 0.5],
        # Second-order MAS-dependent Lorentzian broadening range
        mas2_m_range=[[0.0, 10], [1e4, 1e6]],
        mas2_m_p=[0.5, 0.5],
        # Second-order MAS-dependent shift range
        mas2_s_range=[[0.0, 0.0], [-2e11, 2e11]],
        mas2_s_p=[0.8, 0.2],
        # Other MAS-dependent mixing
        mas_other_mixing_p=0.5,
        mas_other_mixing_trends=["constant", "increase", "decrease"],
        mas_other_mixing_probs=[0.4, 0.3, 0.3],
        mas_other_mixing_range=[0.0, 1.0],
        # Add noise to linewidths
        mas_lw_noise=0.05,
        mas_s_noise=20.0,
        mas_phase=0.01,  # Random phase range for MAS spectra
        peakwise_phase=True,  # Whether the phase should be peak-wise or spectrum-wise
        encode_imag=False,  # Encode the imaginary part of the MAS spectra
        nw=8,  # Number of MAS rates
        mas_w_range=[30000, 100000],  # MAS rate range
        random_mas=True,
        encode_w=True,  # Encode the MAS rate of the spectra
        # Post-processing parameters
        noise=0.0,  # Noise level
        smooth_end_len=10,  # Smooth ends of spectra
        iso_norm=256.0,  #  Normalization factor for peaks
        brd_norm=64.0,  #  Normalization factor for MAS spectra
        offset=0.0,  # Baseline offset
        norm_wr=True,  # Normalize MAS rate values
        wr_inv=False,  # Encode inverse of MAS rate instead of MAS rate
    )

    loss_pars = dict(
        trg_fuzz=5.0,
        trg_fuzz_len=25,
        srp_w=1.0,
        srp_exp=2.0,
        srp_offset=1.0,
        srp_fac=100.,
        brd_w=1e-4,
        brd_sig=5.0,
        brd_len=25,
        brd_exp=2.0,
        brd_offset=1.0,
        brd_fac=0.,
        int_w=0.1,
        int_exp=2.0,
        int_edge=0,
        return_components=True,
    )

    train_pars = dict(
        batch_size=4,  # Dataset batch size
        num_workers=4,  # Number of parallel processes to generate data
        checkpoint=1000,  # Perform evaluation after that many batches
        n_eval=200,  # Number of batches in the evaluation
        max_checkpoints=200,  # Maximum number of checkpoints before finishing training
        out_dir="../data/PIPNet2D_2022_06_13_lin/",  # Output directory
        change_loss={25: {"trg_fuzz": 3.0, "trg_fuzz_len": 15},
                       50: {"trg_fuzz": 0.0, "srp_fac": 0.0}},  # Checkpoints where loss parameter are updated
        avg_models=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
        monitor_end="\n",
    )

    model_pars = dict(
        n_models=1,
        input_dim=2,
        hidden_dim=16,
        kernel_size=[5, 15],
        num_layers=2,
        final_kernel_size=3,
        batch_input=3,
        bias=True,
        final_bias=False,
        independent=True,
        return_all_layers=True,
        final_act="linear",
        noise=2.0e-3,
    )

    dataset = data.PIPDatasetGLS2D(data_pars, data_pars, rot_range=[-45.])

    net = model.ConvLSTMEnsemble(**model_pars).to(train_pars["device"])

    opt = torch.optim.Adam(
        net.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False,
    )

    if not os.path.exists(train_pars["out_dir"]):
        os.mkdir(train_pars["out_dir"])

    loss = model.CustomLoss(**loss_pars, device=train_pars["device"])

    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)

    train.train(dataset, net, opt, loss, sch, train_pars)
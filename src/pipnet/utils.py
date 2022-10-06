###########################################################################
###                               PIPNet                                ###
###                          Utility functions                          ###
###                        Author: Manuel Cordova                       ###
###                       Last edited: 2022-09-30                       ###
###########################################################################

import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import scipy as sp



def load_1d_topspin_spectrum(path):
    """
    Load a 1D spectrum from Topspin and retrieve the MAS rate
    The MAS rate is first looked for in the title of the experiment,
    and if it is not found there the MASR variable in the acquisition
    parameters will be searched.

    Input:      - path  Path to the Topspin directory

    Outputs:    - dr    Real part of the spectrum
                - di    Imaginary part of the spectrum
                - wr    MAS rate (-1 if the rate is not found)
                - ppm   Array of chemical shift values in the spectrum
                - hz    Array of frequency values in the spectrum
    """
    
    if not path.endswith("/"):
        path += "/"

    pd = f"{path}pdata/1/"
    fr = pd + "1r"
    fi = pd + "1i"
    ti = pd + "title"

    # Load spectrum
    with open(fr, "rb") as F:
        dr = np.fromfile(F, np.int32).astype(float)
    with open(fi, "rb") as F:
        di = np.fromfile(F, np.int32).astype(float)
        
    wr = -1.
    # Get MAS rate from title
    wr_found = False
    with open(ti, "r") as F:
        lines = F.read().split("\n")
    for l in lines:
        if "KHZ" in l.upper():
            wr = float(l.upper().split("KHZ")[0].split()[-1]) * 1000
            wr_found = True
        elif "HZ" in l.upper():
            wr = float(l.upper().split("HZ")[0].split()[-1])
            wr_found = True

    # Parse acquisition parameters
    with open(f"{path}acqus", "r") as F:
        lines = F.read().split("\n")
    for l in lines:
        if l.startswith("##$MASR") and not wr_found:
            try:
                wr = int(l.split("=")[1].strip())
            except:
                pass
        if l.startswith("##$TD="):
            TD = int(l.split("=")[1].strip())
        if l.startswith("##$SW_h="):
            SW = float(l.split("=")[1].strip())

    # Parse processing parameters
    with open(f"{pd}procs", "r") as F:
        lines = F.read().split("\n")

    for l in lines:
        if l.startswith("##$SI="):
            n_pts = int(l.split("=")[1].strip())
        if l.startswith("##$OFFSET="):
            offset = float(l.split("=")[1].strip())
        if l.startswith("##$SF="):
            SF = float(l.split("=")[1].strip())
            
    # Generate chemical shift and frequency arrays
    AQ = TD / (2 * SW)
    hz = offset * SF - np.arange(n_pts) / (2 * AQ * n_pts / TD)
    ppm = hz / SF

    return dr, di, wr, ppm, hz



def extract_1d_dataset(path, exp_init, exp_final, exps=None):
    """
    Extract a vmas dataset of 1D spectra

    Inputs:     - path              Path to the vmas dataset
                - exp_init          Index of the first experiment (inclusive)
                - exp_final         Index of the last experiment (inclusive)
                - exps              Custom indices of all experiments

    Outputs:    - ppm               Array of chemical shifts
                - hz                Array of frequencies
                - sorted_ws         Array of MAS rates
                - sorted_X_real     Array of real parts of the spectra
                - sorted_X_imag     Array of imaginary parts of the spectra
    """
    
    if not path.endswith("/"):
        path += "/"
    
    ws = []
    X_real = []
    X_imag = []

    if exps is None:
        exps = np.arange(exp_init, exp_final+1)
    
    for d in os.listdir(path):
        if d.isnumeric() and int(d) in exps:
            Xr, Xi, wr, ppm, hz = load_1d_topspin_spectrum(f"{path}{d}/")
            X_real.append(Xr)
            X_imag.append(Xi)
            ws.append(wr)
    
    sorted_inds = np.argsort(ws)
    
    sorted_ws = np.array([ws[i] for i in sorted_inds])
    
    sorted_X_real = np.array([X_real[i] for i in sorted_inds])
    sorted_X_imag = np.array([X_imag[i] for i in sorted_inds])
    
    return ppm, hz, sorted_ws, sorted_X_real, sorted_X_imag



def prepare_1d_input(xr, ws, data_pars, xi=None, xmax=0.5):

    X = torch.Tensor(xr).unsqueeze(0).unsqueeze(2)

    Xint = torch.sum(X, dim=3)

    if data_pars["encode_imag"]:
        X = torch.cat([X, torch.Tensor(xi).unsqueeze(0).unsqueeze(2)], dim=2)
    
    # Normalize integrals
    X /= Xint[:, :, :, None]
    X /= torch.max(X[:, :, 0]) / xmax


    if data_pars["encode_wr"]:
        W = torch.tensor(ws) / data_pars["wr_norm_factor"]
        if data_pars["wr_inv"]:
            W = 1. / W
        W = W.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        X = torch.cat([X, W.repeat(1, 1, 1, X.shape[-1])], axis=2)

    return X.type(torch.float32)



def plot_1d_iso_prediction(
    X,
    y_pred,
    y_std,
    y_trg=None,
    y_trg_std=None,
    pred_scale=1.,
    trg_scale=1.,
    X_offset=0.,
    pred_offset=0.,
    trg_offset=0.,
    xvals=None,
    x_trg=None,
    wr_factor=1.,
    xinv=False,
    c0=np.array([0., 1., 1.]),
    dc=np.array([0., -1., 0.]),
    ylim=None,
    all_steps=False,
    show=True,
    save=None,
):

    if xvals is None:
        xvals = np.arange(X.shape[-1])
    if x_trg is None:
        x_trg = xvals
    
    if all_steps:
        nsteps = y_pred.shape[0]
    else:
        nsteps = 1
    # Plot all prediction steps
    for step in range(nsteps):
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(1,1,1)

        wr = X[-(step+1), 1, 0] * wr_factor / 1000.
        labels = [f"{wr:.0f} kHz MAS", "PIPNet"]
        hs = []

        h = ax.plot(xvals, X[-(step+1), 0] + X_offset, linewidth=1., zorder=0)
        hs.append(h[0])
        
        h = ax.plot(xvals, y_pred[-(step+1)] * pred_scale + pred_offset, "r", linewidth=1., zorder=-1)
        hs.append(h[0])

        ax.fill_between(
            xvals,
            (y_pred[-(step+1)] - y_std[-(step+1)]) * pred_scale + pred_offset,
            (y_pred[-(step+1)] + y_std[-(step+1)]) * pred_scale + pred_offset,
            color="r",
            alpha=0.3,
            linewidth=0.,
        )

        if y_trg is not None:
            labels.append("Ground-truth")
            h = ax.plot(x_trg, y_trg * trg_scale + trg_offset, "k", linewidth=1., zorder=-2)
            hs.append(h[0])

            if y_trg_std is not None:
                ax.fill_between(
                    x_trg,
                    (y_trg - y_trg_std) * trg_scale + trg_offset,
                    (y_trg + y_trg_std) * trg_scale + trg_offset,
                    color="k",
                    alpha=0.3,
                    linewidth=0.,
                )
    
        if xinv:
            ax.invert_xaxis()
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.legend(hs, labels)
        fig.tight_layout()
        if show:
            plt.show()
        if save is not None:
            tmp = save.split(".")
            if step > 0:
                tmp[-2] += f"_step_{nsteps-step}"
            fig.savefig(".".join(tmp))
        plt.close()

    return



def extract_1d_linewidth(x, y, x_range, verbose=False):
    """
    Extract the linewidth (FWHM) of a peak in the given x-axis range

    Inputs:     - x         x-axis values
                - y         y-axis values
                - x_range   Range of x-values containing the peak

    Outputs:    - lw        Extracted linewidth (FWHM)
                - x0        Extracted peak position
    """

    xmin = np.min(x_range)
    xmax = np.max(x_range)

    # Select range and identify top of the peak
    inds = np.where(np.logical_and(x >= xmin, x <= xmax))[0]
    dx = np.mean(x[1:] - x[:-1])
    top = np.max(y[inds])
    x0 = x[inds[np.argmax(y[inds])]]
    
    xr = None
    xl = None
    
    # Search for crossing of the half maximum
    for i, j in zip(inds[:-1], inds[1:]):
        # Crossing on the right
        if y[i] > top / 2 and y[j] < top / 2:
            dy = y[j] - y[i]
            dy2 = (top / 2) - y[i]
            xr = x[i] + dx * dy2 / dy
            
        # Crossing on the left
        if y[i] < top / 2 and y[j] > top / 2:
            dy = y[j] - y[i]
            dy2 = (top / 2) - y[i]
            xl = x[i] + dx * dy2 / dy
    
    if xl is None:
        if verbose:
            print("Warning: no left boundary found!")
        xl = x[inds[0]]
    if xr is None:
        if verbose:
            print("Warning: no right boundary found!")
        xr = x[inds[-1]]

    lw = abs(xl - xr)

    return lw, x0



def extract_1d_linewidths(x, y, x_ranges):

    lws = np.zeros(len(x_ranges))
    pks = np.zeros(len(x_ranges))

    for i, x_range in enumerate(x_ranges):
        lws[i], pks[i] = extract_1d_linewidth(x, y, x_range)

    return lws, pks



def get_relative_1d_integrals(x, y, regions):
    """
    Compare the relative integrals in different regions of two spectra

    Inputs:     - x             x-axis values
                - y             y-axis values
                - regions       Regions to compute the integrals for

    Outputs:    - integrals     List of integrals for each region
    """

    integrals = np.zeros(len(regions))

    for i, (r1, r2) in enumerate(regions):
        i1 = np.argmin(np.abs(x - r1))
        i2 = np.argmin(np.abs(x - r2))

        il = min(i1, i2)
        ir = max(i1, i2)

        integrals[i] = np.sum(y[il:ir])
        
    integrals /= np.sum(integrals)

    return integrals



def extract_1d_pip(in_dir, compound, parts, res):
    """
    Extract 1d PIP prediction
    """
    
    if compound == "mdma":
        c = compound
    elif compound == "molnupiravir":
        c = "molnu"
    else:
        c = compound[:3]
    
    ys_part_means = []
    ys_part_stds = []
    ys_ppms = []
    
    if len(parts) == 0 or len(res) == 0:
        return [], [], []
    
    for p, n in zip(parts, res):
        
        
        d = f"{in_dir}{compound}_{n}/"
        
        if not os.path.exists(d):
            return [], [], []
    
        ys_part = []

        i_guess = 1
        while os.path.exists(f"{d}{c}_{p}_guess_r{i_guess}.mat"):

            m = sp.io.loadmat(f"{d}{c}_{p}_guess_r{i_guess}.mat")

            ys_part.append(m["x"][:-3])
            ppm = m["ppm"][0, m["range"][0]]
            
            i_guess += 1
        
        if len(ys_ppms) > 0:
            already_ppms = np.concatenate(ys_ppms, axis=0)
            inds = np.where(np.logical_or(ppm < np.min(already_ppms), ppm > np.max(already_ppms)))[0]
        else:
            inds = range(len(ppm))
        
        ys_ppms.append(ppm[inds])
        ys_part = np.concatenate(ys_part, axis=1)
        ys_part_means.append(np.mean(ys_part, axis=1)[inds])
        ys_part_stds.append(np.std(ys_part, axis=1)[inds])
    
    ys_ppms = np.concatenate(ys_ppms, axis=0)
    ys_part_means = np.concatenate(ys_part_means, axis=0)
    ys_part_stds = np.concatenate(ys_part_stds, axis=0)
    
    inds = np.argsort(ys_ppms)
    
    return ys_ppms[inds], ys_part_means[inds], ys_part_stds[inds]
###########################################################################
###                               PIPNet                                ###
###                          Utility functions                          ###
###                        Author: Manuel Cordova                       ###
###                       Last edited: 2022-09-30                       ###
###########################################################################

from re import I
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import scipy as sp



###########################################################################
###                            1D functions                             ###
###########################################################################



def load_1d_topspin_spectrum(path, return_title=False):
    """
    Load a 1D spectrum from Topspin and retrieve the MAS rate
    The MAS rate is first looked for in the title of the experiment,
    and if it is not found there the MASR variable in the acquisition
    parameters will be searched.

    Input:      - path          Path to the Topspin directory
                - return_title  Return the title of the spectrum

    Outputs:    - dr            Real part of the spectrum
                - di            Imaginary part of the spectrum
                - wr            MAS rate (-1 if the rate is not found)
                - ppm           Array of chemical shift values in the spectrum
                - hz            Array of frequency values in the spectrum
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
        title = F.read()
        lines = title.split("\n")
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

    if return_title:
        return dr, di, wr, ppm, hz, title
    else:
        return dr, di, wr, ppm, hz



def extract_1d_dataset(path, exp_init, exp_final, exps=None, return_titles=False):
    """
    Extract a vmas dataset of 1D spectra

    Inputs:     - path              Path to the vmas dataset
                - exp_init          Index of the first experiment (inclusive)
                - exp_final         Index of the last experiment (inclusive)
                - exps              Custom indices of all experiments
                - return_titles     Return spectra titles

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
    titles = []

    if exps is None:
        exps = np.arange(exp_init, exp_final+1)
    
    for d in os.listdir(path):
        if d.isnumeric() and int(d) in exps:
            if return_titles:
                Xr, Xi, wr, ppm, hz, title = load_1d_topspin_spectrum(f"{path}{d}/", return_title=True)
                titles.append(title)
            else:
                Xr, Xi, wr, ppm, hz = load_1d_topspin_spectrum(f"{path}{d}/")
            X_real.append(Xr)
            X_imag.append(Xi)
            ws.append(wr)
    
    sorted_inds = np.argsort(ws)
    
    sorted_ws = np.array([ws[i] for i in sorted_inds])
    
    sorted_X_real = np.array([X_real[i] for i in sorted_inds])
    sorted_X_imag = np.array([X_imag[i] for i in sorted_inds])
    
    if return_titles:
        sorted_titles = [titles[i] for i in sorted_inds]
        return ppm, hz, sorted_ws, sorted_X_real, sorted_X_imag, sorted_titles
    else:
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



def plot_multiple_1d_iso_predictions(
    X,
    y_pred,
    y_std,
    pred_scale=1.,
    pred_offset=0.,
    xvals=None,
    x_trg=None,
    xinv=False,
    ylim=None,
    show=True,
    save=None,
):

    if xvals is None:
        xvals = np.arange(X.shape[-1])
    if x_trg is None:
        x_trg = xvals
    
    nsteps = y_pred.shape[0]
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(1,1,1)
    ax.plot(xvals, X[-1, 0] + (nsteps * pred_offset), linewidth=1., zorder=0)
    # Plot all prediction steps
    for step in range(nsteps):
        ax.plot(xvals, y_pred[-(step+1)] * pred_scale + pred_offset * step, "r", linewidth=1., zorder=-1)

        ax.fill_between(
            xvals,
            (y_pred[-(step+1)] - y_std[-(step+1)]) * pred_scale + pred_offset * step,
            (y_pred[-(step+1)] + y_std[-(step+1)]) * pred_scale + pred_offset * step,
            color="r",
            alpha=0.3,
            linewidth=0.,
        )

    if xinv:
        ax.invert_xaxis()
    if ylim is not None:
        ax.set_ylim(ylim)
    fig.tight_layout()
    if show:
        plt.show()
    if save is not None:
        fig.savefig(save)
    plt.close()

    return



def plot_1d_dataset(
    X,
    y=None,
    y_scale=1.,
    offset=0.,
    y_offset=0.,
    xvals=None,
    xinv=False,
    c0=np.array([0., 1., 1.]),
    dc=np.array([0., -1., 0.]),
    ylim=None,
    show=True,
    save=None,
):

    if xvals is None:
        xvals = np.arange(X.shape[-1])

    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(1,1,1)

    n = X.shape[0]

    for i, Xi in enumerate(X):
        ax.plot(xvals, Xi[0] + offset * i, linewidth=1., color=c0+(i/(n-1))*dc)
    
    if y is not None:
        ax.plot(xvals, y[0]* y_scale + y_offset, linewidth=1., color="r")

    if xinv:
        ax.invert_xaxis()
    if ylim is not None:
        ax.set_ylim(ylim)
    fig.tight_layout()
    if show:
        plt.show()
    if save is not None:
        fig.savefig(save)
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
    top = np.max(y[inds])
    x0 = x[inds[np.argmax(y[inds])]]
    
    xr = None
    xl = None
    
    # Search for crossing of the half maximum
    for i, j in zip(inds[:-1], inds[1:]):
        # Crossing on the right
        if y[i] > top / 2 and y[j] < top / 2:
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            dy2 = (top / 2) - y[i]
            xr = x[i] + dx * dy2 / dy
            
        # Crossing on the left
        if y[i] < top / 2 and y[j] > top / 2:
            dx = x[j] - x[i]
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



###########################################################################
###                            2D functions                             ###
###########################################################################



def load_2d_topspin_spectrum(path, return_title=False, load_imag=False):
    """
    Load a 2D spectrum from Topspin and retrieve the MAS rate
    The MAS rate is first looked for in the title of the experiment,
    and if it is not found there the MASR variable in the acquisition
    parameters will be searched.

    Input:      - path          Path to the Topspin directory
                - return_title  Return the title of the spectrum

    Outputs:    - dr            Real part of the spectrum
                - di            Imaginary part of the spectrum
                - wr            MAS rate (-1 if the rate is not found)
                - ppm           Array of chemical shift values in the spectrum
                - hz            Array of frequency values in the spectrum
    """
    
    if not path.endswith("/"):
        path += "/"
    
    if not os.path.exists(path):
        raise ValueError(f"{path} not found!")

    pd = f"{path}pdata/1/"
    frr = pd + "2rr"
    fri = pd + "2ri"
    fir = pd + "2ir"
    fii = pd + "2ii"
    ti = pd + "title"

    # Load spectrum
    with open(frr, "rb") as F:
        drr = np.fromfile(F, np.int32).astype(float)
    if load_imag:
        with open(fri, "rb") as F:
            dri = np.fromfile(F, np.int32).astype(float)
        with open(fir, "rb") as F:
            dir = np.fromfile(F, np.int32).astype(float)
        with open(fii, "rb") as F:
            dii = np.fromfile(F, np.int32).astype(float)
    else:
        dri = None
        dir = None
        dii = None
        
    wr = -1.
    # Get MAS rate from title
    wr_found = False
    with open(ti, "r") as F:
        title = F.read()
        lines = title.split("\n")
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
            TD_x = int(l.split("=")[1].strip())
        if l.startswith("##$SW_h="):
            SW_x = float(l.split("=")[1].strip())
    
    with open(f"{path}acqu2s", "r") as F:
        lines = F.read().split("\n")
    for l in lines:
        if l.startswith("##$TD="):
            TD_y = int(l.split("=")[1].strip())
        if l.startswith("##$SW_h="):
            SW_y = float(l.split("=")[1].strip())

    # Parse processing parameters
    with open(f"{pd}procs", "r") as F:
        lines = F.read().split("\n")

    for l in lines:
        if l.startswith("##$SI="):
            n_pts_x = int(l.split("=")[1].strip())
        if l.startswith("##$OFFSET="):
            offset_x = float(l.split("=")[1].strip())
        if l.startswith("##$SF="):
            SF_x = float(l.split("=")[1].strip())

    with open(f"{pd}proc2s", "r") as F:
        lines = F.read().split("\n")

    for l in lines:
        if l.startswith("##$SI="):
            n_pts_y = int(l.split("=")[1].strip())
        if l.startswith("##$OFFSET="):
            offset_y = float(l.split("=")[1].strip())
        if l.startswith("##$SF="):
            SF_y = float(l.split("=")[1].strip())
            
    # Generate chemical shift and frequency arrays
    AQ_x = TD_x / (2 * SW_x)
    hz_x = offset_x * SF_x - np.arange(n_pts_x) / (2 * AQ_x * n_pts_x / TD_x)
    ppm_x = hz_x / SF_x

    AQ_y = TD_y / (2 * SW_y)
    hz_y = offset_y * SF_y - np.arange(n_pts_y) / (2 * AQ_y * n_pts_y / TD_y)
    ppm_y = hz_y / SF_y

    if return_title:
        return drr, dri, dir, dii, wr, ppm_x, ppm_y, hz_x, hz_y, title
    else:
        return drr, dri, dir, dii, wr, ppm_x, ppm_y, hz_x, hz_y



def extract_2d_dataset(path, exp_init, exp_final, exps=None, return_titles=False, load_imag=False):
    """
    Extract a vmas dataset of 1D spectra

    Inputs:     - path              Path to the vmas dataset
                - exp_init          Index of the first experiment (inclusive)
                - exp_final         Index of the last experiment (inclusive)
                - exps              Custom indices of all experiments
                - return_titles     Return spectra titles

    Outputs:    - ppm               Array of chemical shifts
                - hz                Array of frequencies
                - sorted_ws         Array of MAS rates
                - sorted_X_real     Array of real parts of the spectra
                - sorted_X_imag     Array of imaginary parts of the spectra
    """
    
    if not path.endswith("/"):
        path += "/"
    
    if not os.path.exists(path):
        raise ValueError(f"{path} not found!")
    
    ws = []
    X_rr = []
    X_ri = []
    X_ir = []
    X_ii = []
    titles = []

    if exps is None:
        exps = np.arange(exp_init, exp_final+1)
    
    first = True
    for d in os.listdir(path):
        if d.isnumeric() and int(d) in exps:
            if return_titles:
                Xrr, Xri, Xir, Xii, wr, ppm_x, ppm_y, hz_x, hz_y, title = load_2d_topspin_spectrum(f"{path}{d}/", return_title=True, load_imag=load_imag)
                titles.append(title)
            else:
                Xrr, Xri, Xir, Xii, wr, ppm_x, ppm_y, hz_x, hz_y = load_2d_topspin_spectrum(f"{path}{d}/", load_imag=load_imag)
            
            Xrr = Xrr.reshape(len(ppm_y), len(ppm_x))
            if load_imag:
                Xri = Xri.reshape(len(ppm_y), len(ppm_x))
                Xir = Xir.reshape(len(ppm_y), len(ppm_x))
                Xii = Xii.reshape(len(ppm_y), len(ppm_x))

            if not first:
                f = sp.interpolate.interp2d(ppm_x, ppm_y, Xrr)
                Xrr = f(ppm_x0, ppm_y0)
                if load_imag:
                    f = sp.interpolate.interp2d(ppm_x, ppm_y, Xri)
                    Xri = f(ppm_x0, ppm_y0)
                    f = sp.interpolate.interp2d(ppm_x, ppm_y, Xir)
                    Xir = f(ppm_x0, ppm_y0)
                    f = sp.interpolate.interp2d(ppm_x, ppm_y, Xii)
                    Xii = f(ppm_x0, ppm_y0)
            else:
                first = False
                ppm_x0 = ppm_x
                ppm_y0 = ppm_y
                hz_x0 = hz_x
                hz_y0 = hz_y

            X_rr.append(Xrr)
            X_ri.append(Xri)
            X_ir.append(Xir)
            X_ii.append(Xii)
            ws.append(wr)
    
    ppm_x = ppm_x0
    ppm_y = ppm_y0
    hz_x = hz_x0
    hz_y = hz_y0
    
    sorted_inds = np.argsort(ws)
    
    sorted_ws = np.array([ws[i] for i in sorted_inds])
    
    sorted_X_rr = np.array([X_rr[i] for i in sorted_inds]).reshape(-1, len(ppm_y), len(ppm_x))
    if load_imag:
        sorted_X_ri = np.array([X_ri[i] for i in sorted_inds])
        sorted_X_ir = np.array([X_ir[i] for i in sorted_inds])
        sorted_X_ii = np.array([X_ii[i] for i in sorted_inds])
    else:
        sorted_X_ri = None
        sorted_X_ir = None
        sorted_X_ii = None

    if return_titles:
        sorted_titles = [titles[i] for i in sorted_inds]
        return ppm_x, ppm_y, hz_x, hz_y, sorted_ws, sorted_X_rr, sorted_X_ri, sorted_X_ir, sorted_X_ii, sorted_titles
    else:
        return ppm_x, ppm_y, hz_x, hz_y, sorted_ws, sorted_X_rr, sorted_X_ri, sorted_X_ir, sorted_X_ii



def prepare_2d_input(xrr, ws, data_pars, xri=None, xir=None, xii=None, xmax=0.5):

    X = torch.Tensor(xrr).unsqueeze(0).unsqueeze(2)

    Xint = torch.sum(X, dim=(3, 4))

    if data_pars["encode_imag"]:
        X = torch.cat([X, torch.Tensor(xri).unsqueeze(0).unsqueeze(2), torch.Tensor(xir).unsqueeze(0).unsqueeze(2), torch.Tensor(xii).unsqueeze(0).unsqueeze(2)], dim=2)
    
    # Normalize integrals
    X /= Xint[:, :, :, None, None]
    X /= torch.max(X[:, :, 0]) / xmax


    if data_pars["encode_wr"]:
        W = torch.tensor(ws) / data_pars["wr_norm_factor"]
        if data_pars["wr_inv"]:
            W = 1. / W
        W = W.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)

        X = torch.cat([X, W.repeat(1, 1, 1, X.shape[-2], X.shape[-1])], axis=2)

    return X.type(torch.float32)



def plot_2d_iso_prediction(
    X,
    y_pred,
    y_trg=None,
    xvals=None,
    yvals=None,
    xtrg=None,
    ytrg=None,
    wr_factor=1.,
    xinv=False,
    yinv=False,
    level_num=32,
    level_min=0.05,
    level_inc=None,
    level_neg=True,
    level_typ="mul",
    all_steps=False,
    normalize=True,
    lw=1.,
    equal_axes=True,
    show=True,
    save=None,
):

    if xvals is None:
        xvals = np.arange(X.shape[-1])
    if yvals is None:
        yvals = np.arange(X.shape[-2])
    
    if level_typ == "mul":
        if level_inc is None:
            level_inc = 1.1
        levels = np.array([level_min*(level_inc**i) for i in range(level_num)])
    elif level_typ == "add":
        if level_inc is None:
            level_inc = 0.01
        levels = np.array([level_min+(level_inc*i) for i in range(level_num)])
    else:
        raise ValueError(f"Unknown level type: {level_typ} (should be \"mul\" or \"add\")")

    if level_neg:
        neg_levels = -1. * levels[::-1]
    else:
        neg_levels = None
    
    if xtrg is None:
        xtrg = xvals
    if ytrg is None:
        ytrg = yvals
    
    XX, YY = np.meshgrid(xvals, yvals)
    XT, YT = np.meshgrid(xtrg, ytrg)
    
    if all_steps:
        nsteps = y_pred.shape[0]
    else:
        nsteps = 1

    # Plot all prediction steps
    for step in range(nsteps):

        if y_trg is None:
            fig = plt.figure(figsize=(6,3))
            ax1 = fig.add_subplot(1,2,1)
            ax2 = fig.add_subplot(1,2,2)
            ax3 = None
        else:
            fig = plt.figure(figsize=(9,3))
            ax1 = fig.add_subplot(1,3,1)
            ax2 = fig.add_subplot(1,3,2)
            ax3 = fig.add_subplot(1,3,3)

        wr = X[-(step+1), 1, 0, 0] * wr_factor / 1000.
        labels = [f"{wr:.0f} kHz MAS", "PIPNet"]
        hs = []

        if normalize:
            X[-(step+1), 0] /= np.max(X[-(step+1), 0])
            y_pred[-(step+1)] /= np.max(y_pred[-(step+1)])
            if y_trg is not None:
                y_trg /= np.max(y_trg)

        if np.max(X[-(step+1), 0]) > levels[0]:
            ax1.contour(XX, YY, X[-(step+1), 0], levels=levels, colors="C0", linewidths=lw)
            ax2.contour(XX, YY, X[-(step+1), 0], levels=levels, colors="C0", linewidths=lw)
            if ax3 is not None:
                ax3.contour(XX, YY, X[-(step+1), 0], levels=levels, colors="C0", linewidths=lw)
        if neg_levels is not None and np.min(X[-(step+1), 0]) < neg_levels[-1]:
            ax1.contour(XX, YY, X[-(step+1), 0], levels=neg_levels, colors="C2", linewidths=lw)
            ax2.contour(XX, YY, X[-(step+1), 0], levels=neg_levels, colors="C2", linewidths=lw)
            if ax3 is not None:
                ax3.contour(XX, YY, X[-(step+1), 0], levels=neg_levels, colors="C2", linewidths=lw)
        hs.append(mpl.lines.Line2D([0], [0], color="C0"))

        if np.max(y_pred[-(step+1)]) > levels[0]:
            ax2.contour(XX, YY, y_pred[-(step+1)], levels=levels, colors="r", linewidths=lw)
        hs.append(mpl.lines.Line2D([0], [0], color="r"))

        if y_trg is not None:
            labels.append("Ground-truth")
            if np.max(y_trg) > levels[0]:
                ax3.contour(XT, YT, y_trg, levels=levels, colors="k", linewidths=lw)
            hs.append(mpl.lines.Line2D([0], [0], color="k"))
        ax2.set_yticklabels([])
        if ax3 is not None:
            ax3.set_yticklabels([])
        
        if equal_axes:
            ax1.axis("equal")
            ax2.axis("equal")
            if ax3 is not None:
                ax3.axis("equal")
        if xinv:
            ax1.invert_xaxis()
            ax2.invert_xaxis()
            if ax3 is not None:
                ax3.invert_xaxis()
        if yinv:
            ax1.invert_yaxis()
            ax2.invert_yaxis()
            if ax3 is not None:
                ax3.invert_yaxis()
        fig.legend(hs, labels, bbox_to_anchor=(0.5, 1.), ncol=len(labels), loc="upper center")

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



def plot_multiple_2d_iso_predictions(
    X,
    y_pred,
    y_std,
    pred_scale=1.,
    pred_offset=0.,
    xvals=None,
    x_trg=None,
    xinv=False,
    ylim=None,
    show=True,
    save=None,
):

    #TODO
    raise NotImplementedError()

    if xvals is None:
        xvals = np.arange(X.shape[-1])
    if x_trg is None:
        x_trg = xvals
    
    nsteps = y_pred.shape[0]
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(1,1,1)
    ax.plot(xvals, X[-1, 0] + (nsteps * pred_offset), linewidth=1., zorder=0)
    # Plot all prediction steps
    for step in range(nsteps):
        ax.plot(xvals, y_pred[-(step+1)] * pred_scale + pred_offset * step, "r", linewidth=1., zorder=-1)

        ax.fill_between(
            xvals,
            (y_pred[-(step+1)] - y_std[-(step+1)]) * pred_scale + pred_offset * step,
            (y_pred[-(step+1)] + y_std[-(step+1)]) * pred_scale + pred_offset * step,
            color="r",
            alpha=0.3,
            linewidth=0.,
        )

    if xinv:
        ax.invert_xaxis()
    if ylim is not None:
        ax.set_ylim(ylim)
    fig.tight_layout()
    if show:
        plt.show()
    if save is not None:
        fig.savefig(save)
    plt.close()

    return



def plot_2d_dataset(
    X,
    y=None,
    y_scale=1.,
    offset=0.,
    y_offset=0.,
    xvals=None,
    xinv=False,
    c0=np.array([0., 1., 1.]),
    dc=np.array([0., -1., 0.]),
    ylim=None,
    show=True,
    save=None,
):

    #TODO
    raise NotImplementedError()

    if xvals is None:
        xvals = np.arange(X.shape[-1])

    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(1,1,1)

    n = X.shape[0]

    for i, Xi in enumerate(X):
        ax.plot(xvals, Xi[0] + offset * i, linewidth=1., color=c0+(i/(n-1))*dc)
    
    if y is not None:
        ax.plot(xvals, y[0]* y_scale + y_offset, linewidth=1., color="r")

    if xinv:
        ax.invert_xaxis()
    if ylim is not None:
        ax.set_ylim(ylim)
    fig.tight_layout()
    if show:
        plt.show()
    if save is not None:
        fig.savefig(save)
    plt.close()

    return



def extract_2d_linewidth(x, y, x_range, verbose=False):
    """
    Extract the linewidth (FWHM) of a peak in the given x-axis range

    Inputs:     - x         x-axis values
                - y         y-axis values
                - x_range   Range of x-values containing the peak

    Outputs:    - lw        Extracted linewidth (FWHM)
                - x0        Extracted peak position
    """

    #TODO
    raise NotImplementedError()

    xmin = np.min(x_range)
    xmax = np.max(x_range)

    # Select range and identify top of the peak
    inds = np.where(np.logical_and(x >= xmin, x <= xmax))[0]
    top = np.max(y[inds])
    x0 = x[inds[np.argmax(y[inds])]]
    
    xr = None
    xl = None
    
    # Search for crossing of the half maximum
    for i, j in zip(inds[:-1], inds[1:]):
        # Crossing on the right
        if y[i] > top / 2 and y[j] < top / 2:
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            dy2 = (top / 2) - y[i]
            xr = x[i] + dx * dy2 / dy
            
        # Crossing on the left
        if y[i] < top / 2 and y[j] > top / 2:
            dx = x[j] - x[i]
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



def extract_2d_linewidths(x, y, x_ranges):

    #TODO
    raise NotImplementedError()

    lws = np.zeros(len(x_ranges))
    pks = np.zeros(len(x_ranges))

    for i, x_range in enumerate(x_ranges):
        lws[i], pks[i] = extract_1d_linewidth(x, y, x_range)

    return lws, pks



def get_relative_2d_integrals(x, y, regions):
    """
    Compare the relative integrals in different regions of two spectra

    Inputs:     - x             x-axis values
                - y             y-axis values
                - regions       Regions to compute the integrals for

    Outputs:    - integrals     List of integrals for each region
    """

    #TODO
    raise NotImplementedError()

    integrals = np.zeros(len(regions))

    for i, (r1, r2) in enumerate(regions):
        i1 = np.argmin(np.abs(x - r1))
        i2 = np.argmin(np.abs(x - r2))

        il = min(i1, i2)
        ir = max(i1, i2)

        integrals[i] = np.sum(y[il:ir])
        
    integrals /= np.sum(integrals)

    return integrals



def extract_2d_pip(in_dir, compound, parts, res):
    """
    Extract 1d PIP prediction
    """

    #TODO
    raise NotImplementedError()
    
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
###############################################################################
#                                   PIPNet                                    #
#                              Utility functions                              #
#                            Author: Manuel Cordova                           #
#                           Last edited: 2023-10-17                           #
###############################################################################


import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from scipy import interpolate as ip


###############################################################################
#                               Misc functions                                #
###############################################################################


def is_2D(path, procno=1, bypass_errors=False):
    """Identify whether the spectrum is 2D.

    Parameters
    ----------
    path : str
        Path to the Topspin dataset.
    procno : int, default=1
        procno of the data to load.
    bypass_errors : bool, default=False
        Bypass errors and continue execution anyway.

    Returns
    -------
    is_2D : bool
        `True` if the spectrum is 2D, `False` if it is 1D.
    msg : str
        Warning/error message.
    """

    msg = ""

    if not path.endswith("/"):
        path += "/"

    if os.path.exists(f"{path}pdata/{procno}/1r"):
        return False, msg

    if os.path.exists(f"{path}pdata/{procno}/2rr"):
        return True, msg

    msg += f"ERROR: Did not find processed 1D or 2D data in {path}!\n"

    if bypass_errors:
        return None, msg

    raise ValueError(msg)


def get_procnos(path, bypass_errors=False):
    """Get the procnos in a dataset.

    Parameters
    ----------
    path : str
        Path to the Topspin dataset.
    bypass_errors : bool, default=False
        Bypass errors and continue execution anyway.

    Returns
    -------
    procnos : list
        List of available procnos
    msg : str
        Warning/error message.
    """

    procnos = []
    msg = ""

    if not os.path.exists(f"{path}pdata/"):
        msg += f"ERROR: Dataset does not exist: {path}\n"

        if not bypass_errors:
            raise ValueError(msg)

    for d in os.listdir(f"{path}pdata/"):
        if d.isnumeric() and os.path.isdir(f"{path}pdata/{d}"):
            procnos.append(int(d))

    return procnos, msg


def load_1d_data(path, procno=1, load_imag=True):
    """Load the data contained in a 1D spectrum.

    Parameters
    ----------
    path : str
        Path to the Topspin dataset.
    procno : int, default=1
        procno of the data to load.
    load_imag : bool, default=False
        Whether or not to load the imaginary part of the spectrum.


    Returns
    -------
    dr : Numpy ndarray
        Real part of the spectrum.
    di : Numpy ndarray or None
        Imaginary part of the spectrum.
    msg : str
        Warning/error message.
    """

    fr = f"{path}pdata/{procno}/1r"
    fi = f"{path}pdata/{procno}/1ri"

    msg = ""

    with open(fr, "rb") as F:
        dr = np.fromfile(F, np.int32).astype(float)
    if load_imag and os.path.exists(fi):
        with open(fi, "rb") as F:
            di = np.fromfile(F, np.int32).astype(float)
    else:
        di = None
        if load_imag:
            msg = "WARNING: No imaginary part found!\n"

    return dr, di, msg


def parse_title(path, procno=1, prev_msg=""):
    """Load spectrum title and get MAS rate.

    Parameters
    ----------
    path : str
        Path to the Topspin directory.
    procno : int, default=1
        procno of the data to load.
    prev_msg : str, default=""
        Previous warning/error message.

    Returns
    -------
    title : str
        Title of the spectrum.
    wr : float
        MAS rate extracted. If no MAS rate is found, returns -1.0.
    msg : str
        Warning/error message.
    """

    msg = prev_msg

    ti = f"{path}pdata/{procno}/title"
    if not os.path.exists(ti):

        if procno == 1:

            this_msg = f"WARNING: No title found in {ti}!\n"
            msg += this_msg
            return "", -1., msg

        else:

            this_msg = f"WARNING: No title found for procno {procno}"
            this_msg += f" in {path}. Falling back to procno 1.\n"
            msg += this_msg

            return parse_title(
                path,
                procno=1,
                prev_msg=msg
            )

    wr = -1.
    with open(ti, "r") as F:
        title = F.read()
        lines = title.split("\n")
    for line in lines:
        if "KHZ" in line.upper():
            wr = float(line.upper().split("KHZ")[0].split()[-1]) * 1000
        elif "HZ" in line.upper():
            wr = float(line.upper().split("HZ")[0].split()[-1])

    return title, wr, msg


def parse_acqus(path, use_acqu2s=False, bypass_errors=False):
    """Parse acquisition parameters to extract time domain and spectral width.

    Parameters
    ----------
    path : str
        Path to the Topspin directory.
    use_acqu2s : bool, default=False
        Use the acqu2s file instead of acqus.
    bypass_errors : bool, default=False
        Bypass errors and continue execution anyway.

    Returns
    -------
    td : int
        Number of points in the time domain.
    sw : float
        Spectral width (in Hertz).
    wr : float
        MAS rate stored in the "$MASR" parameter.
    msg : str
        Warning/error message.
    """

    wr = -1.
    sw = None
    td = None
    msg = ""

    acq = f"{path}acqus"
    if use_acqu2s:
        acq = f"{path}acqu2s"

    if not os.path.exists(acq):
        msg += f"ERROR: File {acq} does not exist!\n"

        if bypass_errors:
            return td, sw, wr, msg

        raise ValueError(msg)

    with open(acq, "r") as F:
        lines = F.read().split("\n")

    for line in lines:

        if line.startswith("##$MASR"):
            try:
                wr = int(line.split("=")[1].strip())
            except ValueError:
                pass

        if line.startswith("##$TD="):
            td = int(line.split("=")[1].strip())

        if line.startswith("##$SW_h="):
            sw = float(line.split("=")[1].strip())

    if td is None:
        msg += f"ERROR: No TD found in {path}!\n"
    if sw is None:
        msg += f"ERROR: No SW found in {path}!\n"

    if (td is None or sw is None) and not bypass_errors:
        raise ValueError(msg)

    return td, sw, wr, msg


def parse_procs(path, use_proc2s=False, bypass_errors=False):
    """Parse processing parameters to extract number of points in the spectrum,
    offset, and spectrometer frequency.

    Parameters
    ----------
    path : str
        Path to the Topspin directory.
    use_proc2s: bool, default=False
        Use que proc2s file instead of procs.
    bypass_errors : bool, default=False
        Bypass errors and continue execution anyway.

    Returns
    -------
    n_pts : int
        Number of points in the spectrum.
    offset : float
        Offset of the spectrum.
    sf : float
        Spectrometer frequency.
    msg : str
        Warning/error message.
    """

    n_pts = None
    offset = None
    sf = None
    msg = ""

    prc = f"{path}procs"
    if use_proc2s:
        prc = f"{path}proc2s"

    if not os.path.exists(prc):
        msg += f"ERROR: File {prc} does not exist!\n"

        if bypass_errors:
            return n_pts, offset, sf, msg

        raise ValueError(msg)

    with open(prc, "r") as F:
        lines = F.read().split("\n")

    for line in lines:

        if line.startswith("##$SI="):
            n_pts = int(line.split("=")[1].strip())

        if line.startswith("##$OFFSET="):
            offset = float(line.split("=")[1].strip())

        if line.startswith("##$SF="):
            sf = float(line.split("=")[1].strip())

    if n_pts is None:
        msg += f"ERROR: No SI found in {path}!\n"
    if offset is None:
        msg += f"ERROR: No OFFSET found in {path}!\n"
    if sf is None:
        msg += f"ERROR: No SF found in {path}!\n"

    if (n_pts is None or offset is None or sf is None) and not bypass_errors:
        raise ValueError(msg)

    return n_pts, offset, sf, msg


def get_chemical_shifts(td, sw, n_pts, offset, sf, bypass_errors=False):
    """Get chemical shifts and frequencies in the spectrum.

    Parameters
    ----------
    td : int
        Number of points in time domain.
    sw : float
        Spectral width (in Hertz).
    n_pts : int
        Number of points in the spectrum.
    offset : float
        Offset chemical shift of the spectrum.
    sf : float
        Spectrometer frequency.
    bypass_errors : bool, default=False
        Bypass errors and continue execution anyway.

    Returns
    -------
    ppm : Numpy ndarray
        Array of chemical shift values.
    hz : Numpy ndarray
        Array of frequency values.
    msg : str
    """

    msg = ""

    print(td)
    print(sw)
    print(n_pts)
    print(offset)
    print(sf)

    try:
        aq = td / (2 * sw)
        hz = offset * sf - np.arange(n_pts) / (2 * aq * n_pts / td)
        ppm = hz / sf

        print(aq)
        print(hz)
        print(ppm)

    except Exception:
        msg = "Could not construct array of chemical shifts!\n"
        if not bypass_errors:
            raise ValueError(msg)

    return ppm, hz, msg


###############################################################################
#                                1D functions                                 #
###############################################################################


def load_1d_topspin_spectrum(
    path,
    load_imag=True,
    procno=1,
    use_acqu2s=False,
    use_proc2s=False,
    bypass_errors=False
):
    """Load a 1D Topspin spectrum.

    Parameters
    ----------
    path : str
        Path to the Topspin directory.
    load_imag : bool, default=True
        Load the imaginary part of the spectrum.
    procno : int, default=1
        Procno to load.
    use_acqu2s : bool, default=False
        Use the acqu2s file instead of acqus to get chemical shifts.
        This is useful when the data is extracted from a 2D spectrum and the
        x-axis corresponds to the F1 dimension.
    use_proc2s : bool, default=False
        Use the proc2s file instead of procs to get processing parameters.
        This is useful when the data is extracted from a 2D spectrum and the
        x-axis corresponds to the F1 dimension.
    bypass_errors : bool, default=False
        Bypass errors and continue execution anyway.

    Returns
    -------
    dr : Numpy ndarray
        Real part of the spectrum.
    di : Numpy ndarray
        Imaginary part of the spectrum.
        Will be zeros if no imaginary part is loaded.
    wr : float
        MAS rate (-1 if the rate is not found).
    ppm : Numpy ndarray
        Array of chemical shift values.
    hz : Numpy ndarray
        Array of frequency values.
    title : str
        Title of the spectrum.
    msg : str
        Warning/error message.
    """

    msg = ""

    if not path.endswith("/"):
        path += "/"

    if not os.path.exists(path):
        msg += f"ERROR: Topspin directory does not exist: {path}.\n"
        if bypass_errors:
            return None, None, None, None, None, None, msg

        raise ValueError(msg)

    pd = f"{path}pdata/{procno}/"
    if not os.path.exists(pd):
        msg += f"ERROR: procno {procno} does not exist"
        msg += f" in this Topspin directory: {path}!\n"

        if bypass_errors:
            return None, None, None, None, None, None, msg

        raise ValueError(msg)

    # Load spectrum
    dr, di, this_msg = load_1d_data(
        path,
        procno=procno,
        load_imag=load_imag
    )
    msg += this_msg

    if "ERROR" in msg:
        return None, None, None, None, None, None, msg

    # Get title and MAS rate
    title, wr, this_msg = parse_title(
        path,
        procno=procno
    )
    msg += this_msg

    if "ERROR" in msg:
        return None, None, None, None, None, None, msg

    # Parse acquisition parameters
    td, sw, _wr, this_msg = parse_acqus(
        path,
        use_acqu2s=use_acqu2s,
        bypass_errors=bypass_errors
    )
    msg += this_msg

    if "ERROR" in msg:
        return None, None, None, None, None, None, msg

    if wr < 0. and _wr > 0.:
        wr = _wr

    if wr < 0.:
        msg += f"WARNING: No MAS rate found for {pd}!\n"

    # Parse processing parameters
    n_pts, offset, sf, this_msg = parse_procs(
        pd,
        use_proc2s=use_proc2s,
        bypass_errors=bypass_errors
    )
    msg += this_msg

    if "ERROR" in msg:
        return None, None, None, None, None, None, msg

    # Generate chemical shift and frequency arrays
    ppm, hz, this_msg = get_chemical_shifts(td, sw, n_pts, offset, sf)
    msg += this_msg

    return dr, di, wr, ppm, hz, title, msg


def extract_1d_dataset(
    path,
    expno_init=1,
    expno_final=1000,
    expnos=None,
    load_imag=True,
    procno=1,
    use_acqu2s=False,
    use_proc2s=False,
    bypass_errors=False
):
    """Extract a dataset of 1D spectra.

    Parameters
    ----------
    path : str
        Path to the dataset containing the Topspin directories.
    expno_init : int, default=1
        Initial expno to parse (inclusive).
    expno_final : int, default=1000
        Final expno to parse (inclusive).
    expnos : array_like
        Custom indices of expnos to parse. This will override the
        `expno_init` and `expno_final` variables.
    load_imag : bool, default=True
        Load the imaginary part of the spectra.
    procno : int, default=1
        Procno to load.
    use_acqu2s : bool, default=False
        Use the acqu2s file instead of acqus to get chemical shifts.
        This is useful when the data is extracted from 2D spectra and the
        x-axis corresponds to the F1 dimension.
    use_proc2s : bool, default=False
        Use the proc2s file instead of procs to get processing parameters.
        This is useful when the data is extracted from 2D spectra and the
        x-axis corresponds to the F1 dimension.
    bypass_errors : bool, default=False
        Bypass errors and continue execution anyway.

    Returns
    -------
    ppm : Numpy ndarray
        Array of chemical shift values.
    hz : Numpy ndarray
        Array of frequency values.
    sorted_ws : Numpy ndarray
        Sorted array of MAS rates.
    sorted_X_real : Numpy ndarray
        Array of real parts of the spectra,
        sorted by increasing MAS rates.
    sorted_X_imag : Numpy ndarray
        Array of imaginary parts of the spectra,
        sorted by increasing MAS rates.
    msg : str
        Warning/error message.
    """

    if not path.endswith("/"):
        path += "/"

    ppm = None
    hz = None
    ws = []
    X_real = []
    X_imag = []
    titles = []
    msg = ""

    if expnos is None:
        expnos = np.arange(expno_init, expno_final+1)

    for expno in expnos:
        if os.path.isdir(f"{path}{expno}/"):
            (
                xr,
                xi,
                wr,
                this_ppm,
                this_hz,
                title,
                this_msg
            ) = load_1d_topspin_spectrum(
                f"{path}{expno}/",
                load_imag=load_imag,
                procno=procno,
                use_acqu2s=use_acqu2s,
                use_proc2s=use_proc2s,
                bypass_errors=bypass_errors
            )
            msg += this_msg

            if ppm is None:
                ppm = this_ppm
                hz = this_hz

            elif this_ppm is not None:
                if len(this_ppm) != len(ppm) or np.max(this_ppm - ppm) > 1e-3:
                    msg += "WARNING: Inconsistent chemical shift values."
                    msg += " Interpolating chemical shift.\n"

                    f = ip.RegularGridInterpolator(
                        (this_ppm, ),
                        xr,
                        bounds_error=False,
                        fill_value=0.
                    )
                    xr = f((ppm, ))

                    if xi is not None:
                        f = ip.RegularGridInterpolator(
                            (this_ppm, ),
                            xi,
                            bounds_error=False,
                            fill_value=0.
                        )
                        xi = f(ppm)

            titles.append(title)
            X_real.append(xr)
            X_imag.append(xi)
            ws.append(wr)

            if "ERROR" in msg:
                return None, None, None, None, None, None, msg

    sorted_inds = np.argsort(ws)

    sorted_ws = np.array([ws[i] for i in sorted_inds])
    sorted_titles = [titles[i] for i in sorted_inds]

    sorted_X_real = np.array([X_real[i] for i in sorted_inds])
    sorted_X_imag = np.array([X_imag[i] for i in sorted_inds])

    return ppm, hz, sorted_ws, sorted_X_real, sorted_X_imag, sorted_titles, msg


def prepare_1d_input(
    xr,
    ppm,
    ppm_range,
    ws,
    data_pars,
    xi=None,
    x_other=None,
    xmax=0.5
):
    """Prepare 1D spectra for processing using PIPNet.

    Parameters
    ----------
    xr : Numpy ndarray
        Array of the real part of the spectra.
    ppm : Numpy ndarray
        Array of chemical shift values in the spectra.
    ppm_range : array_like
        Array of chemical shift boundaries.
    ws : Numpy ndarray
        Array of MAS rates (in Hertz).
    data_pars : dict
        Dictionary of parameters for data representation.
    xi : None or Numpy ndarray, default=None
        Array of the imaginary part of the spectra.
    other : Numpy ndarray or None, default=None
        Array of frequency values in the spectra.
    xmax : float
        Maximum intensity of the real part of the spectra.

    Returns
    -------
    ppm : Numpy ndarray
        Array of chemical shift values within the boundaries
    other : Numpy ndarray
        Array of frequency values within the boundaries (if `x_other`is set)
    X : torch Tensor
        Pytorch tensor of the processed spectra.
    msg : str
        Warning/error message.
    """

    msg = ""

    ppm_min = np.min(ppm_range)
    ppm_max = np.max(ppm_range)
    mask = (ppm >= ppm_min) & (ppm <= ppm_max)

    X = torch.Tensor(xr[:, mask]).unsqueeze(0).unsqueeze(2)

    Xint = torch.sum(X, dim=3)

    # Encode imaginary part of the spectrum
    if data_pars["encode_imag"]:
        if None in xi:
            msg += "WARNING: No imaginary part present"
            msg += " in at least one spectrum!\n"
            xi = np.zeros_like(xr)

        X = torch.cat(
            [
                X,
                torch.Tensor(xi[:, mask]).unsqueeze(0).unsqueeze(2)
            ],
            dim=2
        )

    # Normalize integrals
    X /= Xint[:, :, :, None]
    X /= torch.max(X[:, :, 0]) / xmax

    # Encode MAS rate
    if data_pars["encode_wr"]:
        W = torch.tensor(ws) / data_pars["wr_norm_factor"]
        if data_pars["wr_inv"]:
            W = 1. / W
        W = W.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        X = torch.cat([X, W.repeat(1, 1, 1, X.shape[-1])], axis=2)

    return (
        ppm[mask],
        (None if x_other is None else x_other[mask]),
        X.type(torch.float32),
        msg
    )


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


###############################################################################
#                                2D functions                                 #
###############################################################################


def load_2d_data(path, procno=1, load_imag=True, dtype=np.int32):
    """Load the data contained in a 2D spectrum.

    Parameters
    ----------
    path : str
        Path to the Topspin dataset.
    procno : int, default=1
        procno of the data to load.
    load_imag : bool, default=False
        Whether or not to load the imaginary part of the spectrum.
    dtype : type or str, default=np.int32
        Type to read.

    Returns
    -------
    drr : Numpy ndarray
        Real-real part of the spectrum.
    dri : Numpy ndarray
        Real-imaginary part of the spectrum.
    dir : Numpy ndarray
        Imaginary-real part of the spectrum.
    dii : Numpy ndarray or None
        Imaginary-imaginary part of the spectrum.
    msg : str
        Warning/error message.
    """

    frr = f"{path}pdata/{procno}/2rr"
    fri = f"{path}pdata/{procno}/2ri"
    fir = f"{path}pdata/{procno}/2ir"
    fii = f"{path}pdata/{procno}/2ii"

    msg = ""

    with open(frr, "rb") as F:
        drr = np.fromfile(F, dtype=dtype).astype(float)

    dri = None
    dir = None
    dii = None
    if load_imag:
        if os.path.exists(fri):
            with open(fri, "rb") as F:
                dri = np.fromfile(F, dtype=dtype).astype(float)
        if os.path.exists(fir):
            with open(fir, "rb") as F:
                dir = np.fromfile(F, dtype=dtype).astype(float)
        if os.path.exists(fii):
            with open(fii, "rb") as F:
                dii = np.fromfile(F, dtype=dtype).astype(float)

    if load_imag and (dri is None or dir is None or dii is None):
        msg = "WARNING: No imaginary part found!\n"

    return drr, dri, dir, dii, msg


def load_2d_topspin_spectrum(
    path,
    load_imag=True,
    procno=1,
    bypass_errors=False,
    dtype=np.int32
):
    """Load a 2D Topspin spectrum.

    Parameters
    ----------
    path : str
        Path to the Topspin directory.
    load_imag : bool, default=True
        Load the imaginary part of the spectrum.
    procno : int, default=1
        Procno to load.
    bypass_errors : bool, default=False
        Bypass errors and continue execution anyway.
    dtype : type or str, default=np.int32
        Type to read.

    Returns
    -------
    drr : Numpy ndarray
        Real-real part of the spectrum.
    dri : Numpy ndarray
        Real-imaginary part of the spectrum.
    dir : Numpy ndarray
        Imaginary-real part of the spectrum.
    dii : Numpy ndarray
        Imaginary-imaginary part of the spectrum.
        Will be zeros if no imaginary part is loaded.
    wr : float
        MAS rate (-1 if the rate is not found).
    ppm_x : Numpy ndarray
        Array of chemical shift values in F2.
    ppm_y : Numpy ndarray
        Array of chemical shift values in F1.
    hz_x : Numpy ndarray
        Array of frequency values in F2.
    hz_y : Numpy ndarray
        Array of frequency values in F1.
    title : str
        Title of the spectrum.
    msg : str
        Warning/error message.
    """

    msg = ""

    if not path.endswith("/"):
        path += "/"

    if not os.path.exists(path):
        msg += f"ERROR: Topspin directory does not exist: {path}.\n"
        if bypass_errors:
            return (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                msg
            )

        raise ValueError(msg)

    pd = f"{path}pdata/{procno}/"
    if not os.path.exists(pd):
        msg += f"ERROR: procno {procno} does not exist"
        msg += f" in this Topspin directory: {path}!\n"

        if bypass_errors:
            return None, None, None, None, None, None, msg

        raise ValueError(msg)

    # Load spectrum
    drr, dri, dir, dii, this_msg = load_2d_data(
        path,
        procno=procno,
        load_imag=load_imag,
        dtype=dtype
    )
    msg += this_msg

    if "ERROR" in msg:
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            msg
        )

    # Get title and MAS rate
    title, wr, this_msg = parse_title(
        path,
        procno=procno
    )

    msg += this_msg
    if "ERROR" in msg:
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            msg
        )

    # Parse acquisition parameters
    td_x, sw_x, _wr, this_msg = parse_acqus(
        path,
        use_acqu2s=False,
        bypass_errors=bypass_errors
    )
    msg += this_msg
    if "ERROR" in msg:
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            msg
        )
    if wr < 0. and _wr > 0.:
        wr = _wr

    td_y, sw_y, _wr, this_msg = parse_acqus(
        path,
        use_acqu2s=True,
        bypass_errors=bypass_errors
    )
    msg += this_msg
    if "ERROR" in msg:
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            msg
        )
    if wr < 0. and _wr > 0.:
        wr = _wr

    if wr < 0.:
        msg += f"WARNING: No MAS rate found for {pd}!\n"

    # Parse processing parameters
    n_pts_x, offset_x, sf_x, this_msg = parse_procs(
        pd,
        use_proc2s=False,
        bypass_errors=bypass_errors
    )
    msg += this_msg
    if "ERROR" in msg:
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            msg
        )

    n_pts_y, offset_y, sf_y, this_msg = parse_procs(
        pd,
        use_proc2s=True,
        bypass_errors=bypass_errors
    )
    msg += this_msg
    if "ERROR" in msg:
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            msg
        )

    # Generate chemical shift and frequency arrays
    ppm_x, hz_x, this_msg = get_chemical_shifts(
        td_x,
        sw_x,
        n_pts_x,
        offset_x,
        sf_x
    )
    ppm_y, hz_y, this_msg = get_chemical_shifts(
        td_y,
        sw_y,
        n_pts_y,
        offset_y,
        sf_y
    )

    drr = drr.reshape((n_pts_y, n_pts_x))
    if dri is not None:
        dri = dri.reshape((n_pts_y, n_pts_x))
    if dir is not None:
        dir = dir.reshape((n_pts_y, n_pts_x))
    if dii is not None:
        dii = dii.reshape((n_pts_y, n_pts_x))

    return drr, dri, dir, dii, wr, ppm_x, ppm_y, hz_x, hz_y, title, msg


def extract_2d_dataset(
    path,
    expno_init=1,
    expno_final=1000,
    expnos=None,
    load_imag=True,
    procno=1,
    bypass_errors=False
):
    """Extract a dataset of 2D spectra.

    Parameters
    ----------
    path : str
        Path to the dataset containing the Topspin directories.
    expno_init : int, default=1
        Initial expno to parse (inclusive).
    expno_final : int, default=1000
        Final expno to parse (inclusive).
    expnos : array_like
        Custom indices of expnos to parse. This will override the
        `expno_init` and `expno_final` variables.
    load_imag : bool, default=True
        Load the imaginary part of the spectra.
    procno : int, default=1
        Procno to load.
    bypass_errors : bool, default=False
        Bypass errors and continue execution anyway.

    Returns
    -------
    ppm_x : Numpy ndarray
        Array of chemical shift values in F2.
    ppm_y : Numpy ndarray
        Array of chemical shift values in F1.
    hz_x : Numpy ndarray
        Array of frequency values in F2.
    hz_y : Numpy ndarray
        Array of frequency values in F1.
    sorted_ws : Numpy ndarray
        Sorted array of MAS rates.
    sorted_X_rr : Numpy ndarray
        Array of real-real parts of the spectra,
        sorted by increasing MAS rates.
    sorted_X_ri : Numpy ndarray
        Array of real-imaginary parts of the spectra,
        sorted by increasing MAS rates.
    sorted_X_ir : Numpy ndarray
        Array of imaginary-real parts of the spectra,
        sorted by increasing MAS rates.
    sorted_X_ii : Numpy ndarray
        Array of imaginary-imaginary parts of the spectra,
        sorted by increasing MAS rates.
    msg : str
        Warning/error message.
    """

    if not path.endswith("/"):
        path += "/"

    ppm_x = None
    ppm_y = None
    hz_x = None
    hz_y = None
    ws = []
    X_rr = []
    X_ri = []
    X_ir = []
    X_ii = []
    titles = []
    msg = ""

    if expnos is None:
        expnos = np.arange(expno_init, expno_final+1)

    for expno in expnos:
        if os.path.isdir(f"{path}{expno}/"):
            (
                xrr,
                xri,
                xir,
                xii,
                wr,
                this_ppm_x,
                this_ppm_y,
                this_hz_x,
                this_hz_y,
                title,
                this_msg
            ) = load_2d_topspin_spectrum(
                f"{path}{expno}/",
                load_imag=load_imag,
                procno=procno,
                bypass_errors=bypass_errors
            )
            msg += this_msg

            if ppm_x is None:
                ppm_x = this_ppm_x
                ppm_y = this_ppm_y
                hz_x = this_hz_x
                hz_y = this_hz_y

                if ppm_x is not None:
                    ppm_xx, ppm_yy = np.meshgrid(ppm_x, ppm_y)

            elif this_ppm_x is not None:
                if (
                    len(this_ppm_x) != len(ppm_x) or
                    len(this_ppm_y) != len(ppm_y) or
                    np.max(this_ppm_x - ppm_x) > 1e-3 or
                    np.max(this_ppm_y - ppm_y) > 1e-3
                ):
                    msg += "WARNING: Inconsistent chemical shift values."
                    msg += " Interpolating chemical shift.\n"

                    f = ip.RegularGridInterpolator(
                        (this_ppm_x, this_ppm_y),
                        xrr.T,
                        bounds_error=False,
                        fill_value=0.
                    )
                    xrr = f((ppm_xx, ppm_yy))

                    if xri is not None:
                        f = ip.RegularGridInterpolator(
                            (this_ppm_x, this_ppm_y),
                            xri.T,
                            bounds_error=False,
                            fill_value=0.
                        )
                        xri = f((ppm_xx, ppm_yy))

                    if xir is not None:
                        f = ip.RegularGridInterpolator(
                            (this_ppm_x, this_ppm_y),
                            xir.T,
                            bounds_error=False,
                            fill_value=0.
                        )
                        xir = f((ppm_xx, ppm_yy))

                    if xii is not None:
                        f = ip.RegularGridInterpolator(
                            (this_ppm_x, this_ppm_y),
                            xii.T,
                            bounds_error=False,
                            fill_value=0.
                        )
                        xii = f((ppm_xx, ppm_yy))

            titles.append(title)
            X_rr.append(xrr)
            X_ri.append(xri)
            X_ir.append(xir)
            X_ii.append(xii)
            ws.append(wr)

    sorted_inds = np.argsort(ws)

    sorted_ws = np.array([ws[i] for i in sorted_inds])
    sorted_titles = [titles[i] for i in sorted_inds]

    sorted_X_rr = np.array([X_rr[i] for i in sorted_inds])
    sorted_X_ri = np.array([X_ri[i] for i in sorted_inds])
    sorted_X_ir = np.array([X_ir[i] for i in sorted_inds])
    sorted_X_ii = np.array([X_ii[i] for i in sorted_inds])

    return (
        ppm_x,
        ppm_y,
        hz_x,
        hz_y,
        sorted_ws,
        sorted_X_rr,
        sorted_X_ri,
        sorted_X_ir,
        sorted_X_ii,
        sorted_titles,
        msg
    )


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
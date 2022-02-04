#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import sys
import copy

import torch
from torch import nn

import importlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as anm

sys.path.insert(0, "../src/")
import data
import model
import train

import nmrglue as ng
import scipy
import scipy.io

import gradio as gr
import zipfile as zp
import io
import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"
plt.rcParams["figure.max_open_warning"] = 100


# # Load model

# In[2]:


mod = "Ensemble_PIPNet_2022_01_25_batch_6"

in_dir = f"../data/{mod}/"

if not os.path.exists(in_dir):
    raise ValueError(f"Unknown model: {mod}")


# In[3]:


def clean_split(l, delimiter):
    """
    Split a line with the desired delimiter, ignoring delimiters present in arrays or strings

    Inputs: - l     Input line

    Output: - ls    List of sub-strings making up the line
    """

    # Initialize sub-strings
    ls = []
    clean_l = ""

    # Loop over all line characters
    in_dq = False
    in_sq = False
    arr_depth = 0
    for li in l:
        # Identify strings with double quotes
        if li == "\"":
            if not in_dq:
                in_dq = True
            else:
                in_dq = False

        # Identify strings with single quotes
        if li == "\'":
            if not in_sq:
                in_sq = True
            else:
                in_sq = False

        # Identify arrays
        if li == "[":
            if not in_sq and not in_dq:
                arr_depth += 1
        if li == "]":
            if not in_sq and not in_dq:
                arr_depth -= 1

        # If the delimiter is not within quotes or in an array, split the line at that character
        if li == delimiter and not in_dq and not in_sq and arr_depth == 0:
            ls.append(clean_l)
            clean_l = ""
        else:
            clean_l += li

    ls.append(clean_l)

    return ls


# In[4]:


def get_array(l):
    """
    Get the values in an array contained in a line

    Input:  - l         Input line

    Output: - vals      Array of values
    """

    # Identify empty array
    if l.strip() == "[]":
        return []

    # Initialize array
    vals = []
    clean_l = ""

    # Loop over all line characters
    arr_depth = 0
    for li in l:

        # Identify end of array
        if li == "]":
            arr_depth -= 1

            # Check that there are not too many closing brackets for the opening ones
            if arr_depth < 0:
                raise ValueError("Missing \"[\" for matching the number of \"]\"")

        # If we are within the array, extract the character
        if arr_depth > 0:
            clean_l += li

        # Identify start of array
        if li == "[":
            arr_depth += 1

    # Check that the array is properly closed at the end
    if arr_depth > 0:
        raise ValueError("Missing \"]\" for matching the number of \"[\"")

    # Extract elements in the array
    ls = clean_split(clean_l, ",")

    # Get the value of each element in the array
    for li in ls:
        vals.append(get_val(li.strip()))

    return vals


# In[5]:


def get_val(val):

    # Remove tailing comma
    if val.endswith(","):
        val = val[:-1]

    # Float / Int
    if val.isnumeric():

        if "." in val:
            return float(val)
        else:
            return int(val)

    # Bool
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False

    # String
    if val.startswith("\""):
        return val.split("\"")[1]

    # List
    if val.startswith("["):

        return get_array(val)

    # Try to return a float anyway
    return float(val)


# In[6]:


# Get model architecture
with open(mod + ".py", "r") as F:
    lines = F.read().split("\n")

model_pars = {}
in_pars = False

# Parse script
for l in lines:

    # Identify model parameter block start
    if "model_pars = " in l:
        in_pars = True

    # Identify model parameter block end
    if l.strip() == ")":
        in_pars = False

    if in_pars:
        # Get line
        if "(" in l:
            L = l.split("(")[1].split("#")[0]
        else:
            L = l.strip().split("#")[0]

        key, val = L.split("=")

        v = get_val(val.strip())

        model_pars[key.strip()] = v

model_pars["noise"] = 0.


# In[7]:


# Get data parameters
with open(mod + ".py", "r") as F:
    lines = F.read().split("\n")

data_pars = {}
in_pars = False

# Parse script
for l in lines:

    # Identify model parameter block start
    if "data_pars = " in l:
        in_pars = True

    # Identify model parameter block end
    if l.strip() == ")":
        in_pars = False

    if in_pars:
        # Get line
        if "(" in l:
            L = l.split("(")[1].split("#")[0]
        else:
            L = l.strip().split("#")[0]

        if "=" in L:

            key, val = L.split("=")

            v = get_val(val.strip())

            data_pars[key.strip()] = v


# In[8]:


# Load loss and learning rate
all_lrs = np.load(in_dir + "all_lrs.npy")
all_losses = np.load(in_dir + "all_losses.npy")
all_val_losses = np.load(in_dir + "all_val_losses.npy")

mean_losses = np.mean(all_losses, axis=1)
mean_val_losses = np.mean(all_val_losses, axis=1)

n_chk = all_losses.shape[0]
best_chk = np.argmin(mean_val_losses)
print(best_chk)


# In[9]:


global net
net = model.ConvLSTMEnsemble(**model_pars)
net.eval()
# Load best model
net.load_state_dict(torch.load(in_dir + f"checkpoint_{best_chk+1}_network", map_location=torch.device("cpu")))
net.to(device)


# In[10]:


def load_topspin_spectrum(zfile, d):

    pd = d + "pdata/1/"

    fr = pd + "1r"
    fi = pd + "1i"

    try:
        with zfile.open(fr, "r") as F:
            data = F.read()
            dr = np.frombuffer(data, np.int32).astype(float)

        with zfile.open(fi, "r") as F:
            data = F.read()
            di = np.frombuffer(data, np.int32).astype(float)

        with zfile.open(f"{d}acqus", "r") as F:
            lines = F.read().decode('utf-8').split("\n")

        for l in lines:
            if l.startswith("##$MASR"):
                wr = int(l.split("=")[1].strip())
            if l.startswith("##$TD="):
                TD = int(l.split("=")[1].strip())
            if l.startswith("##$SW_h="):
                SW = float(l.split("=")[1].strip())

        with zfile.open(f"{pd}procs", "r") as F:
            lines = F.read().decode('utf-8').split("\n")

        for l in lines:
            if l.startswith("##$SI="):
                n_pts = int(l.split("=")[1].strip())

            if l.startswith("##$OFFSET="):
                offset = float(l.split("=")[1].strip())

            if l.startswith("##$SF="):
                SF = float(l.split("=")[1].strip())
    except:
        print(f"WARNING: dataset {d} not loaded properly!")
        return None, None, None, None, None

    AQ = TD / (2 * SW)

    hz = offset * SF - np.arange(n_pts) / (2 * AQ * n_pts / TD)

    ppm = hz / SF

    return dr, di, wr, ppm, hz


# In[11]:


def extract_exp_topspin(zfile, d0):

    X = []
    ws = []
    for d in zfile.namelist():
        if d.startswith(d0) and d.endswith("/") and d.count("/") == 2 and d.split("/")[1].isnumeric():
            Xi, _, wr, ppm, hz = load_topspin_spectrum(zfile, d)
            if hz is not None:
                X.append(Xi)
                ws.append(wr)

    sorted_inds = np.argsort(ws)

    sorted_ws = np.array([ws[i] for i in sorted_inds])

    sorted_X = np.array([X[i] for i in sorted_inds])

    return ppm, sorted_ws, sorted_X


# In[12]:


def make_input(X, ws, x_max=0.25):

    # Normalize spectra
    X /= np.sum(X, axis=1)[:, np.newaxis]

    inds = np.argsort(ws)
    X_torch = torch.Tensor(X[inds])
    X_torch = torch.unsqueeze(X_torch, dim=0)
    X_torch = torch.unsqueeze(X_torch, dim=2)

    X_torch /= torch.max(X_torch)
    X_torch *= x_max

    if data_pars["encode_w"]:
        W = torch.Tensor(ws[inds])
        W = torch.unsqueeze(W, dim=0)
        W = torch.unsqueeze(W, dim=2)
        W = torch.unsqueeze(W, dim=3)
        W = W.repeat(1, 1, 1, X_torch.shape[-1])

        if data_pars["norm_wr"]:
            W -= data_pars["mas_w_range"][0]
            W /= data_pars["mas_w_range"][1] - data_pars["mas_w_range"][0]

    X_torch = torch.cat([X_torch, W], dim=2)

    return X[inds], X_torch, ws[inds]


# In[13]:


def plot_exp_vs_pred(ppm, X, y_pred, y_std, y_pred_scale=0.5, x_offset=0.1, xl=[20., -5.], c0=[0., 1., 1.], dc = [0., -1., 0.]):

    # Initialize figure
    try:
        X2 = np.copy(X.numpy())
    except:
        X2 = np.copy(X)

    X2 /= np.max(X2)

    if len(X2.shape) == 1:
        X2 = np.expand_dims(X2, 0)
    n = X2.shape[0]

    dy = (n-1) * x_offset

    fig = plt.figure(figsize=(4,3+dy))
    ax = fig.add_subplot(1,1,1)

    if n == 1:
        colors = [[ci + dci for ci, dci in zip(c0, dc)]]

    else:
        colors = [[ci + (dci * i / (n-1)) for ci, dci in zip(c0, dc)] for i in range(n)]


    try:
        y_pred2 = y_pred.numpy()
        y_std2 = y_std.numpy()
    except:
        y_pred2 = y_pred
        y_std2 = y_std

    factor = np.max(y_pred2) / y_pred_scale
    y_pred2 /= factor
    y_std2 /= factor

    # Plot inputs
    for i, (c, x) in enumerate(zip(colors, X2)):
        h1 = ax.plot(ppm, x + i * x_offset, color=c, linewidth=1)

    # Plot predictions
    h2 = ax.plot(ppm, y_pred2, "r", linewidth=1)
    ax.fill_between(ppm, y_pred2 - y_std2, y_pred2 + y_std2, color="r", alpha=0.3)

    # Update axis
    ax.set_xlim(xl)
    ax.set_ylim(-0.05, 1.05 + ((n-1) * x_offset))
    ax.set_yticks([])
    ax.set_xlabel("Chemical shift [ppm]")

    ax.legend([h1[0], h2[0]], ["MAS spectra", "PIPNet prediction"])

    # Cleanup layout
    fig.tight_layout()

    b = io.BytesIO()
    fig.savefig(b, format="pdf")
    plt.close()

    return fig, b


# In[14]:


def make_file(x):

    if len(x.shape) > 2:
        raise ValueError("Only up to 2D numpy arrays are handled.")

    if len(x.shape) == 1:
        pp = ",".join([f"{a:.8f}" for a in x])
    else:
        pp = ""
        for xi in x:
            pp += ",".join([f"{a:.8f}" for a in xi])
            pp += "\n"

    return pp


# In[17]:


def zip_files(files, names, figs=[], fig_names=[]):

    cur_dt = datetime.datetime.now().strftime("%Y_%m_%d-%Hh%Mm%Ss")
    out_name = f"preds_{cur_dt}.zip"

    with zp.ZipFile(out_name, mode="w", compression=zp.ZIP_DEFLATED) as z:
        for file, name in zip(files, names):
            z.writestr(name + ".csv", file)

        for fig, name in zip(figs, fig_names):
            z.writestr(name + ".pdf", fig.getvalue())
    return out_name


# In[18]:


def make_prediction(file_obj, w0, w1, dw, x0, x1, scale, extend=10):

    # Load spectra
    global zfile
    zfile = zp.ZipFile(file_obj.name)
    basename = zfile.namelist()[0]

    ppm, ws, X = extract_exp_topspin(zfile, basename)

    # Restrict spectral range for performance
    i0 = np.where(ppm > x0)[0][-10]
    i1 = np.where(ppm < x1)[0][9]
    ppm = ppm[i0:i1]
    X = X[:, i0:i1]

    # Pre-process spectra
    X, X_torch, ws = make_input(X, ws)
    sel_ws = np.arange(w0, w1+dw, dw) * 1000

    w_inds = []
    for w in sel_ws:
        w_inds.append(np.argmin(np.abs(ws - w)))

    X_net = X_torch[:, w_inds]
    X_net[:, :, 0] /= torch.max(X_net[:, :, 0]) / scale

    # Perform prediction
    y_pred, y_std, ys = net(X_net)

    y_pred = y_pred.detach().numpy()[0]
    y_std = y_std.detach().numpy()[0]

    ymax = np.max(y_pred)
    y_pred /= ymax / 0.5

    y_std /= ymax / 0.5

    all_figs = []
    fig_names = []
    figs = []
    for i, (yi_pred, yi_std) in enumerate(zip(y_pred, y_std)):
        fig, b = plot_exp_vs_pred(ppm, X[w_inds[model_pars["batch_input"]+i-1]], yi_pred, yi_std, x_offset=0., xl=[x0, x1])
        figs.append(fig)
        all_figs.append(b)
        fig_names.append(f"Prediction_step_{i+1}")
    fig, b = plot_exp_vs_pred(ppm, X[w_inds], y_pred[-1], y_std[-1], x_offset=0., xl=[x0, x1])
    all_figs.append(b)
    fig_names.append(f"Final_prediction")

    zipped_results = zip_files([make_file(ppm), make_file(y_pred), make_file(y_std)], ["ppm", "pred", "err"], all_figs, fig_names)

    return fig, figs[-1], figs, zipped_results

iface = gr.Interface(fn=make_prediction,

                     inputs=[gr.inputs.File(label="ZIP file containing the Topspin dataset"),
                                             gr.inputs.Slider(minimum=20, maximum=100, step=1, default=30, label="Starting MAS rate [kHz]"),
                                             gr.inputs.Slider(minimum=20, maximum=100, step=1, default=80, label="Final MAS rate [kHz]"),
                                             gr.inputs.Slider(minimum=1, maximum=10, step=1, default=2, label="MAS rate step [kHz]"),
                                             gr.inputs.Number(default=20, label="Left chemical shift limit [ppm]"),
                                             gr.inputs.Number(default=-5, label="Right chemical shift limit [ppm]"),
                                             gr.inputs.Slider(minimum=0., maximum=1., step=0.01, default=0.2, label="MAS spectra scale (default: 0.2)")],

                     outputs=[gr.outputs.Image("plot", label="Final prediction with all spectra used"),
                              gr.outputs.Image("plot", label="Final prediction with final spectrum used"),
                              gr.outputs.Carousel("plot", label="Evolution of the prediction with additional spectra"),
                              "file"],
                     title="PIPNet: Prediction of isotropic 1H NMR spectra using deep learning",
                     theme="dark")
#iface.launch(share=True)
iface.launch(enable_queue=True, server_name="0.0.0.0", server_port=8060)


# In[ ]:

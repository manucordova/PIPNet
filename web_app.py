##########################################################################################
###                                                                                    ###
###                                 PIPNet web version                                 ###
###                  Author: Manuel Cordova (manuel.cordova@epfl.ch)                   ###
###                                                                                    ###
##########################################################################################

import flask as flk
import werkzeug.utils as wk

import os
import sys

import numpy as np
import torch

import plotly
import plotly.express as px
import json
import zipfile
import pickle as pk

from pipnet import utils
from pipnet import model

torch.set_num_threads(os.cpu_count())
model_name = "final_model_mixed"
epoch = 250
device = "cuda" if torch.cuda.is_available() else "cpu"
debug = True

# Initialize the Flask app
app = flk.Flask(__name__)
app.config["UPLOAD_FOLDER"] = "./tmp/"
app.config["MAX_CONTENT_PATH"] = int(1e6)

spectra = []

# Load PIPNet model
with open(f"data/1D/{model_name}/data_pars.pk", "rb") as F:
    data_pars = pk.load(F)
with open(f"data/1D/{model_name}/model_pars.pk", "rb") as F:
    model_pars = pk.load(F)

net = model.ConvLSTMEnsemble(**model_pars)
net.load_state_dict(
    torch.load(f"data/1D/{model_name}/epoch_{epoch}_network", map_location=torch.device(device))
)
net.eval()

# Homepage
@app.route("/")
@app.route("/home")
def home():

    return flk.render_template("home.html")

# Predict page
@app.route("/predict")
def predict():
    return flk.render_template("predict.html", spectra=spectra)

# On dataset upload
@app.route("/upload_dataset", methods=["POST"])
def upload_dataset():

    try:

        if flk.request.method == "POST":

            if "dataset" not in flk.request.files:

                resp = flk.jsonify({"message": "No file in the request"})
                resp.status_code = 400
                return resp

            # Save dataset locally            
            f = flk.request.files["dataset"]
            path = wk.secure_filename(f.filename)
            d = app.config["UPLOAD_FOLDER"] + str(np.random.random()).replace(".", "")
            os.mkdir(d)
            dataset = os.path.join(d, path)
            f.save(dataset)
            with zipfile.ZipFile(dataset, "r") as zip_ref:
                zip_ref.extractall(d)
            
            # Extract spectra from directory
            ppm, _, ws, xrs, xis, titles = utils.extract_1d_dataset(dataset.split(".zip")[0] + "/", 1, 1000, return_titles=True)

            print(titles)
            print(ws)

            spectra = []
            for i, (w, xr, xi, title) in enumerate(zip(ws, xrs, xis, titles)):
                fig = px.line(x=ppm, y=xr)
                fig.update_layout({"plot_bgcolor": "rgba(0,0,0,0)"})
                spectra.append(
                    {
                        "x": ppm.tolist(),
                        "yr": xr.tolist(),
                        "yi": xi.tolist(),
                        "plot": json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
                        "wr": int(w / 1000),
                        "num": i+1,
                        "title": title,
                    }
                )

    except:
        resp = flk.jsonify({"message": "Could not upload dataset"})
        resp.status_code = 500
        return resp
    
    return flk.jsonify(
        spectra=spectra
    )
    
# On running prediction
@app.route("/run_prediction", methods=["POST"])
def run_prediction():

    # Read form
    if flk.request.method == "POST":
        data = json.loads(flk.request.form["data"])

        rl = float(flk.request.form["rl"])
        rr = float(flk.request.form["rr"])
        sens = float(flk.request.form["sens"])

    # Load data
    xr = []
    xi = []
    ws = []
    for d in data:
        if d["include"]:
            xr.append(d["yr"])
            xi.append(d["yi"])
            ws.append(float(d["wr"])*1000)
    
    ppm = np.array(d["x"])
    xr = np.array(xr)
    xi = np.array(xi)
    ws = np.array(ws)

    # Select spectral range
    r0 = min(rl, rr)
    r1 = max(rl, rr)
    inds = np.where(np.logical_and(ppm >= r0, ppm <= r1))[0]
    ppm = ppm[inds]
    xr = xr[:, inds]
    xi = xi[:, inds]

    # Perform prediction
    X = utils.prepare_1d_input(xr, ws, data_pars=data_pars, xi=xi, xmax=sens/2.)
    print(X.shape)
    print(torch.sum(X[0, :, 0, :], dim=1))
    print(torch.max(X[0, :, 0, :], dim=1))
    with torch.no_grad():
        y_pred, y_std, ys = net(X)
    y_pred = y_pred[0].numpy()
    y_std = y_std[0].numpy()
    ys = ys[:, 0].numpy()
    
    return flk.jsonify(
        preds={
            "x": ppm.tolist(),
            "specs": xr.tolist(),
            "wrs": (ws/1000.).tolist(),
            "preds": y_pred.tolist(),
            "err": y_std.tolist(),
            "all": ys.tolist(),
        }
    )


@app.route("/examples")
def examples():
    return flk.render_template("examples.html")


@app.route("/about")
def about():
    return flk.render_template("about.html")


@app.route("/contact")
def contact():
    return flk.render_template("contact.html")


if __name__ == "__main__":
    app.run(debug=debug)

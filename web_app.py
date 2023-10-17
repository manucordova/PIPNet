###############################################################################
#                                                                             #
#                             PIPNet web version                              #
#                            Author: Manuel Cordova                           #
#                           Last edited: 2023-10-09                           #
###############################################################################


import flask as flk
import werkzeug.utils as wk

import numpy as np
import os
import torch

import plotly
import plotly.express as px
import json
import zipfile

from pipnet import utils
from pipnet import model

# Initialize pytorch and model name
torch.set_num_threads(os.cpu_count())
model_name_1D = "PIPNet_model"
model_name_2D = "PIPNet2D_model"
device = "cuda" if torch.cuda.is_available() else "cpu"
debug = False

# Initialize the Flask app
app = flk.Flask(__name__)
app.config["UPLOAD_FOLDER"] = "./tmp/"
app.config["MAX_CONTENT_PATH"] = int(1e6)

# Create temporary directory to store spectra
if not os.path.exists("./tmp/"):
    os.mkdir("./tmp/")

spectra = []

# Load 1D PIPNet model
with open(f"trained_models/{model_name_1D}/data_pars.json", "r") as F:
    data_pars_1D = json.load(F)
with open(f"trained_models/{model_name_1D}/model_pars.json", "r") as F:
    model_pars_1D = json.load(F)

net_1D = model.ConvLSTMEnsemble(**model_pars_1D)
net_1D.load_state_dict(
    torch.load(
        f"trained_models/{model_name_1D}/network",
        map_location=torch.device(device)
    )
)
net_1D.eval()

# Load 2D PIPNet model
with open(f"trained_models/{model_name_2D}/data_pars.json", "r") as F:
    data_pars_2D = json.load(F)
with open(f"trained_models/{model_name_2D}/model_pars.json", "r") as F:
    model_pars_2D = json.load(F)

net_2D = model.ConvLSTMEnsemble(**model_pars_2D)
net_2D.load_state_dict(
    torch.load(
        f"trained_models/{model_name_2D}/network",
        map_location=torch.device(device)
    )
)
net_2D.eval()

data = {}


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
            d = app.config["UPLOAD_FOLDER"]
            d += str(np.random.random()).replace(".", "")
            os.mkdir(d)
            dataset = os.path.join(d, path)
            f.save(dataset)
            with zipfile.ZipFile(dataset, "r") as zip_ref:
                zip_ref.extractall(d)
            data_path = dataset.split(".zip")[0] + "/"

            # Get all expnos and procnos
            procnos = {}
            are_2d = {}
            msg = ""

            for expno in os.listdir(data_path):
                d = data_path + expno + "/"
                if expno.isnumeric() and os.path.isdir(d):
                    procnos[expno], this_msg = utils.get_procnos(
                        d,
                        bypass_errors=True
                    )
                    msg += this_msg
                    for procno in procnos[expno]:
                        are_2d[f"{expno}-{procno}"], this_msg = utils.is_2D(
                            d,
                            procno=procno,
                            bypass_errors=True
                        )
                        msg += this_msg

            data["data_path"] = data_path

    except Exception as err:
        resp = flk.jsonify({
                "message": f"Could not upload dataset: {err}"
            })
        resp.status_code = 500
        return resp

    return flk.jsonify(
        params={
            "procnos": procnos,
            "are_2d": are_2d
        }
    )


@app.route("/load_dataset", methods=["POST"])
def load_dataset():

    try:

        if flk.request.method == "POST":

            expnos = list(map(int, flk.request.form["expnos"].split(",")))
            expnos = sorted(expnos)
            procno = flk.request.form["procno"].split("-")[-1]

            ppm, hz, ws, xrs, xis, titles, msg = utils.extract_1d_dataset(
                data["data_path"],
                expnos=expnos,
                procno=procno,
                load_imag=data_pars_1D["encode_imag"],
                bypass_errors=True
            )

            (
                ppm2,
                hz2,
                _,
                _,
                _,
                _,
                msg2
            ) = utils.extract_1d_dataset(
                data["data_path"],
                expnos=expnos,
                procno=procno,
                load_imag=data_pars_1D["encode_imag"],
                use_acqu2s=True,
                bypass_errors=True
            )

            if "ERROR" in msg2:
                ppm2 = []
                hz2 = []
            else:
                ppm2 = ppm2.tolist()
                hz2 = hz2.tolist()

            if "ERROR" in msg:
                resp = flk.jsonify({
                    "message": f"Could not upload dataset.\n{msg}"
                })

            # Normalize spectra for visualization
            norm = np.sum(xrs, axis=1)
            xrs_plot = xrs / norm[:, np.newaxis]
            xis_plot = xis.copy()
            if None in xis_plot:
                xis_plot = np.zeros_like(xrs)
            xis_plot /= norm[:, np.newaxis]

            spectra = []
            for i, (w, xr, xi, title) in enumerate(zip(
                ws,
                xrs_plot,
                xis_plot,
                titles
            )):
                fig = px.line(x=ppm, y=xr)
                fig.update_layout({"plot_bgcolor": "rgba(0,0,0,0)"})
                spectra.append(
                    {
                        "ppm": ppm.tolist(),
                        "ppm2": ppm2,
                        "Hz": hz.tolist(),
                        "Hz2": hz2,
                        "yr": xr.tolist(),
                        "yi": xi.tolist(),
                        "plot": json.dumps(
                            fig,
                            cls=plotly.utils.PlotlyJSONEncoder
                        ),
                        "wr": int(w / 1000),
                        "num": i+1,
                        "title": title,
                    }
                )
            data["spectra"] = spectra

    except Exception as err:
        resp = flk.jsonify({
                "message": f"Could not load dataset: {err}"
            })
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
        acqu2 = flk.request.form["acqu2"] == "true"
        unit = flk.request.form["units"]

    # Load data
    xr = []
    xi = []
    ws = []
    for d in data:
        if d["include"]:
            xr.append(d["yr"])
            xi.append(d["yi"])
            ws.append(float(d["wr"])*1000)

    if unit.endswith("2"):
        ppm = np.array(d["ppm2"])
        hz = np.array(d["Hz2"])

    else:
        ppm = np.array(d["ppm"])
        hz = np.array(d["Hz"])

    xr = np.array(xr)
    xi = np.array(xi)
    ws = np.array(ws)
    x_range = [rl, rr]

    # Perform prediction
    if unit.startswith("ppm"):
        ppm_pred, hz_pred, X, msg = utils.prepare_1d_input(
            xr,
            ppm,
            x_range,
            ws,
            data_pars=data_pars_1D,
            xi=xi,
            x_other=hz,
            xmax=sens/2.
        )
    else:
        hz_pred, ppm_pred, X, msg = utils.prepare_1d_input(
            xr,
            hz,
            x_range,
            ws,
            data_pars=data_pars_1D,
            xi=xi,
            x_other=ppm,
            xmax=sens/2.
        )

    with torch.no_grad():
        y_pred, y_std, ys = net_1D(X)
    y_pred = y_pred[0].numpy()
    y_std = y_std[0].numpy()
    ys = ys[:, 0].numpy()

    if hz_pred is None:
        hz_pred = np.array([])

    return flk.jsonify(
        preds={
            "ppm": ppm_pred.tolist(),
            "Hz": hz_pred.tolist(),
            "specs": X[0, :, 0, :].tolist(),
            "wrs": (ws/1000.).tolist(),
            "preds": y_pred.tolist(),
            "err": y_std.tolist(),
            "all": ys.tolist(),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8008, debug=debug)

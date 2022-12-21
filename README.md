# PIPNet: Pure Isotropic Proton NMR Spectra in Solids using Deep Learning

This is the code for the PIPNet machine learning model described [here](https://doi.org/10.1002/anie.202216607).

## Abstract of the paper

>The resolution of proton solid-state NMR spectra is usually limited by broadening arising from dipolar interactions between spins. Magic-angle spinning alleviates this broadening by inducing coherent averaging. However, even the highest spinning rates experimentally accessible today are not able to completely remove dipolar interactions. Here, we introduce a deep learning approach to determine pure isotropic proton spectra from a two-dimensional set of magic-angle spinning spectra acquired at different spinning rates. Applying the model to 8 organic solids yields high-resolution 1H solid-state NMR spectra with isotropic linewidths in the 50-400 Hz range.

## Installation

The Python package can be installed using pip

    pip install .

Alternatively, a conda environment can be created with the PIPNet package by running the following script:

    ./install_env.sh

## Using the model

To easily use the model, you can deploy the web interface by running:

    python web_app.py

You can then open the link [localhost:8008/](http://localhost:8008/) to access the web version of PIPNet. Please be aware that by default the web app can be accessed from any other machine on the same network by opening the link to the IP address of the machine runing the web app: [<IP_address>:8008/](https://www.youtube.com/watch?v=dQw4w9WgXcQ).

Alternatively, the model can be used throught the Jupyter notebook "ANALYSE-predict_experimental.ipynb" in the "scripts/1D/" directory.

## Training a model

You can train a model by running the script "train_PIPNet.py" in the "scripts/1D" directory. It is advised to have GPU-capable machines to efficiently train the model.

## Analysing a model

Several analysis scripts are available, allowing you to evaluate trained model on generated data, or predicting isotropic spectra from experimental datasets.
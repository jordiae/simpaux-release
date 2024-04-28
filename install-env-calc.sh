#!/usr/bin/env bash

source ~/.bashrc
conda create --name simpaux

conda activate simpaux

conda install numpy scipy opencv matplotlib scikit-learn scikit-image \
    pandas tqdm imageio pytables h5py gitpython tensorboard fasttext nltk
conda install -c conda-forge ipdb
pip install --no-cache-dir torchmeta future
pip install --no-cache-dir git+https://github.com/epistimio/orion.git@develop
pip install pytorch-warmup
pip install einops
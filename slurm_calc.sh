#!/usr/bin/env bash

#SBATCH -p veu-fast
#SBATCH --job-name=simpaux-mini
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --ignore-pbs
#SBATCH --output=logs/simpaux-mini_%j.out
#SBATCH --error=logs/simpaux-mini_%j.err


source ~/.bashrc
conda activate simpaux

mkdir -p data
mkdir -p output
mkdir -p logs

stdbuf -i0 -e0 -o0 python protonet/train.py test --download
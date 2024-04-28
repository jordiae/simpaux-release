#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=20Gb

mkdir -p data
mkdir -p output
mkdir -p logs

export PYTHONUNBUFFERED=1
module purge
module load anaconda/3 python/3.7/cuda/10.1/cudnn/7.6/pytorch/1.5.0

source $CONDA_ACTIVATE
conda activate simpaux
"$CONDA_PREFIX/bin/python" train.py test --download

#stdbuf -i0 -e0 -o0 python train.py test --download
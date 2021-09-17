#! /bin/bash

#SBATCH -A snic2018-1-13
#SBATCH -N 1

# ${model} ${train_years} ${train_years_offset} ${load_lazily} must be passed in by sbatch
set -u
source activate largescale-ML

python train_network_puma_plasim_100epochlargemem.py ${model} ${train_years} ${train_years_offset} ${load_lazily}

python predict_network_puma_plasim.py ${model} ${train_years} ${train_years_offset} False

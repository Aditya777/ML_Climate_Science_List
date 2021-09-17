#! /bin/bash

#SBATCH -A snic2018-1-13
#SBATCH -N 1

# ${model} ${train_years} ${train_years_offset} ${load_lazily} must be passed in by sbatch
set -u
source activate largescale-ML

python train_network_clim_puma_plasim_localSSD.py ${model} ${train_years} ${train_years_offset} ${load_lazily} 



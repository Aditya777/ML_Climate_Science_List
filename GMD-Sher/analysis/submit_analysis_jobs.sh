#!/bin/bash


#models="pumat21_noseas
#pumat21
#pumat42
#pumat42_regridt21
#plasimt21
#plasimt42
#plasimt42_regridt21"

models="pumat21_noseas
pumat21
pumat42_regridt21
plasimt21
plasimt42_regridt21"


for model in ${models}; do

    sbatch <<EOF
#! /bin/bash

#SBATCH -A snic2018-1-13
#SBATCH -N 1
#SBATCH --time=24:30:00

set -u
source activate gcm_complexity_analysis

python analyze_predictions.py ${model}
EOF

done
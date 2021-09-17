#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 16:09:35 2018

@author: sebastian
"""

import os
            # model train_years train_years_offset



# create a list of all runs we want to make
models = ['pumat21', 'pumat21_noseas', 'plasimt21','pumat42_regridt21' ,'plasimt42_regridt21']
train_years_list = [50,100]
train_years_offset_list = [0]

batches = []
for model in models:
    for train_years in train_years_list:
        for train_years_offset in train_years_offset_list:
            if train_years > 100:
                time = '3-00:00:00'
                load_lazily = "True"
            else:
                load_lazily = "False"
                time ='06:00:00'

            batches.append((model, train_years, train_years_offset,load_lazily,time))

print(batches)


# now submit every run as a single job
for model,train_years, train_years_offset, load_lazily, runtime in batches:

    runstring='sbatch --time='+runtime+' --export=model='+model+',train_years='+str(train_years)+',train_years_offset='+str(train_years_offset)+',load_lazily='+load_lazily+   ' tetralith_job.sh'
    print(runstring)

    os.system(runstring)

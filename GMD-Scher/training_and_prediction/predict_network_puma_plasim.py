#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:09:29 2018

use the trained network to make predictions on the test data

input files: architecture_*.json, weights_*.h5, model data

output: .nc files with predictions, one file for each lead time
the predictions are rescaled to absolute values with the normalization
weights used during training


@author: sebastian
"""


import json
import os
import sys

import numpy as np
import xarray as xr


from keras.models import model_from_json

from keras import backend as K

# set max number of CPUs
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=30,
inter_op_parallelism_threads=30)))

from dask.diagnostics import ProgressBar
ProgressBar().register()

#read in arguments from commandline
modelname=sys.argv[1]
train_years = int(sys.argv[2])
train_years_offset = int(sys.argv[3])
load_data_lazily = sys.argv[4] == "True"



ifile='/proj/bolinc/users/x_sebsc/gcm_complexity_machinelearning/models/preprocessed/'+modelname+'reordered.merged.nc'



# where to save the predictions
outdir = '/proj/bolinc/users/x_sebsc/gcm_complexity_machinelearning/predictions/'
os.system('mkdir -p ' + outdir)

test_years=30  # note that when changing this here (default is 30), the train data returned by prepare_data will be wrong

lead_time = 1

num_epochs = 100


days_per_year_per_model={'pumat21':360,'pumat31':360,'pumat42':360,'pumat21_noseas':360,
                         'plasimt21':365,'plasimt31':365,'plasimt42':365,
                         'pumat42_regridt21':360,
                         'plasimt42_regridt21':365} # noteL we are ignoring leap years in plasim here,
                                                                            

days_per_year = days_per_year_per_model[modelname]

N_test = days_per_year * test_years
N_train = days_per_year * train_years
N_train_offset = days_per_year * train_years_offset



param_string = modelname +'_'+ '_'.join([str(e) for e in (train_years,train_years_offset,num_epochs,lead_time)])


def prepare_data(x,lead_time):
    ''' split up data in predictor and predictant set by shifting
     it according to the given lead time, and then split up
     into train, developement and test set'''
    if lead_time == 0:
        X = x
        y = X[:]
    else:

        X = x[:-lead_time]
        y = x[lead_time:]

    X_train = X[N_test+train_years_offset:N_test+train_years_offset+N_train]
    y_train = y[N_test+train_years_offset:N_test+train_years_offset+N_train]

    X_test = X[:N_test]
    y_test = y[:N_test]

    return X_train,y_train, X_test, y_test



data = xr.open_dataarray(ifile, chunks={'time':3600})  # we have to define chunks,
# then the data is opened as dask -array (out of core)

# note that the time-variable in the input file is confusing: it contains the
# day of the year of the simulation, thus it is repeating all the time
# (it loops from 1 to 360 and then jumps back to one)


# convert to 32 bit
data = data.astype('float32')


# check that we have enough data for the specifications
if N_train + N_test > data.shape[0]:
    raise Exception('not enough timesteps in input file!')


X_train,y_train, X_test, y_test = prepare_data(data,lead_time)

# save the "truth" before normalizing, we will use this when evaluating the forecasts
X_test.to_netcdf(outdir+'/truth_'+param_string+'.nc')

# load the normalization weights to normalize the test data in the same way
# as the training data
norm_mean = xr.open_dataarray('data/norm_mean_'+param_string+'.nc')
norm_std = xr.open_dataarray('data/norm_std_'+param_string+'.nc')

X_test = (X_test - norm_mean) / norm_std
y_test = (y_test - norm_mean) / norm_std


if not load_data_lazily:
    print('load test data into memory')
    X_test.load()
    y_test.load()
    
    
## now load the trained network
weight_file = 'data/weights_'+param_string+'.h5'
architecture_file = 'data/modellayout_'+param_string+'.json'

model = model_from_json(json.load(open(architecture_file,'r')))
# load the weights form the training
model.load_weights(weight_file)



# now make multi-step forecasts
max_leadtime=14
lead_times = np.arange(1,max_leadtime+1)

res = []
y_test_predicted = X_test[:-max_leadtime] # initialize, we have to discard the last max_leadtime initial fields



for lead_time in lead_times:
    
    print('lead_time',lead_time)
    
    # this is now the 1-day forecast the network was trained on
    # now make a forecast for the given lead time. as we hvae trained on daily forecasts
    # and lead_time are full days, we have to make lead_time forecasts

    y_test_predicted = model.predict(y_test_predicted)

    # this is now a numpy array, convert to xarray
    
    # for this we ned to get teh right coordinates, namely the same ones as for the target data
    if lead_time < max_leadtime:
        y_test = X_test[lead_time:-(max_leadtime-lead_time)]
    else:
        y_test = X_test[lead_time:]
            

    assert(y_test.shape == y_test_predicted.shape)
    
    y_test_predicted = xr.DataArray(data=y_test_predicted, dims=y_test.dims, coords = y_test.coords)
    
    # rescale the precitions with the normalization weights
    y_test_predicted_scaled = (y_test_predicted * norm_std ) + norm_mean
    
    
    y_test_predicted_scaled.to_dataset(name='data').to_netcdf(outdir+'/network_prediction_'+param_string+'_leadtime'+str(lead_time)+'.nc',encoding={'data':{'zlib':True,'complevel':4}})
        


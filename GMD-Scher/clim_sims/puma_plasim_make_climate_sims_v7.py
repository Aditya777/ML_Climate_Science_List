
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""



@author: sebastian
"""
import matplotlib
matplotlib.use('agg')

import json
import sys
import os
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns

import keras
from keras.models import model_from_json
from pylab import plt
import tensorflow as tf

from dask.diagnostics import ProgressBar
ProgressBar().register()

# limit number of CPUs used
config = tf.ConfigProto(intra_op_parallelism_threads=32, inter_op_parallelism_threads=32,
                        allow_soft_placement=True)
session = tf.Session(config=config)
keras.backend.set_session(session)

plt.rcParams['savefig.bbox'] = 'tight'


# #read in arguments from commandline
modelname = sys.argv[1]
train_years = int(sys.argv[2])

train_years_offset = 0
load_data_lazily = True

#ifile='/proj/bolinc/users/x_sebsc/gcm_complexity_machinelearning/models/preprocessed/'+modelname+'reordered.merged.nc'
ifile='/climstorage/sebastian/gcm_complexity_machinelearning/modelruns/preprocessed/'+modelname+'reordered.merged.nc'
network_dir='./data/'

os.system('mkdir -p plots')

# the level and variable dimension is the same, it is ua100hPa, ua200hPa,....ua10000hPa, va100hPa....]
varnames = ['ua','va','ta','zg']
keys = [ varname+str(lev) for varname in varnames for lev in range (100,1001,100)]
varname_to_levidx = { key:levidx for key,levidx in zip(keys,range(len(keys)))  }

test_years = 30  # note that when changing this here (default is 30), the train data returned by prepare_data will be wrong

lead_time = 1

num_epochs = 100

days_per_year_per_model = {'pumat21': 360, 'pumat31': 360, 'pumat42': 360, 'pumat21_noseas': 360,
                           'plasimt21': 365, 'plasimt31': 365,
                           'plasimt42': 365}  # noteL we are ignoring leap years in plasim here,

days_per_year = days_per_year_per_model[modelname]

N_test = days_per_year * test_years
N_train = days_per_year * train_years
N_train_offset = days_per_year * train_years_offset

param_string = 'climnet'+modelname + '_' + '_'.join([str(e) for e in (train_years, train_years_offset, num_epochs, lead_time)])


def prepare_data(x, lead_time):
    ''' split up data in predictor and predictant set by shifting
     it according to the given lead time, and then split up
     into train, developement and test set'''
    if lead_time == 0:
        X = x
        y = X[:]
    else:

        X = x[:-lead_time]
        y = x[lead_time:]

    X_train = X[N_test + train_years_offset:N_test + train_years_offset + N_train]
    y_train = y[N_test + train_years_offset:N_test + train_years_offset + N_train]

    X_test = X[:N_test]
    y_test = y[:N_test]

    return X_train, y_train, X_test, y_test


data = xr.open_dataarray(ifile, chunks={'time': 1})  # we have to define chunks,
# then the data is opened as dask -array (out of core)

# note that the time-variable in the input file is confusing: it contains the
# day of the year of the simulation, thus it is repeating all the time
# (it loops from 1 to 360 and then jumps back to one)


# convert to 32 bit
data = data.astype('float32')

# add dayofyear as additionaly layer. dayofyear is already in data.time (but here it is 1-dimensional),
# we need to expand it to lat lon. This can be done with broadcasting it onto the data. if we broadcast against
# the whole data, then it is also multiplied for each level, therefore we first select the first level (it doesnt
# matter which level we select). broadcast returns a tuple, in this case with 2 entries which are both the same,
# so we just use the first one
doy = xr.broadcast(data.time,data.isel(lev=0))[0]
# add an empty level dimension to doy
doy = doy.expand_dims(dim='lev',axis=-1)
# we have to add a value to the level dimension, which one does not mater
doy['lev'] = [-1]
data = xr.concat([data,doy], dim='lev')

# check that we have enough data for the specifications
if N_train + N_test > data.shape[0]:
    raise Exception('not enough timesteps in input file!')

X_train, y_train, X_test, y_test = prepare_data(data, lead_time)



# load the normalization weights to normalize the test data in the same way
# as the training data
norm_mean = xr.open_dataarray(network_dir+'/norm_mean_' + param_string + '.nc')
norm_std = xr.open_dataarray(network_dir+'/norm_std_' + param_string + '.nc')

X_test = (X_test - norm_mean) / norm_std
y_test = (y_test - norm_mean) / norm_std

if not load_data_lazily:
    print('load test data into memory')
    X_test.load()
    y_test.load()

## now load the trained network
weight_file = network_dir+'/weights_' + param_string + '.h5'
architecture_file = network_dir+'/modellayout_' + param_string + '.json'

model = model_from_json(json.load(open(architecture_file, 'r')))
# load the weights form the training
model.load_weights(weight_file)

N_years = 30

# the number of forecast steps to make depends on the lead_time

N_steps = int(np.round(N_years * 365 / lead_time))
# now we need to select an initial condition, and then repeatedly 
# apply the prediction    


# for the timeline plot it is important to have an initial condiditon form the beginning
# of the test set (because we plot the evolution of model and network next to each other
# here we ues only 2 randomly smapled initial states due to memory leak problems in xarray.
# to get more simply rerun the script (close and restart python console first)
for count_init, i_init in enumerate(np.random.randint(0,360, size=2)):


    
    initial_state = X_test[i_init]    
    # load the initial state into memory
    initial_state.load()
    clim_run = np.zeros(shape=[N_steps+1] + list(initial_state.shape))
    
    clim_run = xr.DataArray(data = clim_run, coords={'time':np.arange(N_steps+1),
                                                     'lat':X_test.lat,
                                                     'lon':X_test.lon,
                                                     'lev':X_test.lev},dims=X_test.dims
        )
    
    clim_run[0] = initial_state
    # get the dayofyear of the initial state, it is the data on the last level. it is the same for
    # each lat and lon, so we can simply use the first elements of lat and lon
    dayofyear_init_normed = initial_state[0,0,-1]
    doy_norm_mean = norm_mean[-1].values
    doy_norm_std = norm_std[-1].values

    current_doy_normed = dayofyear_init_normed
    current_doy = (current_doy_normed * doy_norm_std) + doy_norm_mean

    for i in range(N_steps):
        print(i,N_steps)
        # we have to add an (empty) time dimenstion
        current_state = np.expand_dims(clim_run[i],0)
        # replace the predicted doy by the real one
        current_state[:,:,:,-1] = current_doy_normed

        prediction = model.predict(current_state)
        clim_run[i+1] = np.squeeze(prediction)
    
        #update the dayofyear
        if current_doy == days_per_year -1:
            current_doy = 0
        else:
            current_doy = current_doy + 1

        current_doy_normed = (current_doy - doy_norm_mean) / doy_norm_std

    
    # compute statistics of the prediction
    climmean = clim_run.mean('time')
    
    var='zg500'

    plt.figure(figsize=(7,3))
    climmean.isel(lev=varname_to_levidx[var]).plot.contourf(
            levels=np.arange(-2,2.01,0.1)
            )
    plt.title('network 500hPa height  mean')
    plt.savefig('plots/network_climmean'+str(lead_time)+param_string+str(count_init)+'.pdf')
    
    
    # plot the climatology of the model
    climmean_puma = X_test.mean('time')
    
    plt.figure(figsize=(7,3))
    climmean_puma.isel(lev=varname_to_levidx[var]).plot.contourf(
            levels=np.arange(-2,2.01,0.1)
            )
    plt.title('gcm 500hPa height mean')
    plt.savefig('plots/model_climmean'+str(lead_time)+param_string+str(count_init)+'.pdf')
    
    
    
    # plot one gridpoint
    var='ta800'
    
    plt.figure(figsize = (8,4))
    plt.plot(clim_run[:N_steps,10,10,varname_to_levidx[var]], label='network')
    plt.plot(X_test[i_init:i_init+N_steps*lead_time:lead_time,10,10,varname_to_levidx[var]], label='puma')
    plt.legend()
    sns.despine()
    plt.ylabel(var +' normalized')
    plt.xlabel('days')
    plt.savefig('plots/timeevolution_one_gridpoint_climatemode'+str(lead_time)+param_string+'_init'+str(count_init)+'.pdf')

    # plot only first years
    var = 'ta800'

    plt.figure(figsize=(8, 4))
    plt.plot(clim_run[:360*2, 10, 10, varname_to_levidx[var]], label='network')
    plt.plot(X_test[i_init:i_init + 360*2 * lead_time:lead_time, 10, 10, varname_to_levidx[var]], label='puma')
    plt.legend()
    sns.despine()
    plt.ylabel(var + ' normalized')
    plt.xlabel('days')
    plt.savefig('plots/timeevolution_one_gridpoint_climatemode_firstyears' + str(lead_time) + param_string + '_init' + str(count_init) + '.pdf')
    
    
    
#
# # make single plots
#
plt.ioff()
data_network    = clim_run
plot_kwargs = {'extend': 'both'}
for ii, i_train in enumerate(range(i_init, i_init + 360*2)):
    plt.figure(figsize=(13, 8))
    plt.subplot(221)
    plt.title('z 500hPa')

    levels = levels = np.array([-10, -8, -6, -4, -2, 2, 4, 6, 8, 10]) * 0.1 * 1.9

    var = 'zg500'
    x = X_test.isel(time=i_train).isel(lev=varname_to_levidx[var])

    x.plot.contourf(cmap=plt.cm.plasma_r, levels=levels, add_labels=False, **plot_kwargs)

    plt.ylabel('latitude')

    plt.subplot(222)
    plt.title('u 300hPa')
    var = 'ua300'
    x = X_test.isel(time=i_train).isel(lev=varname_to_levidx[var])

    x.plot.contourf(cmap=plt.cm.RdBu_r, levels=levels, add_labels=False, **plot_kwargs)

    plt.ylabel('latitude')

    # plot network climate
    plt.subplot(223)
    plt.title('z 500hPa network')
    var = 'zg500'
    x = data_network.isel(time=ii).isel(lev=varname_to_levidx[var])

    x.plot.contourf(cmap=plt.cm.plasma_r, levels=levels, add_labels=False, **plot_kwargs)
    plt.xlabel('longitude')
    plt.ylabel('latitude')

    plt.subplot(224)
    plt.title('u 300hPa network')
    var = 'ua300'
    x = data_network.isel(time=ii).isel(lev=varname_to_levidx[var])

    x.plot.contourf(cmap=plt.cm.RdBu_r, levels=levels, add_labels=False, **plot_kwargs)
    plt.xlabel('longitude')
    plt.ylabel('latitude')

    plt.suptitle('day ' + str(ii))

    plt.savefig('plots/'+param_string+'z_plus_u_' + str(ii).zfill(3) + '.png', dpi=300)

    plt.close('all')



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: sebastian
"""
import matplotlib
matplotlib.use('agg')


import numpy as np
import xarray as xr
from pylab import plt

from dask.diagnostics import ProgressBar

ProgressBar().register()



for modelname in ['pumat21_noseas','pumat21', 'pumat42','plasimt21', 'plasimt42']:

    
    ifile = '/climstorage/sebastian/gcm_complexity_machinelearning/modelruns/preprocessed/' + modelname + 'reordered.merged.nc'
    
    
    varnames = ['ua', 'va', 'ta', 'zg']
    keys = [varname + str(lev) for varname in varnames for lev in range(100, 1001, 100)]
    varname_to_levidx = {key: levidx for key, levidx in zip(keys, range(len(keys)))}
    
    
    
    # we need the original training data to compute the climatology for the anomaly
    # correlation coefficient (ACC)
    print('open inputdata')
    full_data = xr.open_dataset(ifile, decode_times=False, chunks={'time': 360})
    
    # conver to datarray (the data is named "ua" for all variables)
    full_data = full_data['ua']
    
    
    areamean = full_data.mean(('lat','lon'))
    areamean.load()
    window = 360*10  # days
    resampled = areamean.rolling(time=window, min_periods=1, center=True).mean()[::window]
    print('computing climatology')
    
    
    
    x_values = np.arange(len(resampled))*window / 365 # years
    
    
    for var in ('zg500', 'ta800'):
    
        plt.figure()
        sub = resampled.isel(lev=varname_to_levidx[var])
        plt.plot(x_values,sub.values)
        plt.ylabel(var+' '+str(window)+' daymean')
        plt.xlabel('year')
        plt.tight_layout()
        plt.savefig('model_drift_'+modelname+'_'+var+'.pdf')
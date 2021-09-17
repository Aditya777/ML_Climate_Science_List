#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

make mapplots of zg500 and ta80 error of network forecasts, based
on the .nc output from analyze_predictions_mapplots.py

@author: sebastian
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import xarray as xr
from pylab import plt

from dask.diagnostics import ProgressBar

ProgressBar().register()



varnames = ['ua', 'va', 'ta', 'zg']
keys = [varname + str(lev) for varname in varnames for lev in range(100, 1001, 100)]
varname_to_levidx = {key: levidx for key, levidx in zip(keys, range(len(keys)))}

num_epochs = 100
lead_time_training = 1


train_years = 100
train_years_offset = 0
models = ['pumat21_noseas','pumat21', 'pumat42','plasimt21', 'plasimt42']

for modelname in models:


    param_string = modelname + '_' + '_'.join(
        [str(e) for e in (train_years, train_years_offset, num_epochs, lead_time_training)])


    for lead_time in [1,6]:
        mse = xr.open_dataarray('mse_timmean_3d_'+param_string + '_day'+str(lead_time)+'.nc')
        #mse_normed =xr.open_dataarray('mse_normed_timmean_3d_'+param_string + '_day'+str(lead_time)+'.nc')

        rmse = np.sqrt(mse)

        var='zg500'
        vmin, vmax=0,120
        levels = np.arange(vmin,vmax,10)
        plt.figure(figsize=(8,3))
        rmse.isel(lev=varname_to_levidx[var]).plot.contourf(levels=levels,cmap=plt.cm.gist_heat_r,
                                                            cbar_kwargs={'label':'RMSE '+var+' [m]'})
        plt.title(modelname+' leadday '+str(lead_time))
        plt.savefig('rmse_2d_' + modelname + '_' + var + '_leadtime' + str(lead_time) + '.pdf', bbox_inches='tight')

        var='ta800'
        vmin,vmax=0,8
        plt.figure(figsize=(8,3))
        rmse.isel(lev=varname_to_levidx[var]).plot.contourf(vmax=vmax,vmin=vmin,cmap=plt.cm.gist_heat_r,
                                                            cbar_kwargs={'label':'RMSE '+var+' [K]'})
        plt.title(modelname+' leadday '+str(lead_time))
        plt.savefig('rmse_2d_'+modelname+'_'+var+'_leadtime'+str(lead_time)+'.pdf', bbox_inches='tight')



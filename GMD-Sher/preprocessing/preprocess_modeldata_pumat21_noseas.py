#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


reads in postprocessed PUMA/PLASIM output, and reorders the data
so that it is more convenient for the training, and merges all files into one
single file. the first 30 years are discarded (spinup)

@author: sebastian
"""

import numpy as np
import pandas as pd
import xarray as xr



from dask.diagnostics import ProgressBar
ProgressBar().register()


model='pumat21_noseas'
basepath='/climstorage/sebastian/gcm_complexity_machinelearning/modelruns/'
spinupyears = 30 # we discard the first 30 years
years = range(1+spinupyears,1030)
ifiles=[basepath+'/'+model+'/MOST.'+str(year).zfill(3)+'_plevel.nc' for year in years]

for ifile in ifiles:
    
    print('open inputdata', ifile)
    data = xr.open_dataset(ifile, chunks={'time':1},decode_times=False)
    
    # we have different variables, each dimension (time,lev,lat,lon)
    # we want to stack all variables, so that we have dimension (time,lat,lon,channel)
    # where channel is lev1,lev2... of variable 1, lev1,lev2,... of variable 2 and son on
    
    print(' stack data')
    varnames = ['ua','va','ta','zg']
    # stack along level dimension
    stacked = xr.concat((data[varname] for varname in varnames), dim='lev')
    
    
    print('reorder data')
    # now reorder so that lev is last dimension
    x = stacked.transpose('time','lat','lon','lev')
    
    
    
    ofile = ifile+'.reordered.nc'
    #
    x.to_netcdf(ofile)
    
    data.close()
    x.close()
    
    
    
## merge all files into one long one
ifiles=[basepath+'/'+model+'/MOST.'+str(year).zfill(3)+'_plevel.nc.reordered.nc' for year in years]

data = xr.open_mfdataset(ifiles, decode_times=False, concat_dim='time')

data.to_netcdf(basepath+'/'+model+'/'+model+'reordered.merged.nc')




    
    
    
    

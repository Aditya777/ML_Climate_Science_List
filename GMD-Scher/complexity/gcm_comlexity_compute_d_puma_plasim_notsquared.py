#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

compute local dimension (d) for puma and plasim runs

@author: sebastian
"""


import numpy as np
import pandas as pd
import xarray as xr
import dask

from multiprocessing.pool import ThreadPool
from dask.diagnostics import ProgressBar


import sklearn.metrics


dask.set_options(pool = ThreadPool(32))
pbar = ProgressBar()
pbar.register()


def compute_d(data):
    '''
    input: 3d datarray (time,lat,lon)
    '''
    # flatten latlon into one dimension

    flat = data.stack(xy=('lat','lon'))

    # convert to numpy array
    flat = flat.values
    # normalize
    flat = flat / flat.mean()

    assert(len(flat.shape)==2)
    # compute pairwise euclidiean distances.
    distances = sklearn.metrics.pairwise.euclidean_distances(flat)

    m = distances.shape[0]
    perc=98
    
    sigma = []

    # loop through timesteps
    for i in np.arange(m):
        # extract distances of the current state to all other states
        dists = distances[i]
        #remove the diagonal zero
        dists = np.delete(dists,i)

        # normalize distances with euclidain norm
        dists = dists / np.linalg.norm(dists)

        gx = - np.log(dists)

        thresh=np.percentile(gx,perc)

        my = gx[gx>thresh] - thresh
        asigma = np.mean(my)

        sigma.append(asigma)

    sigma = np.array(sigma)

    return sigma






# the level and variable dimension is the same, it is ua100hPa, ua200hPa,....ua10000hPa, va100hPa....]
varnames = ['ua','va','ta','zg']
keys = [ varname+str(lev) for varname in varnames for lev in range (100,1001,100)]
varname_to_levidx = { key:levidx for key,levidx in zip(keys,range(len(keys)))  }

# loop throug all models and compute d for every timestep, save the time-mean
# of d


modelnames = ['plasimt21','plasimt42','pumat21_noseas','pumat21','pumat42',
             'pumat42_regridt21','plasimt42_regridt21']
res = []
for modelname in modelnames:
    print(modelname)
    ifile = '/climstorage/sebastian/gcm_complexity_machinelearning/modelruns/preprocessed/' + modelname + 'reordered.merged.nc'
    data_allvars = xr.open_mfdataset(ifile, decode_times=False)

    # loop over variables
    res_model = []
    varnames = []
    for var in varname_to_levidx.keys():
        print(var)
        data = data_allvars.isel(lev=varname_to_levidx[var])
    


        # how many years to use? in era-interim we have 38 available, t
        # therefore we also use 38 here, and we use 365 day calendar for all models
        # so that we have exactly the same amount of timesteps (except for leap years which are so few that we can ignore them)
        N_years = 38
        data = data.isel(time=slice(None,N_years*365))
    
        # convert to dataset to dataarray (we only have one varialbe - called ua -because everything is stacked)
        data = data['ua']
        # load data to pseed up things
        data.load()
        

        sigma = compute_d(data)
    
        df = pd.DataFrame( {'sigma':sigma}, index=data.time)
        df['d'] = 1/df['sigma']
    
        # compute timemean of d
    
        d_timmean = df['d'].mean()
    
    
        df = pd.DataFrame({'model':modelname,
                           'var':var,
                           'd':d_timmean}
                           ,index=[0])

        res.append(df)
        


# combine everything into one dataframe
res = pd.concat(res)

res.to_csv('model_vs_d_spread_plasim_puma_multivar_notsquared.csv')

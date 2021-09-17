#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 09:21:43 2018

@author: sebastian
"""

import os

import numpy as np
import pandas as pd
import xarray as xr
import dask

from multiprocessing.pool import ThreadPool
from dask.diagnostics import ProgressBar


import sklearn.metrics


dask.set_options(pool = ThreadPool(20))
pbar = ProgressBar()
pbar.register()


def compute_d(data):
    # flatten latlon into one dimension

    flat = data.stack(xy=('lat','lon'))

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
        print(i, m)
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

def standardize_dataset(ds):
    
    ''' change dimension names to standard names (lat,lon,lev,time),

    return: None
    '''
    pairs = {'latitude':'lat',
             'longitude':'lon',
             'level':'lev'}
    for key in pairs.keys():
        if key in ds.dims.keys():
            ds = ds.rename({key:pairs[key]})

    # extract variable name
    var_keys = ds.data_vars.keys()
    assert ( len(var_keys) ==1)
    for e in var_keys:
        name = e
    ds.attrs['varname'] = name

    
    return ds

def read_einterim(var,startdate,enddate, typ='surface_analysis'):
    
    startdate=pd.to_datetime(startdate)
    enddate = pd.to_datetime(enddate)
    basepath = '/climstorage/obs/ERAInt/6hourly/'
    
    path = basepath+typ+'/'+var
    
    years = range(startdate.year, enddate.year+1)
    if typ=='surface_analysis':
        files = [path+'/'+var+'_'+str(year)+'.nc' for year in years]
    elif typ=='plev':
        months  = range(1,12+1)
        files = [path+'/'+var+'_'+str(year)+'_'+str(month).zfill(2)+'.nc' for year in years for month in months]
        print(files)
    
    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f)
            
    
    ds = xr.open_mfdataset(files, autoclose=True)
    # subset time
    ds = ds.sel(time=slice(startdate,enddate))        
    ds = standardize_dataset(ds)
    return ds[var]



res_model = []
varnames = []
for var in ('z','u','v','t'):
    data_all  =  read_einterim(var,'19790101', '20161231', typ='plev')
    # the data is 6-hourly, compute daily mean
    data_all = data_all.resample(time='1d').mean('time')
    # load data
    data_all.load()
    for level in (100,200,300,400,500,600,700,850,1000):    
        print(var,level)
        data = data_all.sel(lev=level)
        
        
        
        # normalize 
        data = data / data.mean()
        
        
        sigma = compute_d(data)
        
        df = pd.DataFrame( {'sigma':sigma}, index=data.time)
        df['d'] = 1/df['sigma']
        
        
        # compute timemean of d
        
        d_timmean = df['d'].mean()
        
        res_model.append(d_timmean)
        varname_mapping = {'z':'zg','u':'ua','v':'va','t':'ta'}
        varnames.append(varname_mapping[var]+str(level))
        
df = pd.DataFrame({'model':'era-interim', **{varname:d for varname,d in zip(varnames,res_model)}},
                                          index=[0])
df.to_csv('model_vs_d_einterim_multivar_notsquared.csv')


#only neccessary wehen run interactively in ipython:
# if everything went fine, quite the interpreter to free up memory
os._exit(0)



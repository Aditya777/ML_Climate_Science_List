#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 17:36:45 2018


analyze the predictions mad by the neural networks


note: this script does not work on tetralith ( some problem with xarray when computing the running climatology,
xarray needs the bottleneck library for computing rolling means with dask, but when I installed bottlenecl on triolith
pylab did not work anymore.....

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


def sellonlatbox(data, west, north, east, south, allow_single_point=False, check_bounds = False):
    '''
    select a rectangular box.
    if west==east and north==south, then a single point will be returned
    if allow_single_point=True, otherwise an error will be raised
    '''
    if north == south and west == east:
        if allow_single_point:
            # in this case we return a single point
            return sellonlatpoint(data, west, north)
        else:
            raise ValueError('''the requsted area is a single point 
                             (north==south and east==west)''')

    if north < south:
        raise ValueError('north ({}) is smaller than south ({})'.
                         format(north, south))

    if north < south:
        raise ValueError('west ({}) is larger than east ({})'.
                         format(west, east))

    if check_bounds:
        # check wehther the requested point is in the domain
        lonrange = (float(data.lon.min()), float(data.lon.max()))
        if (west < lonrange[0] or west > lonrange[1] or
                east < lonrange[0] or east > lonrange[1]):
            raise ValueError('''the requested lon extend {} to {} is outside 
                             the range of the
                             dataset {}'''.format(west, east, lonrange))

        latrange = (float(data.lat.min()), float(data.lat.max()))
        if (north < latrange[0] or north > latrange[1] or
                south < latrange[0] or south > latrange[1]):
            raise ValueError('''the requested lat extend {} to {} is outside 
                             the range of the 
                             dataset {}'''.format(north, south, latrange))

        # depending on whether lat is increasing or decreasing (which is
    # different for different datasets) the indexing has to be done in
    # a different way
    if _lat_is_increasing(data):
        indexers = {'lon': slice(west, east), 'lat': slice(south, north)}
    else:
        indexers = {'lon': slice(west, east), 'lat': slice(north, south)}

    sub = data.sel(**indexers)

    return sub


def _lat_is_increasing(data):
    if data.lat[1] > data.lat[0]:
        return True
    else:
        return False



def acc_score(pred, truth, clim):
    '''timestepwise anomaly correlation coefficient, averaged over time
       '''

    assert (pred.shape == truth.shape)

    assert (np.array_equal(pred.coords, truth.coords))
    # clim is an xarray with time-dimensino "dayofyear"
    # first we have to expand this to our data, corresponding to the right
    # tim eof the year
    clim_expanded = [clim.sel(dayofyear=i).values for i in truth.dayofyear.values]

    pred_anom = pred - clim_expanded
    truth_anom = truth - clim_expanded
    # convert to numpy array
    pred_anom = pred_anom.values
    truth_anom = truth_anom.values
    return np.mean([np.corrcoef(pred_anom[i].flatten(), truth_anom[i].flatten())[0, 1] for i in range(len(pred))])


modelname = 'plasimt21'

num_epochs = 10
lead_time_training = 1

#ifile = '/climstorage/sebastian/gcm_complexity_machinelearning/modelruns/'+modelname+'/' + modelname + 'reordered.merged.nc'
ifile='/proj/bolinc/users/x_sebsc/gcm_complexity_machinelearning/models/preprocessed/'+modelname+'reordered.merged.nc'
#prediction_dir = '/climstorage/sebastian/gcm_complexity_machinelearning/predictions/'
prediction_dir = '/proj/bolinc/users/x_sebsc/gcm_complexity_machinelearning/predictions/'



varnames = ['ua', 'va', 'ta', 'zg']
keys = [varname + str(lev) for varname in varnames for lev in range(100, 1001, 100)]
varname_to_levidx = {key: levidx for key, levidx in zip(keys, range(len(keys)))}



# we need the original training data to compute the climatology for the anomaly
# correlation coefficient (ACC)
print('open inputdata')
full_data = xr.open_dataset(ifile, decode_times=False, chunks={'time': 360})

# conver to datarray (the data is named "ua" for all variables)
full_data = full_data['ua']

# data to be used for computation of climatology, there we use the same regardless of
# training length in order to be consistent. We use the first 30 years of the training data
# remmeber that the training data starts after 30 years of the data (the first
# 30 years is the test data)
days_per_year_per_model = {'pumat21': 360, 'pumat31': 360, 'pumat42': 360,
                           'plasimt21': 365, 'plasimt31': 365,
                           'plasimt42': 365}  # noteL we are ignoring leap years in plasim here,

days_per_year = days_per_year_per_model[modelname]

data_for_clim = full_data[30 * days_per_year:(30 + 30) * days_per_year]

# the time axis of the puma data does not contain the acutal date, but day of year
# as float. we convert it to int and name it "dayofyear"
data_for_clim['dayofyear'] = data_for_clim['time'].astype('int')

# wecompute a 50week running mean (ecmwf does the same for their ACC)
window = 5 * 7  # days
clim = data_for_clim.rolling(time=window, min_periods=1, center=True).mean().groupby('dayofyear').mean('time')
print('computing climatology')
clim.load()

max_leadtime = 14  # this must be the same as in the prediction script!!
lead_times = np.arange(1, max_leadtime + 1)

subreg = [0, 90, 360, -90]  # subregion to compute the scores
subregstr = str(subreg[0])+'-'+str(subreg[2])+'E'+str(subreg[3])+'-'+str(subreg[1])+'N'

#varnames = varname_to_levidx.keys()
varnames = ['zg500', 'ta800']
res = []


train_years_list = [1,2,5,10,20,50,100, 200, 400, 800]

for train_years in train_years_list:

    if train_years <=20:
        train_years_offset_list = [0, 10, 20, 30, 40]
    else:
        train_years_offset_list = [0]

        
    for train_years_offset in train_years_offset_list:

        param_string = modelname + '_' + '_'.join(
            [str(e) for e in (train_years, train_years_offset, num_epochs, lead_time_training)])

        print(param_string)

        network_path = '/home/x_sebsc/gcm_complexity_machinelearning/machine_learning/leadday1/data/'

        # load the normalization weights to normalize the test data in the same way
        # as the training data
        norm_mean = xr.open_dataarray(network_path + '/norm_mean_' + param_string + '.nc')
        norm_std = xr.open_dataarray(network_path + '/norm_std_' + param_string + '.nc')

        truth_all = xr.open_dataarray(prediction_dir + '/truth_'+modelname+'_10_0_10_1.nc')

        # this is now the truth for all lead times (up to 14 days), we have to remove
        # lead_time days at the end or beginning, depending on the lead time

        # loop over variables
        for var in varnames:
            print(var)

            for lead_time in lead_times:
                print(lead_time)
                # load the predictions
                pred = xr.open_dataarray(
                    prediction_dir + '/network_prediction_' + param_string + '_leadtime' + str(lead_time) + '.nc')

                # the prediction has max_leadtime less timesteps, therefore we
                # have to crop the truth accordingly
                if lead_time < max_leadtime:
                    truth = truth_all[lead_time:-(max_leadtime - lead_time)]
                else:
                    truth = truth_all[lead_time:]

                assert (truth.dims == pred.dims)
                assert (np.array_equal(truth.coords, pred.coords))

                # select desired variable and subregion
                truth = truth.isel(lev=varname_to_levidx[var])
                pred = pred.isel(lev=varname_to_levidx[var])

                truth = sellonlatbox(truth, *subreg)
                pred = sellonlatbox(pred, *subreg)

                truth['dayofyear'] = truth['time'].astype('int')
                pred['dayofyear'] = pred['time'].astype('int')

                mse = ((truth - pred) ** 2).mean(('time', 'lat', 'lon'))

                # compute not-normalized mse
                truth_normed = (truth - norm_mean) / norm_std.isel(lev=varname_to_levidx[var])
                pred_normed = (pred - norm_mean) / norm_std.isel(lev=varname_to_levidx[var])

                mse_normed = ((truth_normed - pred_normed) ** 2).mean(('time', 'lat', 'lon'))

                # select the right climate for the variable and subregion
                clim_current_var = sellonlatbox(clim.isel(lev=varname_to_levidx[var]), *subreg)


                # compute acc
                acc = acc_score(pred, truth, clim_current_var)

                df = pd.DataFrame({'lead_time': lead_time,
                               'mse': np.array(mse), 'mse_normed': np.array(mse_normed),
                               'acc': np.array(acc), 'var': var,
                               'train_years': train_years,
                               'train_years_offset': train_years_offset})

                res.append(df)

res_df = pd.concat(res)
res_df.to_csv('predction_skill_vs_trainyears_acc' + modelname +subreg+ '.csv', index=False)


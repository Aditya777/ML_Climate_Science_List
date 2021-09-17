#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


analyze the predictions mad by the neural networks. compute
maps of forecast error and save as netcdf files


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





num_epochs = 100
lead_time_training = 1


train_years_list = [100]
models = ['pumat21_noseas','pumat21', 'pumat42','plasimt21', 'plasimt42']

for modelname in models:


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
    days_per_year_per_model = {'pumat21': 360, 'pumat31': 360, 'pumat42': 360,'pumat21_noseas': 360,
                               'plasimt21': 365, 'plasimt31': 365,
                               'plasimt42': 365}  # noteL we are ignoring leap years in plasim here,

    days_per_year = days_per_year_per_model[modelname]



    max_leadtime = 14  # this must be the same as in the prediction script!!
    lead_times = np.arange(1, max_leadtime + 1)

    # hack to speed things up
    lead_times = [1,6]






    for train_years in train_years_list:

        # if train_years <=20:
        #     train_years_offset_list = [0, 10, 20, 30, 40]
        # else:
        #     train_years_offset_list = [0]
        train_years_offset_list = [0]

        for train_years_offset in train_years_offset_list:

            param_string = modelname + '_' + '_'.join(
                [str(e) for e in (train_years, train_years_offset, num_epochs, lead_time_training)])

            print(param_string)

            network_path = '/home/x_sebsc/gcm_complexity_machinelearning/machine_learning/leadday1_more_epochs/data/'

            # load the normalization weights to normalize the test data in the same way
            # as the training data
            norm_mean = xr.open_dataarray(network_path + '/norm_mean_' + param_string + '.nc')
            norm_std = xr.open_dataarray(network_path + '/norm_std_' + param_string + '.nc')

            truth_all = xr.open_dataarray(prediction_dir + '/truth_'+modelname+'_10_0_10_1.nc')

            # this is now the truth for all lead times (up to 14 days), we have to remove
            # lead_time days at the end or beginning, depending on the lead time

            # loop over variables

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



                truth['dayofyear'] = truth['time'].astype('int')
                pred['dayofyear'] = pred['time'].astype('int')

                mse = ((truth - pred) ** 2).mean('time')

                # compute not-normalized mse
                truth_normed = (truth - norm_mean) / norm_std
                pred_normed = (pred - norm_mean) / norm_std

                mse_normed = ((truth_normed - pred_normed) ** 2).mean('time')


                # save to netcdf file
                mse.to_netcdf('mse_timmean_3d_'+param_string + '_day'+str(lead_time)+'.nc')
                mse_normed.to_netcdf('mse_normed_timmean_3d_' + param_string +'_day'+str(lead_time)+'.nc')







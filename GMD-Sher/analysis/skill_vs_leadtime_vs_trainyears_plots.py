#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 18:00:48 2018

@author: sebastian
"""

import pandas as pd
from pylab import plt
import seaborn as sns


import numpy as np
from scipy.interpolate import interp1d
from scipy import optimize
        
        
# plot n_years vs skill
#%%
 






#var='ta1000'
var='zg500'


models = ['pumat21_noseas','pumat21', 'pumat42','plasimt21', 'plasimt42', 'plasimt42_regridt21', 'pumat42_regridt21']

res = []
for model in models:
    _df = pd.read_csv('predction_skill_vs_trainyears_acc'+model+'0-360E-90-90N.csv')
    
    # due to a small error in the analyze_predictsion_allmocels_acc_****.py scripts,
    # we have every line 40 times (because we have 40 levels). this happenns at 
    # L232-233, where I forgot to select the right level of "norm_mean", and it
    # everything is tehrefore atomatically extended to 40 levels.
    # the final result comes from 235, where the differences for each level cancel 
    # out again (except for very small changes due to numerical precision, after
    # the 8th point after the comme)
    
    # therefore, here we select every 40th line only
    _df = _df.iloc[::40]
    _df['model'] = model
    res.append(_df)

# for model in ('pumat42','plasimt42'):
#     _df = pd.read_csv('predction_skill_vs_trainyears_acc'+model+'regridT21global.csv')
#     _df['model'] = model+'_regridt21'
#     res.append(_df)

df = pd.concat(res)

#df = df[df['train_years_offset']==0]

df['rmse'] = np.sqrt(df['mse'])



sns.set_context('notebook')
sns.set_palette('deep')


figsize = (8,4)
for score in ('rmse','acc'):



    plt.figure(figsize=figsize)
    train_years = 100
    var='zg500'

    sub = df[(df['train_years']==train_years) & (df['var']==var)]
    sns.lineplot('lead_time', score,data=sub, hue='model', legend='full')
    plt.title(var+' train_years '+str(train_years))
    plt.ylabel(var+' '+score+' [m]')
    sns.despine()
    plt.savefig('lead_time_vs_'+score+'_allmodels'+str(train_years)+'_nooffset_'+str(var)+'.pdf')

    plt.figure(figsize=figsize)
    train_years = 100

    var='ta800'
    sub = df[(df['train_years']==train_years) & (df['var']==var)]
    sns.lineplot('lead_time', score,data=sub, hue='model', legend='full')
    plt.title(var+' train_years '+str(train_years))
    plt.ylabel(var + ' ' + score + ' [K]')
    sns.despine()
    plt.savefig('lead_time_vs_'+score+'_allmodels'+str(train_years)+'_nooffset_'+str(var)+'.pdf')


    var='zg500'
    lead_time=1
    plt.figure(figsize=figsize)
    sub = df[(df['lead_time']==lead_time) & (df['var']==var)]
    sns.lineplot('train_years',score,hue='model', data=sub, legend='full')
    plt.suptitle(var+' lead_time='+str(lead_time))
    plt.ylabel(var + ' ' + score + ' [m]')
    sns.despine()
    plt.savefig('train_year_vs_'+score+'_leadtime_'+str(lead_time)+'_'+str(var)+'.pdf')

    lead_time=6

    plt.figure(figsize=figsize)
    sub = df[(df['lead_time']==lead_time) & (df['var']==var)]
    sns.lineplot('train_years',score,hue='model', data=sub, legend='full', estimator='mean')
    plt.suptitle(var+' lead_time='+str(lead_time))
    plt.ylabel(var + ' ' + score + ' [m]')
    sns.despine()
    plt.savefig('train_year_vs_'+score+'_leadtime_'+str(lead_time)+'_'+str(var)+'.pdf')



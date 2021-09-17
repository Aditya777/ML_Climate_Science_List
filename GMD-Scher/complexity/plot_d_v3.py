#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: sebastian
"""

import pandas as pd
import seaborn as sns
from pylab import plt
import numpy as np

df_comp = pd.read_csv('model_vs_d_spread_plasim_puma_multivar_notsquared.csv',index_col=0)




einterim = pd.read_csv('model_vs_d_einterim_multivar_notsquared.csv',index_col=0)
# fix variable names in einterim
einterim = einterim.melt(id_vars='model', var_name='var', value_name='d')
df = pd.concat([df_comp,einterim])

var = 'zg500'
df_sub = df[df['var'] == var]

sns.set_context('notebook')
fig = plt.figure()
sns.barplot('model','d',data=df_sub, color='#1b9e77',
            order=['pumat21_noseas', 'pumat21','plasimt21', 'pumat42', 'plasimt42', 
       'pumat42_regridt21', 'plasimt42_regridt21','era-interim'])
plt.xticks(rotation=25, ha='right')
plt.ylabel('d z500')
fig.tight_layout()
sns.despine()
plt.savefig('model_vs_d_all.pdf')



plt.figure(figsize=(10,13))
for i,v in enumerate(('zg','ta','ua','va')):
    plt.subplot(4,1,i+1)

    vars = [v+str(lev) for lev in np.arange(100,1001,100)]
    df_sub = df[df['var'].isin(vars)]
    sns.barplot(x='var',y='d', hue='model', data=df_sub)
    plt.xticks(rotation=20)
    plt.ylabel('d')
    sns.despine()
plt.tight_layout()
plt.savefig('d_barplot_puma_plasim_allvars.pdf')
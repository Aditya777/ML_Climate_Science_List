import numpy as np
np.random.seed(42)

import math
import keras
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import rmsprop, SGD, Adagrad, Adadelta
import json
import matplotlib.pyplot as plt
import sys

from eccodes import *


n_halo = int(sys.argv[1])
n_level = int(sys.argv[2])

print 'n_halo = ',n_halo, '; n_level: ', n_level

n_offset = 1
n_inputs = 1860
n_batch = 128
n_total = 67199
n_iter = 1
n_train = n_total-n_offset
n_train_sets = n_train*(n_inputs-120*n_halo)
n_input_sets = 2*(2*n_halo+1)**2+4
n_stencil = (2*n_halo+1)**2
n_output_sets = 2


infile1 = './Era5_Z500_6degree_hourly.grib'
infile2 = './Era5_2mt_6degree_hourly.grib'

Z500_2d = np.zeros((n_total,n_inputs))
Z500 = np.zeros((n_total,n_inputs/60,60))

f2mt_2d = np.zeros((n_total,n_inputs))
f2mt = np.zeros((n_total,n_inputs/60,60))

##############
#Define NN:

n_dofs=n_stencil+4

modelx1 = Sequential()

modelx1.add(Dense(n_input_sets, input_dim=n_input_sets, activation='tanh'))
if n_level>0: 
    modelx1.add(Dense(n_dofs, activation='tanh'))
if n_level>1: 
    modelx1.add(Dense(n_dofs, activation='tanh'))
if n_level>2: 
    modelx1.add(Dense(n_dofs, activation='tanh'))
if n_level>3: 
    modelx1.add(Dense(n_dofs, activation='tanh'))
modelx1.add(Dense(n_output_sets, activation='tanh'))

modelx1.compile(loss='mean_absolute_error', optimizer='SGD', metrics=['mae'])
modelx1.load_weights('./weights/best_'+str(n_inputs)+'_'+str(n_halo)+'_'+str(n_level)+'_yearly_pole_2fields')



for nset in range(n_iter):
    data_list = []
    with open(infile1,'rb') as gf1:
        for i in range(nset*n_total+1):
            gid = codes_grib_new_from_file(gf1)
            codes_release(gid)
        for _ in range(n_total):
            gid = codes_grib_new_from_file(gf1)
            if gid is None:
                break
            param =codes_get(gid, 'shortName')
            data_list.append(codes_get_values(gid))
            codes_release(gid)

    print param, np.shape(data_list) 
    Z500_2d = np.array(data_list)
    del(data_list)
             
    for i in range(n_total):
        for j in range(n_inputs/60):
            for k in range(60):
                Z500[i,j,k]=Z500_2d[i,j*60+k]


    data_list = []
    with open(infile2,'rb') as gf1:
        for i in range(nset*n_total+1):
            gid = codes_grib_new_from_file(gf1)
            codes_release(gid)
            #    while True:
        for _ in range(n_total):
            gid = codes_grib_new_from_file(gf1)
            if gid is None:
                break
            param =codes_get(gid, 'shortName')
            data_list.append(codes_get_values(gid))
            codes_release(gid)

    print param, np.shape(data_list) 
    f2mt_2d = np.array(data_list)
    del(data_list)
             
    for i in range(n_total):
        for j in range(n_inputs/60):
            for k in range(60):
                f2mt[i,j,k]=f2mt_2d[i,j*60+k]

    x_train = np.zeros((n_train_sets,n_input_sets))
    y_train = np.zeros((n_train_sets,n_output_sets))

    n_hour = nset*n_total+1
    print 'hour: ', n_hour, ' , halo: ', n_halo
    n_count = -1
    for i in range(n_train):
        n_hour=n_hour+1 
        n_lat = -1
        for j in range(n_halo,n_inputs/60-n_halo):
            n_lat = n_lat + 1
            n_lon = -1
            for k in range(60):
                n_lon = n_lon+1
                n_count = n_count+1
                y_train[n_count,0] = (Z500[i+n_offset,j,k]-Z500[i,j,k])            
                y_train[n_count,1] = (f2mt[i+n_offset,j,k]-f2mt[i,j,k])            
                x_train[n_count,n_input_sets-4] = np.mod(4*n_hour,24*1461)/(24.0*1461.0)-0.5
                x_train[n_count,n_input_sets-3] = np.mod(n_hour,24)/24.0-0.5
                x_train[n_count,n_input_sets-2] = n_lat/(n_inputs/60.0-(2*n_halo))-0.5
                x_train[n_count,n_input_sets-1] = n_lon/60.0-0.5
                m_count = -1
                for l in range(-n_halo,n_halo+1):
                    for m in range(-n_halo,n_halo+1):
                        m_count = m_count+1
                        n_lon=(k+m)%60
                        x_train[n_count,m_count] = Z500[i,j+l,n_lon]
                        x_train[n_count,n_stencil+m_count] = f2mt[i,j+l,n_lon]

    max_Z500I =  60000.0 
    min_Z500I =  47000.0 
    max_f2mtI =  315.0 
    min_f2mtI =  228.0

    x_train[:,:n_input_sets-n_stencil-4] = 2.0*(x_train[:,:n_input_sets-n_stencil-4]-min_Z500I)/(max_Z500I-min_Z500I)-1.0
    x_train[:,n_stencil:n_input_sets-4] = 2.0*(x_train[:,n_stencil:n_input_sets-4]-min_f2mtI)/(max_f2mtI-min_f2mtI)-1.0

    max_Z500O =  350.0
    min_Z500O = -350.0
    max_f2mtO =  5.0 
    min_f2mtO = -5.0

    y_train[:,0] = 2.0*(y_train[:,0]-min_Z500O)/(max_Z500O-min_Z500O)-1.0
    y_train[:,1] = 2.0*(y_train[:,1]-min_f2mtO)/(max_f2mtO-min_f2mtO)-1.0

    modelx1.fit(x_train, y_train, epochs=200,batch_size=n_batch,validation_split=0.2)
    modelx1.save_weights('./weights/best_'+str(n_inputs)+'_'+str(n_halo)+'_'+str(n_level)+'_yearly_pole_2fields')
    

del(Z500_2d)
del(f2mt_2d)


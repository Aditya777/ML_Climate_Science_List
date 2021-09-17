import numpy as np
np.random.seed(42)

import math
import keras
from keras.models import Sequential
from keras import layers
from keras.layers import Conv1D, Dense, Dropout, Activation
from keras.optimizers import rmsprop, SGD, Adagrad, Adadelta
import json
import matplotlib.pyplot as plt
import sys


from eccodes import *

n_level = int(sys.argv[1])

print 'n_level: ', n_level

n_offset = 1
n_inputs = 1860
n_batch = 128
n_total = 67199
n_iter = 1 
n_train = n_total-n_offset
n_test_sets = n_test
n_train_sets = n_train
n_input_sets = 2*n_inputs+2
n_output_sets = 2*n_inputs


infile1 = './Era5_Z500_6degree_hourly.grib'
infile2 = './Era5_2mt_6degree_hourly.grib'

Z500_2d = np.zeros((n_total,n_inputs))
Z500 = np.zeros((n_total,n_inputs/60,60))
f2mt_2d = np.zeros((n_total,n_inputs))
f2mt = np.zeros((n_total,n_inputs/60,60))

##############
#Define NN:

n_dofs=n_input_sets

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
modelx1.load_weights('./weights/best_'+str(n_inputs)+'_global_'+str(n_level)+'_2fields')


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

    data_list = []
    with open(infile2,'rb') as gf1:
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
    f2mt_2d = np.array(data_list)
    del(data_list)

             
    #Normailse
    x_train = np.zeros((n_train_sets,n_input_sets))
    y_train = np.zeros((n_train_sets,n_output_sets))
    x_test = np.zeros((n_test_sets,n_input_sets))
    y_test = np.zeros((n_test_sets,n_output_sets))
    y_result = np.zeros((n_test_sets,n_output_sets))

    n_hour = nset*n_total+1
    print 'hour: ', n_hour
    n_count = -1
    for i in range(n_train):
        n_hour=n_hour+1 
        n_count = n_count+1
        y_train[n_count,:n_inputs] = (Z500_2d[i+n_offset,:]-Z500_2d[i,:])            
        y_train[n_count,n_inputs:] = (f2mt_2d[i+n_offset,:]-f2mt_2d[i,:])            
        x_train[n_count,0:n_input_sets-n_inputs-2] = (Z500_2d[i,:])                  
        x_train[n_count,n_inputs:n_input_sets-2] = (f2mt_2d[i,:])            
        x_train[n_count,n_input_sets-2] = np.mod(4*n_hour,24*1461)/(24.0*1461.0)-0.5
        x_train[n_count,n_input_sets-1] = np.mod(n_hour,24)/24.0-0.5  

    max_Z500I =  60000.0
    min_Z500I =  47000.0
    max_Z500O =   350.0
    min_Z500O =  -350.0
    max_f2mtI =  315.0
    min_f2mtI =  228.0
    max_f2mtO =   5.0
    min_f2mtO =  -5.0

    x_train[:,:n_input_sets-n_inputs-2] = 2.0*(x_train[:,:n_input_sets-n_inputs-2]-min_Z500I)/(max_Z500I-min_Z500I)-1.0
    x_test[:,:n_input_sets-n_inputs-2] = 2.0*(x_test[:,:n_input_sets-n_inputs-2]-min_Z500I)/(max_Z500I-min_Z500I)-1.0
    y_train[:,:n_inputs] = 2.0*(y_train[:,:n_inputs]-min_Z500O)/(max_Z500O-min_Z500O)-1.0

    x_train[:,n_inputs:n_input_sets-2] = 2.0*(x_train[:,n_inputs:n_input_sets-2]-min_f2mtI)/(max_f2mtI-min_f2mtI)-1.0
    x_test[:,n_inputs:n_input_sets-2] = 2.0*(x_test[:,n_inputs:n_input_sets-2]-min_f2mtI)/(max_f2mtI-min_f2mtI)-1.0
    y_train[:,n_inputs:] = 2.0*(y_train[:,n_inputs:]-min_f2mtO)/(max_f2mtO-min_f2mtO)-1.0


    modelx1.fit(x_train, y_train, epochs=200,batch_size=n_batch,validation_split=0.2)
    modelx1.save_weights('./best_'+str(n_inputs)+'_global_'+str(n_level)+'_2fields')
    

    modelx1.fit(x_train, y_train, epochs=500,batch_size=n_batch,validation_split=0.2)
    modelx1.save_weights('./best_'+str(n_inputs)+'_global_'+str(n_level)+'_2fields')
    

    modelx1.fit(x_train, y_train, epochs=500,batch_size=n_batch,validation_split=0.2)
    modelx1.save_weights('./best_'+str(n_inputs)+'_global_'+str(n_level)+'_2fields')
    

    modelx1.fit(x_train, y_train, epochs=500,batch_size=n_batch,validation_split=0.2)
    modelx1.save_weights('./best_'+str(n_inputs)+'_global_'+str(n_level)+'_2fields')
       

del(Z500_2d)
del(f2mf_2d)


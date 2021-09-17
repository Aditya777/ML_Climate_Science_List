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

from eccodes import *

n_dates = 10
n_inputs = 1860
n_batch = 128
n_test =  121
n_level = 4

n_input_sets = n_inputs+2
n_output_sets = n_inputs



######################
#Setup NN:

n_dofs=n_input_sets

model = Sequential()
model.add(Dense(n_input_sets, input_dim=n_input_sets, activation='tanh'))
if n_level>0: 
    model.add(Dense(n_dofs, activation='tanh'))
if n_level>1: 
    model.add(Dense(n_dofs, activation='tanh'))
if n_level>2: 
    model.add(Dense(n_dofs, activation='tanh'))
if n_level>3: 
    model.add(Dense(n_dofs, activation='tanh'))
model.add(Dense(n_input_sets, activation='tanh'))
model.compile(loss='mean_absolute_error', optimizer='SGD', metrics=['mae'])
model.load_weights('./best_'+str(n_inputs)+'_global_'+str(n_level))


#Normalise
max_Z500I = 63000.0
min_Z500I = 43000.0
max_Z500O =  400.0
min_Z500O = -400.0

#Define State:
state = np.zeros((1,n_input_sets))
out0 = np.zeros((1,n_output_sets))
out1 = np.zeros((1,n_output_sets))
out2 = np.zeros((1,n_output_sets))
out3 = np.zeros((1,n_output_sets))
out4 = np.zeros((1,n_output_sets))


for nday in range(n_dates):

    if nday==0:     
        fout_Z500 = open("./Z500_NN_global_20170301","w")
        infile = './../Analysis/Z500_analysis_20170301'
        n_hour = 1416
    if nday==1:     
        fout_Z500 = open("./Z500_NN_global_20170410","w")
        infile = './../Analysis/Z500_analysis_20170410'
        n_hour = 2376
    if nday==2:     
        fout_Z500 = open("./Z500_NN_global_20170520","w")
        infile = './../Analysis/Z500_analysis_20170520'
        n_hour = 3336
    if nday==3:     
        fout_Z500 = open("./Z500_NN_global_20170629","w")
        infile = './../Analysis/Z500_analysis_20170629'
        n_hour = 4296
    if nday==4:     
        fout_Z500 = open("./Z500_NN_global_20170808","w")
        infile = './../Analysis/Z500_analysis_20170808'
        n_hour = 5256
    if nday==5:     
        fout_Z500 = open("./Z500_NN_global_20170917","w")
        infile = './../Analysis/Z500_analysis_20170917'
        n_hour = 6216
    if nday==6:     
        fout_Z500 = open("./Z500_NN_global_20171027","w")
        infile = './../Analysis/Z500_analysis_20171027'
        n_hour = 7176
    if nday==7:     
        fout_Z500 = open("./Z500_NN_global_20171206","w")
        infile = './../Analysis/Z500_analysis_20171206'
        n_hour = 8136
    if nday==8:     
        fout_Z500 = open("./Z500_NN_global_20180115","w")
        infile = './../Analysis/Z500_analysis_20180115'
        n_hour = 9096
    if nday==9:     
        fout_Z500 = open("./Z500_NN_global_20180224","w")
        infile = './../Analysis/Z500_analysis_20180224'
        n_hour = 10056

    Z500_list = []

    with open(infile,'rb') as gf1:
        gid = codes_grib_new_from_file(gf1)
        param =codes_get(gid, 'shortName')
        Z500_list.append(codes_get_values(gid))
        codes_release(gid)
        
    with open(infile,'rb') as gp1:
        gid = codes_grib_new_from_file(gp1)
        param =codes_get(gid, 'shortName')
        output_Z500 = codes_clone(gid)
        codes_release(gid)

    print param, np.shape(Z500_list)

    Z500 = np.zeros(n_inputs)
    Z500 = Z500_list[0]
    del(Z500_list)

    ######################
    # Calculate trajectory

    for i in range(n_test):
        out3=out2
        out2=out1
        n_hour = n_hour+1
        if i%6 == 0:
            print i
            codes_set_values(output_Z500,Z500)   
            codes_write(output_Z500,fout_Z500)
        state[0,:n_input_sets-2] = Z500[:]
        state[0,n_input_sets-2] = np.mod(4*n_hour,24*1461)/(24.0*1461.0)-0.5
        state[0,n_input_sets-1] = np.mod(n_hour,24)/24.0-0.5
        state[:,:n_input_sets-2] = 2.0*(state[:,:n_input_sets-2]-min_Z500I)/(max_Z500I-min_Z500I)-1.0
        out1 = model.predict(state,batch_size=1)
        out1 = (out1+1.0)*(max_Z500O-min_Z500O)/2.0+min_Z500O
        if i==0: 
            out0 = out1
        if i==1: 
            out0 = 1.5*out1-0.5*out2
        if i>1: 
            out0 = (23.0/12.0)*out1-(4.0/3.0)*out2+(5.0/12.0)*out3
        Z500 = Z500+out0[0,:]

    codes_release(output_Z500)
    fout_Z500.close()


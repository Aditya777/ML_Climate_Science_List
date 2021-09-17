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
n_level = 3

n_input_sets = 2*n_inputs+2
n_output_sets = 2*n_inputs



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
model.load_weights('./best_'+str(n_inputs)+'_global_'+str(n_level)+'_2fields')


max_Z500I =  60000.0
min_Z500I =  47000.0
max_Z500O =   350.0
min_Z500O =  -350.0
max_f2mtI =  315.0
min_f2mtI =  228.0
max_f2mtO =   5.0
min_f2mtO =  -5.0

#Define State:
state = np.zeros((1,n_input_sets))
out0 = np.zeros((1,n_output_sets))
out1 = np.zeros((1,n_output_sets))
out2 = np.zeros((1,n_output_sets))
out3 = np.zeros((1,n_output_sets))
out4 = np.zeros((1,n_output_sets))


for nday in range(n_dates):

    if nday==0:     
        fout_Z500 = open("./Z500_NN_global_20170301_2fields","w")
        infile1 = './../Analysis/Z500_analysis_20170301'
        fout_f2mt = open("./t2m_NN_global_20170301_2fields","w")
        infile2 = './../Analysis/t2m_analysis_20170301'
        n_hour = 1416
    if nday==1:     
        fout_Z500 = open("./Z500_NN_global_20170410_2fields","w")
        infile1 = './../Analysis/Z500_analysis_20170410'
        fout_f2mt = open("./t2m_NN_global_20170410_2fields","w")
        infile2 = './../Analysis/t2m_analysis_20170410'
        n_hour = 2376
    if nday==2:     
        fout_Z500 = open("./Z500_NN_global_20170520_2fields","w")
        infile1 = './../Analysis/Z500_analysis_20170520'
        fout_f2mt = open("./t2m_NN_global_20170520_2fields","w")
        infile2 = './../Analysis/t2m_analysis_20170520'
        n_hour = 3336
    if nday==3:     
        fout_Z500 = open("./Z500_NN_global_20170629_2fields","w")
        infile1 = './../Analysis/Z500_analysis_20170629'
        fout_f2mt = open("./t2m_NN_global_20170629_2fields","w")
        infile2 = './../Analysis/t2m_analysis_20170629'
        n_hour = 4296
    if nday==4:     
        fout_Z500 = open("./Z500_NN_global_20170808_2fields","w")
        infile1 = './../Analysis/Z500_analysis_20170808'
        fout_f2mt = open("./t2m_NN_global_20170808_2fields","w")
        infile2 = './../Analysis/t2m_analysis_20170808'
        n_hour = 5256
    if nday==5:     
        fout_Z500 = open("./Z500_NN_global_20170917_2fields","w")
        infile1 = './../Analysis/Z500_analysis_20170917'
        fout_f2mt = open("./t2m_NN_global_20170917_2fields","w")
        infile2 = './../Analysis/t2m_analysis_20170917'
        n_hour = 6216
    if nday==6:     
        fout_Z500 = open("./Z500_NN_global_20171027_2fields","w")
        infile1 = './../Analysis/Z500_analysis_20171027'
        fout_f2mt = open("./t2m_NN_global_20171027_2fields","w")
        infile2 = './../Analysis/t2m_analysis_20171027'
        n_hour = 7176
    if nday==7:     
        fout_Z500 = open("./Z500_NN_global_20171206_2fields","w")
        infile1 = './../Analysis/Z500_analysis_20171206'
        fout_f2mt = open("./t2m_NN_global_20171206_2fields","w")
        infile2 = './../Analysis/t2m_analysis_20171206'
        n_hour = 8136
    if nday==8:     
        fout_Z500 = open("./Z500_NN_global_20180115_2fields","w")
        infile1 = './../Analysis/Z500_analysis_20180115'
        fout_f2mt = open("./t2m_NN_global_20180115_2fields","w")
        infile2 = './../Analysis/t2m_analysis_20180115'
        n_hour = 9096
    if nday==9:     
        fout_Z500 = open("./Z500_NN_global_20180224_2fields","w")
        infile1 = './../Analysis/Z500_analysis_20180224'
        fout_f2mt = open("./t2m_NN_global_20180224_2fields","w")
        infile2 = './../Analysis/t2m_analysis_20180224'
        n_hour = 10056

    data_list = []
    with open(infile1,'rb') as gf1:
        gid = codes_grib_new_from_file(gf1)
        param =codes_get(gid, 'shortName')
        data_list.append(codes_get_values(gid))
        codes_release(gid)
        
    with open(infile1,'rb') as gp1:
        gid = codes_grib_new_from_file(gp1)
        param =codes_get(gid, 'shortName')
        output_Z500 = codes_clone(gid)
        codes_release(gid)

    print param, np.shape(data_list)

    Z500 = np.zeros(n_inputs)
    Z500 = data_list[0] 
    del(data_list)


    data_list = []
    with open(infile2,'rb') as gf1:
        gid = codes_grib_new_from_file(gf1)
        param =codes_get(gid, 'shortName')
        data_list.append(codes_get_values(gid))
        codes_release(gid)
        
    with open(infile2,'rb') as gp1:
        gid = codes_grib_new_from_file(gp1)
        param =codes_get(gid, 'shortName')
        output_f2mt = codes_clone(gid)
        codes_release(gid)

    print param, np.shape(data_list)

    f2mt = np.zeros(n_inputs)
    f2mt = data_list[0]
    del(data_list)


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
            codes_set_values(output_f2mt,f2mt)   
            codes_write(output_f2mt,fout_f2mt)
        state[0,0:n_input_sets-n_inputs-2] = Z500[:]
        state[0,n_inputs:n_input_sets-2] = f2mt[:]
        state[0,n_input_sets-2] = np.mod(4*n_hour,24*1461)/(24.0*1461.0)-0.5
        state[0,n_input_sets-1] = np.mod(n_hour,24)/24.0-0.5
        state[:,:n_input_sets-n_inputs-2] = 2.0*(state[:,:n_input_sets-n_inputs-2]-min_Z500I)/(max_Z500I-min_Z500I)-1.0
        state[:,n_inputs:n_input_sets-2] = 2.0*(state[:,n_inputs:n_input_sets-2]-min_f2mtI)/(max_f2mtI-min_f2mtI)-1.0
        out1 = model.predict(state,batch_size=1)
        out1[:,:n_inputs] = (out1[:,:n_inputs]+1.0)*(max_Z500O-min_Z500O)/2.0+min_Z500O
        out1[:,n_inputs:] = (out1[:,n_inputs:]+1.0)*(max_f2mtO-min_f2mtO)/2.0+min_f2mtO
        if i==0: 
            out0 = out1
        if i==1: 
            out0 = 1.5*out1-0.5*out2
        if i>1: 
            out0 = (23.0/12.0)*out1-(4.0/3.0)*out2+(5.0/12.0)*out3
        Z500 = Z500+out0[0,:n_inputs]
        f2mt = f2mt+out0[0,n_inputs:]

    codes_release(output_Z500)
    fout_Z500.close()
    codes_release(output_f2mt)
    fout_f2mt.close()


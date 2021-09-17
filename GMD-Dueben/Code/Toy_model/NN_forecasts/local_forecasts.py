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

n_state_sets_0 = 2
n_input_sets_0 = 60+1
n_state_sets_1 = (120)
n_input_sets_1 = (2+1)**2+4
n_state_sets_2 = (120)
n_input_sets_2 = (2*2+1)**2+4
n_state_sets_3 = (120)
n_input_sets_3 = (2*3+1)**2+4
n_state_sets_4 = (n_inputs-120*4)
n_input_sets_4 = (2*4+1)**2+4

n_output_sets = 1



######################
#Setup NN:

n_dofs=60

model_0 = Sequential()
model_0.add(Dense(n_input_sets_0, input_dim=n_input_sets_0, activation='tanh'))
model_0.add(Dense(n_dofs, activation='tanh'))
model_0.add(Dense(n_dofs, activation='tanh'))
model_0.add(Dense(n_dofs, activation='tanh'))
model_0.add(Dense(n_input_sets_0, activation='tanh'))
model_0.compile(loss='mean_absolute_error', optimizer='SGD', metrics=['mae'])
model_0.load_weights('./best_pole', by_name=False)

n_dofs=n_input_sets_1
model_1 = Sequential()
model_1.add(Dense(n_input_sets_1, input_dim=n_input_sets_1, activation='tanh'))
if n_level>0: 
    model_1.add(Dense(n_dofs, activation='tanh'))
if n_level>1: 
    model_1.add(Dense(n_dofs, activation='tanh'))
if n_level>2: 
    model_1.add(Dense(n_dofs, activation='tanh'))
if n_level>3: 
    model_1.add(Dense(n_dofs, activation='tanh'))
model_1.add(Dense(n_input_sets_1, activation='tanh'))
model_1.compile(loss='mean_absolute_error', optimizer='SGD', metrics=['mae'])
model_1.load_weights('./best_'+str(n_inputs)+'_1_'+str(n_level)+'_yearly_pole', by_name=False)

n_dofs=n_input_sets_2
model_2 = Sequential()
model_2.add(Dense(n_input_sets_2, input_dim=n_input_sets_2, activation='tanh'))
if n_level>0: 
    model_2.add(Dense(n_dofs, activation='tanh'))
if n_level>1: 
    model_2.add(Dense(n_dofs, activation='tanh'))
if n_level>2: 
    model_2.add(Dense(n_dofs, activation='tanh'))
if n_level>3: 
    model_2.add(Dense(n_dofs, activation='tanh'))
model_2.add(Dense(n_input_sets_2, activation='tanh'))
model_2.compile(loss='mean_absolute_error', optimizer='SGD', metrics=['mae'])
model_2.load_weights('./best_'+str(n_inputs)+'_2_'+str(n_level)+'_yearly_pole', by_name=False)

n_dofs=n_input_sets_3
model_3 = Sequential()
model_3.add(Dense(n_input_sets_3, input_dim=n_input_sets_3, activation='tanh'))
if n_level>0: 
    model_3.add(Dense(n_dofs, activation='tanh'))
if n_level>1: 
    model_3.add(Dense(n_dofs, activation='tanh'))
if n_level>2: 
    model_3.add(Dense(n_dofs, activation='tanh'))
if n_level>3: 
    model_3.add(Dense(n_dofs, activation='tanh'))
model_3.add(Dense(n_input_sets_3, activation='tanh'))
model_3.compile(loss='mean_absolute_error', optimizer='SGD', metrics=['mae'])
model_3.load_weights('./best_'+str(n_inputs)+'_3_'+str(n_level)+'_yearly_pole', by_name=False)

n_dofs=n_input_sets_4
model_4 = Sequential()
model_4.add(Dense(n_input_sets_4, input_dim=n_input_sets_4, activation='tanh'))
if n_level>0: 
    model_4.add(Dense(n_dofs, activation='tanh'))
if n_level>1: 
    model_4.add(Dense(n_dofs, activation='tanh'))
if n_level>2: 
    model_4.add(Dense(n_dofs, activation='tanh'))
if n_level>3: 
    model_4.add(Dense(n_dofs, activation='tanh'))
model_4.add(Dense(n_input_sets_4, activation='tanh'))
model_4.compile(loss='mean_absolute_error', optimizer='SGD', metrics=['mae'])
model_4.load_weights('./best_'+str(n_inputs)+'_4_'+str(n_level)+'_yearly_pole', by_name=False)


#Normalise
max_Z500I_0 = 53609.0
min_Z500I_0 = 45351.0
max_Z500I =   60000.0
min_Z500I =   47000.0

max_Z500O_0 =  300.0
min_Z500O_0 = -300.0
max_Z500O =  350.0
min_Z500O = -350.0

#Define State:
state_0 = np.zeros((n_state_sets_0,n_input_sets_0))
state_1 = np.zeros((n_state_sets_1,n_input_sets_1))
state_2 = np.zeros((n_state_sets_2,n_input_sets_2))
state_3 = np.zeros((n_state_sets_3,n_input_sets_3))
state_4 = np.zeros((n_state_sets_4,n_input_sets_4))
out0 = np.zeros((n_state_sets_0,1))
out1 = np.zeros((n_state_sets_1,1))
out2 = np.zeros((n_state_sets_2,1))
out3 = np.zeros((n_state_sets_3,1))
out4 = np.zeros((n_state_sets_4,1))
out1_0 = np.zeros((n_state_sets_0,1))
out1_1 = np.zeros((n_state_sets_1,1))
out1_2 = np.zeros((n_state_sets_2,1))
out1_3 = np.zeros((n_state_sets_3,1))
out1_4 = np.zeros((n_state_sets_4,1))
out2_0 = np.zeros((n_state_sets_0,1))
out2_1 = np.zeros((n_state_sets_1,1))
out2_2 = np.zeros((n_state_sets_2,1))
out2_3 = np.zeros((n_state_sets_3,1))
out2_4 = np.zeros((n_state_sets_4,1))
out3_0 = np.zeros((n_state_sets_0,1))
out3_1 = np.zeros((n_state_sets_1,1))
out3_2 = np.zeros((n_state_sets_2,1))
out3_3 = np.zeros((n_state_sets_3,1))
out3_4 = np.zeros((n_state_sets_4,1))

for nday in range(n_dates):

    if nday==0:     
        fout_Z500 = open("./Z500_NN_local_20170301","w")
        infile = './../Analysis/Z500_analysis_20170301'
        n_hour = 1416
    if nday==1:     
        fout_Z500 = open("./Z500_NN_local_20170410","w")
        infile = './../Analysis/Z500_analysis_20170410'
        n_hour = 2376
    if nday==2:     
        fout_Z500 = open("./Z500_NN_local_20170520","w")
        infile = './../Analysis/Z500_analysis_20170520'
        n_hour = 3336
    if nday==3:     
        fout_Z500 = open("./Z500_NN_local_20170629","w")
        infile = './../Analysis/Z500_analysis_20170629'
        n_hour = 4296
    if nday==4:     
        fout_Z500 = open("./Z500_NN_local_20170808","w")
        infile = './../Analysis/Z500_analysis_20170808'
        n_hour = 5256
    if nday==5:     
        fout_Z500 = open("./Z500_NN_local_20170917","w")
        infile = './../Analysis/Z500_analysis_20170917'
        n_hour = 6216
    if nday==6:     
        fout_Z500 = open("./Z500_NN_local_20171027","w")
        infile = './../Analysis/Z500_analysis_20171027'
        n_hour = 7176
    if nday==7:     
        fout_Z500 = open("./Z500_NN_local_20171206","w")
        infile = './../Analysis/Z500_analysis_20171206'
        n_hour = 8136
    if nday==8:     
        fout_Z500 = open("./Z500_NN_local_20180115","w")
        infile = './../Analysis/Z500_analysis_20180115'
        n_hour = 9096
    if nday==9:     
        fout_Z500 = open("./Z500_NN_local_20180224","w")
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
    Z500_2ds = np.zeros((n_inputs/60,60))
    del(Z500_list)

    ######################
    # Calculate trajectory
    n_lat_0 = 0
    n_lat_28 = 28
    n_lat_26 = 26
    n_lat_24 = 24
    n_lon = -1
    for k in range(60):
        n_lon = n_lon+1
        state_1[k,n_input_sets_1-2] = n_lat_0/(n_inputs/60.0-(2))-0.5
        state_1[k,n_input_sets_1-1] = n_lon/60.0-0.5
        state_1[k+60,n_input_sets_1-2] = n_lat_28/(n_inputs/60.0-(2))-0.5
        state_1[k+60,n_input_sets_1-1] = n_lon/60.0-0.5
        state_2[k,n_input_sets_2-2] = n_lat_0/(n_inputs/60.0-(2*2))-0.5
        state_2[k,n_input_sets_2-1] = n_lon/60.0-0.5
        state_2[k+60,n_input_sets_2-2] = n_lat_26/(n_inputs/60.0-(2*2))-0.5
        state_2[k+60,n_input_sets_2-1] = n_lon/60.0-0.5
        state_3[k,n_input_sets_3-2] = n_lat_0/(n_inputs/60.0-(2*3))-0.5
        state_3[k,n_input_sets_3-1] = n_lon/60.0-0.5
        state_3[k+60,n_input_sets_3-2] = n_lat_24/(n_inputs/60.0-(2*3))-0.5
        state_3[k+60,n_input_sets_3-1] = n_lon/60.0-0.5

    n_count = -1
    n_lat = -1
    for j in range(4,n_inputs/60-4):
        n_lat = n_lat + 1
        n_lon = -1
        for k in range(60):
            n_lon = n_lon+1
            n_count = n_count+1
            state_4[n_count,n_input_sets_4-2] = n_lat/(n_inputs/60.0-(2*4))-0.5
            state_4[n_count,n_input_sets_4-1] = n_lon/60.0-0.5
    for i in range(n_test):
        out3_0=out2_0
        out3_1=out2_1
        out3_2=out2_2
        out3_3=out2_3
        out3_4=out2_4
        out2_0=out1_0
        out2_1=out1_1
        out2_2=out1_2
        out2_3=out1_3
        out2_4=out1_4

        n_hour = n_hour+1
        if i%6 == 0:
            print i
            codes_set_values(output_Z500,Z500)   
            codes_write(output_Z500,fout_Z500)

        for j in range(n_inputs/60):
            for k in range(60):
                Z500_2ds[j,k]=Z500[j*60+k]

        state_0[0,60] = Z500_2ds[0,0]
        state_0[1,60] = Z500_2ds[31-1,0]
        for j in range(60):
            state_0[0,j] = Z500_2ds[1,j]
            state_0[1,j] = Z500_2ds[31-2,j]

        n_lon = -1
        for k in range(60):
            n_lon = n_lon+1
            state_1[k,n_input_sets_1-4] = np.mod(4*n_hour,24*1461)/(24.0*1461.0)-0.5
            state_1[k,n_input_sets_1-3] = np.mod(n_hour,24)/24.0-0.5
            state_1[k+60,n_input_sets_1-4] = np.mod(4*n_hour,24*1461)/(24.0*1461.0)-0.5
            state_1[k+60,n_input_sets_1-3] = np.mod(n_hour,24)/24.0-0.5
            state_2[k,n_input_sets_2-4] = np.mod(4*n_hour,24*1461)/(24.0*1461.0)-0.5
            state_2[k,n_input_sets_2-3] = np.mod(n_hour,24)/24.0-0.5
            state_2[k+60,n_input_sets_2-4] = np.mod(4*n_hour,24*1461)/(24.0*1461.0)-0.5
            state_2[k+60,n_input_sets_2-3] = np.mod(n_hour,24)/24.0-0.5
            state_3[k,n_input_sets_3-4] = np.mod(4*n_hour,24*1461)/(24.0*1461.0)-0.5
            state_3[k,n_input_sets_3-3] = np.mod(n_hour,24)/24.0-0.5
            state_3[k+60,n_input_sets_3-4] = np.mod(4*n_hour,24*1461)/(24.0*1461.0)-0.5
            state_3[k+60,n_input_sets_3-3] = np.mod(n_hour,24)/24.0-0.5

            m_count = -1
            for l in range(-1,1+1):
                for m in range(-1,1+1):
                    m_count = m_count+1
                    n_lon=(k+m)%60
                    state_1[k,m_count] = Z500_2ds[1+l,n_lon]
                    state_1[k+60,m_count] = Z500_2ds[31-2+l,n_lon]

            m_count = -1
            for l in range(-2,2+1):
                for m in range(-2,2+1):
                    m_count = m_count+1
                    n_lon=(k+m)%60
                    state_2[k,m_count] = Z500_2ds[2+l,n_lon]
                    state_2[k+60,m_count] = Z500_2ds[31-3+l,n_lon]

            m_count = -1
            for l in range(-3,3+1):
                for m in range(-3,3+1):
                    m_count = m_count+1
                    n_lon=(k+m)%60
                    state_3[k,m_count] = Z500_2ds[3+l,n_lon]
                    state_3[k+60,m_count] = Z500_2ds[31-4+l,n_lon]

        n_count = -1
        for j in range(4,n_inputs/60-4):
            n_lon = -1
            for k in range(60):
                n_lon = n_lon+1
                n_count = n_count+1
                state_4[n_count,n_input_sets_4-4] = np.mod(4*n_hour,24*1461)/(24.0*1461.0)-0.5
                state_4[n_count,n_input_sets_4-3] = np.mod(n_hour,24)/24.0-0.5    
                m_count = -1
                for l in range(-4,4+1):
                    for m in range(-4,4+1):
                        m_count = m_count+1
                        n_lon=(k+m)%60
                        state_4[n_count,m_count] = Z500_2ds[j+l,n_lon]
        state_0[:,:] = 2.0*(state_0[:,:]-min_Z500I_0)/(max_Z500I_0-min_Z500I_0)-1.0
        state_1[:,:n_input_sets_1-4] = 2.0*(state_1[:,:n_input_sets_1-4]-min_Z500I)/(max_Z500I-min_Z500I)-1.0
        state_2[:,:n_input_sets_2-4] = 2.0*(state_2[:,:n_input_sets_2-4]-min_Z500I)/(max_Z500I-min_Z500I)-1.0
        state_3[:,:n_input_sets_3-4] = 2.0*(state_3[:,:n_input_sets_3-4]-min_Z500I)/(max_Z500I-min_Z500I)-1.0
        state_4[:,:n_input_sets_4-4] = 2.0*(state_4[:,:n_input_sets_4-4]-min_Z500I)/(max_Z500I-min_Z500I)-1.0

        out1_0 = model_0.predict(state_0,batch_size=2)
        out1_1 = model_1.predict(state_1,batch_size=2)
        out1_2 = model_2.predict(state_2,batch_size=2)
        out1_3 = model_3.predict(state_3,batch_size=2)
        out1_4 = model_4.predict(state_4,batch_size=n_batch)

        out1_0 = (out1_0+1.0)*(max_Z500O_0-min_Z500O_0)/2.0+min_Z500O_0
        out1_1 = (out1_1+1.0)*(max_Z500O-min_Z500O)/2.0+min_Z500O
        out1_2 = (out1_2+1.0)*(max_Z500O-min_Z500O)/2.0+min_Z500O
        out1_3 = (out1_3+1.0)*(max_Z500O-min_Z500O)/2.0+min_Z500O
        out1_4 = (out1_4+1.0)*(max_Z500O-min_Z500O)/2.0+min_Z500O

        if i==0: 
            out0 = out1_0
            out1 = out1_1
            out2 = out1_2
            out3 = out1_3
            out4 = out1_4

        if i==1: 
            out0 = 1.5*out1_0-0.5*out2_0
            out1 = 1.5*out1_1-0.5*out2_1
            out2 = 1.5*out1_2-0.5*out2_2
            out3 = 1.5*out1_3-0.5*out2_3
            out4 = 1.5*out1_4-0.5*out2_4

        if i>1: 
            out0 = (23.0/12.0)*out1_0-(4.0/3.0)*out2_0+(5.0/12.0)*out3_0
            out1 = (23.0/12.0)*out1_1-(4.0/3.0)*out2_1+(5.0/12.0)*out3_1
            out2 = (23.0/12.0)*out1_2-(4.0/3.0)*out2_2+(5.0/12.0)*out3_2
            out3 = (23.0/12.0)*out1_3-(4.0/3.0)*out2_3+(5.0/12.0)*out3_3
            out4 = (23.0/12.0)*out1_4-(4.0/3.0)*out2_4+(5.0/12.0)*out3_4

        n_count = -1
        for j in range(4,n_inputs/60-4):
            for k in range(60):
                n_count = n_count+1
                Z500[j*60+k]=Z500[j*60+k]+(out4[n_count,0])
  
    codes_release(output_Z500)
    fout_Z500.close()


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

file = open('./input.dat', 'r') 

n_run_str  = file.readline()
n_dummy =  eval(n_run_str)
n_forecasts = n_dummy[0]
n_steps =  n_dummy[1]

print 'Load Network'

model1 = Sequential()
model1.add(Dense(8, input_dim=8, activation='tanh'))
model1.add(Dense(100, activation='tanh'))
model1.add(Dense(100, activation='tanh'))
model1.add(Dense(100, activation='tanh'))
model1.add(Dense(100, activation='tanh'))
model1.add(Dense(8, activation='tanh'))

model1.compile(loss='mean_absolute_error', optimizer='SGD', metrics=['mae'])
model1.load_weights('./weights', by_name=False)

max_train = 30.0
min_train = -20.0


print 'Read reference state'
ref_state = np.zeros((n_forecasts*(n_steps+1)*8))
fore_state = np.zeros((n_forecasts*(n_steps+1),8))
state = np.zeros(8)
state_n = np.zeros((1,8))

out0 = np.zeros((8,1))
out1 = np.zeros((8,1))
out2 = np.zeros((8,1))
out3 = np.zeros((8,1))

data_list_ref = []
for i in range(n_forecasts*(n_steps+1)):
    a_str = file.readline()  
    a = eval('[' + a_str + ']')
    data_list_ref.append(a)        

ref_state = np.array(data_list_ref)

del(data_list_ref)
file.close() 


ref_state = ref_state.reshape((n_forecasts*(n_steps+1), 8))


print 'Perform forecast: ', n_forecasts, n_steps 

for i in range(n_forecasts):    
    state[:] = ref_state[i*(n_steps+1),:]
    fore_state[i*(n_steps+1),:] = state[:]
    for j in range(n_steps):
        out3=out2
        out2=out1
        state_n[0,:] = 2.0*(state-min_train)/(max_train-min_train)-1.0
        out1 = model1.predict(state_n,batch_size=1)
        if j==0: 
            out0 = out1
        if j==1: 
            out0 = 1.5*out1-0.5*out2
        if j>1: 
            out0 = (23.0/12.0)*out1-(4.0/3.0)*out2+(5.0/12.0)*out3
        state[:] = state[:] + out0
        fore_state[i*(n_steps+1)+j+1,:] = state[:]


for j in range(n_steps+1):
    error=0.0
    for i in range(n_forecasts):
        for k in range(8):
            error = error+abs(ref_state[i*(n_steps+1)+j,k]-fore_state[i*(n_steps+1)+j,k])/(8.0*float(n_forecasts))
    time = j*0.005
    print j*0.005, error, ref_state[j,1], fore_state[j,1]


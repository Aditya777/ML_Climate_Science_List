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

file = open('/trainingdata.dat', 'r') 

n_run_str  = file.readline()
n_dummy =  eval(n_run_str)
n_run = n_dummy[0]
n_steps =  n_dummy[1]

n_run = 2001000

print n_run, n_steps 

x_train = np.zeros((n_run)*8)
y_train = np.zeros((n_run)*8)


data_list_i = []
data_list_o = []
for i in range(n_run):
    a_str = file.readline()
    a = eval('[' + a_str + ']')
    data_list_i.append(a)
    a_str = file.readline()
    a = eval('[' + a_str + ']')
    data_list_o.append(a)

x_train = np.array(data_list_i)
y_train = np.array(data_list_o)
y_train[:] = y_train[:] - x_train[:]

del(data_list_i)
del(data_list_o)

x_train = x_train.reshape((n_run, 8))
y_train = y_train.reshape((n_run, 8))

max_train = 30.0
min_train = -20.0

x_train = 2.0*(x_train-min_train)/(max_train-min_train)-1.0

file.close() 


model = Sequential()
model.add(Dense(8, input_dim=8, activation='tanh'))
model.add(Dense(100, activation='tanh'))
model.add(Dense(100, activation='tanh'))
model.add(Dense(100, activation='tanh'))
model.add(Dense(100, activation='tanh'))
model.add(Dense(8, activation='tanh'))

model.compile(loss='mean_absolute_error', optimizer='SGD', metrics=['mae'])

model.fit(x_train, y_train, epochs=200,batch_size=128,validation_split=0.2)
model.save_weights("./weights")



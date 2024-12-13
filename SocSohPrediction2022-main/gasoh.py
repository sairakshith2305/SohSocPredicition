# -*- coding: utf-8 -*-
"""
Created on Sun May 29 22:40:03 2022

@author: home
"""

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from tf.keras.models import load_model
import matplotlib.pyplot as plt

import pyswarms as ps
from pyswarms.utils.plotters import (plot_cost_history)
from keras.models import load_model

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense, TimeDistributed, Activation

from pandas import read_csv
from matplotlib import pyplot
import random
Rand = 42
random.seed(Rand)
np.random.seed(Rand)

from sklearn.preprocessing import MinMaxScaler

dataset = read_csv('soh_1.csv', header=0, index_col=0)
values = dataset.values
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))

y = values[:, -1]
x = values[:,:-1]
x = scaler.fit_transform(x)
y = scaler.fit_transform(np.array(y).reshape(-1,1))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle = False)
x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

y_train.shape

x_train.shape


model = Sequential()
model.add(LSTM(7, return_sequences=False, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dense(1, activation='linear'))

def rmse_nn(y_true, y_pred):
    from keras import backend
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

def r_square_loss(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return 1 - ( 1 - SS_res/(SS_tot + K.epsilon()))


#Neural_Network using GA

modelGA = Sequential()
modelGA.add(LSTM(7, return_sequences=False, input_shape=(x_train.shape[1], x_train.shape[2]), name='layer_1'))
modelGA.add(Dense(1, activation='linear', name='layer_2'))

modelGA.compile(loss="mean_squared_error", optimizer='sgd', metrics=["mean_squared_error"])
modelGA.summary()

def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true), axis=-1))

def getFitness(params):
    layer_1 = modelGA.get_layer('layer_1').get_weights()
    layer_2 = modelGA.get_layer('layer_2').get_weights()
    
    layer1_W1_shape = (layer_1[0].shape[0] * layer_1[0].shape[1])
    layer1_W2_shape = layer1_W1_shape + (layer_1[1].shape[0] * layer_1[1].shape[1])
    layer1_B_shape = layer1_W2_shape + (layer_1[2].shape[0])
    layer2_W_shape = layer1_B_shape + (layer_2[0].shape[0] * layer_2[0].shape[1])
    layer2_B_shape = layer2_W_shape + (layer_2[1].shape[0])

    layer1_W1 = params[0:layer1_W1_shape].reshape(layer_1[0].shape)
    layer1_W2 = params[layer1_W1_shape:layer1_W2_shape].reshape(layer_1[1].shape)
    layer1_B = params[layer1_W2_shape:layer1_B_shape].reshape(layer_1[2].shape)
    layer2_W = params[layer1_B_shape:layer2_W_shape].reshape(layer_2[0].shape)
    layer2_B = params[layer2_W_shape:layer2_B_shape].reshape(layer_2[1].shape)
    
    modelGA.get_layer('layer_1').set_weights([layer1_W1, layer1_W2, layer1_B])
    modelGA.get_layer('layer_2').set_weights([layer2_W, layer2_B])
    
    predY = modelGA.predict(x_train)
    loss = rmse(y_train.reshape(y_train.shape[0]), predY.reshape(y_train.shape[0]))
    return loss

def f(params):
    print('Number of particles: %d' % params.shape[0])
    losses = np.array([getFitness(params[i]) for i in range(params.shape[0])])
    print('List of losses for all particles')
    print(losses)
    return losses


options = {'c1': 0.1, 'c2': 0.9, 'w':0.9}

layer_1 = modelGA.get_layer('layer_1').get_weights()
layer_2 = modelGA.get_layer('layer_2').get_weights()
dimensions = (layer_1[0].shape[0] * layer_1[0].shape[1]) + (layer_1[0].shape[0] * layer_1[0].shape[1]) + (layer_1[1].shape[0] * layer_1[1].shape[1]) + (layer_2[0].shape[0] * layer_2[0].shape[1]) +(layer_2[1].shape[0])

print("Number of params in Neural Network: %d" % dimensions)

optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=dimensions, options=options)

cost, pos = optimizer.optimize(f, iters=50)

plot_cost_history(cost_history=optimizer.cost_history)
plt.show()


layer_1 = modelGA.get_layer('layer_1').get_weights()
layer_2 = modelGA.get_layer('layer_2').get_weights()

layer1_W1_shape = (layer_1[0].shape[0] * layer_1[0].shape[1])
layer1_W2_shape = layer1_W1_shape + (layer_1[1].shape[0] * layer_1[1].shape[1])
layer1_B_shape = layer1_W2_shape + (layer_1[2].shape[0])
layer2_W_shape = layer1_B_shape + (layer_2[0].shape[0] * layer_2[0].shape[1])
layer2_B_shape = layer2_W_shape + (layer_2[1].shape[0])

layer1_W1 = pos[0:layer1_W1_shape].reshape(layer_1[0].shape)
layer1_W2 = pos[layer1_W1_shape:layer1_W2_shape].reshape(layer_1[1].shape)
layer1_B = pos[layer1_W2_shape:layer1_B_shape].reshape(layer_1[2].shape)
layer2_W = pos[layer1_B_shape:layer2_W_shape].reshape(layer_2[0].shape)
layer2_B = pos[layer2_W_shape:layer2_B_shape].reshape(layer_2[1].shape)

modelGA.get_layer('layer_1').set_weights([layer1_W1, layer1_W2, layer1_B])
modelGA.get_layer('layer_2').set_weights([layer2_W, layer2_B])

modelGA.save('modelGASOH.h5')


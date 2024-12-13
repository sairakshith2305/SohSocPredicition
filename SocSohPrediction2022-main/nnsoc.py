# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 20:27:34 2022

@author: home
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyswarms as ps
import random
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
    
from scipy.optimize import differential_evolution
    
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
    
Rand = 42
random.seed(Rand)
np.random.seed(Rand)
    
        
from pandas import read_csv
dataset = read_csv('soc1.csv', header=0, index_col=0)
values = dataset.values
    # ensure all data is float
values = values.astype('float32')
    # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaler_x = MinMaxScaler(feature_range=(0, 1))

y = values[:, -1]
x = values[:,:-1]
df = read_csv('soc1.csv')
X_data=df.drop(['Relative State of Charge'],axis=1)
x_data_orig=X_data[1202:]
x_data_orig=x_data_orig.values.tolist()

x = scaler_x.fit_transform(x)
y = scaler.fit_transform(np.array(y).reshape(-1,1))
    
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle = False)
    
def cal(n_inputs, n_hidden, n_classes):
    i_weights = n_inputs*n_hidden
    i_bias = n_hidden
    h_weights = n_hidden * n_classes
    h_bias = n_classes
    n_params = i_weights + i_bias + h_weights + h_bias
        
    return i_weights, i_bias, h_weights, h_bias, n_params
n_inputs = 6
n_hidden = 20
n_classes = 1
i_weights , i_bias, h_weights, h_bias, n_params = cal(n_inputs, n_hidden, n_classes)
def forward(params):
    w1 = params[:i_weights].reshape((n_inputs, n_hidden))
    b1 = params[i_weights:i_weights+i_bias].reshape((n_hidden,))
    
    w2 = params[i_weights+i_bias:i_weights+i_bias+h_weights].reshape((n_hidden, n_classes))
    b2 = params[i_weights+i_bias+h_weights:].reshape((n_classes,))
        
    z1 = x_train.dot(w1) + b1
    a1 = np.where(z1 > 0, z1, z1*0.01)
    z2 = a1.dot(w2) + b2
        
    loss = mean_squared_error(y_train, z2)
        
    return loss

def f(x):
    n_particles = x.shape[0]
    j = [ forward(x[i]) for i in range(n_particles)]
    
    return np.array(j)
    
def train(options):
    optimizer = ps.single.GlobalBestPSO(n_particles = 100, dimensions = n_params, options = options)
    cost,pos = optimizer.optimize(f, iters = 50)
    
    return cost, pos, optimizer.cost_history
    
def predict(X, pos):
    w1 = pos[:i_weights].reshape((n_inputs, n_hidden))
    b1 = pos[i_weights:i_weights+i_bias].reshape((n_hidden,))
    
    w2 = pos[i_weights+i_bias:i_weights+i_bias+h_weights].reshape((n_hidden, n_classes))
    b2 = pos[i_weights+i_bias+h_weights:].reshape((n_classes,))
        
    z1 = X.dot(w1) + b1
    a1 = np.where(z1 > 0, z1 , z1*0.01)
    z2 = a1.dot(w2)
        
    ypred = z2
    return ypred

checkpoint_state = np.random.get_state()
np.random.set_state(checkpoint_state)
options = {'c1':0.9,'c2':0.1,'w': 0.9}
cost , pos, history = train(options)
y_pred = predict(x_test, pos)
y_pred= scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)
import pickle
file=open("nnsoc.txt","wb")
pickle.dump(y_pred,file)
file.close()


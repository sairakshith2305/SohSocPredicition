# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 20:23:53 2022

@author: home
"""

from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler

from matplotlib import pyplot
# load dataset
dataset = read_csv('soh_1.csv', header=0,index_col=False)
dataset.head()

values = dataset.values

from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
import numpy as np
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

import random
Rand = 42
random.seed(Rand)
np.random.seed(Rand)

dataset = read_csv('soh_1.csv', header=0, index_col=0)
values = dataset.values
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
#scaled = scaler.fit_transform(values)
# frame as supervised learning
y = values[:, -1]
x = values[:,:-1]
x = scaler.fit_transform(x)
y = scaler.fit_transform(np.array(y).reshape(-1,1))
print(x)
print(y)
#c = np.concatenate([x, np.tile(y, (x.shape[0],1))], axis = 1)
result = np.column_stack((x,y))
reframed = series_to_supervised(result, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
print(reframed.head())
reframed
#scaled

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense, TimeDistributed, Activation
reframed = np.array(reframed)
y = reframed[:, -1]
x = reframed[:,:-1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle = False)
x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

# design network

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
model = Sequential()
model.add(LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(x_train, y_train, epochs=20, batch_size=72, validation_data=(x_test, y_test), verbose=2, shuffle=False)

from numpy import concatenate
from math import sqrt
yhat = model.predict(x_test)
x_test = x_test.reshape((x_test.shape[0], x_test.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, x_test[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
y_test = y_test.reshape((len(y_test), 1))
inv_y = concatenate((y_test, x_test[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

import pickle
file=open("mlstmsoh.txt","wb")
pickle.dump(inv_yhat,file)

file.close()
file=open("mlstmsohpred.txt","wb")
pickle.dump(inv_y,file)
file.close()

df = read_csv('soh_1.csv', header=0)

X_data=df.drop(['soh'],axis=1)
X_data=X_data[100:]
x_data_orig=X_data.values.tolist()
print(x_data_orig)

file=open("mlstmsohx.txt","wb")
pickle.dump(x_data_orig,file)
file.close()
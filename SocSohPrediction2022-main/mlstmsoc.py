from pandas import read_csv
from matplotlib import pyplot
# load dataset
dataset = read_csv('soc1.csv', header=0,index_col=False)
dataset.head()

values = dataset.values
# specify columns to plot
groups = [1, 2, 3,4, 5, 6, 7]
i = 1
# plot each column
pyplot.figure()
for group in groups:
	pyplot.subplot(len(groups), 1, i)
	pyplot.plot(values[:, group])
	pyplot.title(dataset.columns[group], y=0.5, loc='right')
	i += 1
pyplot.show()

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

dataset = read_csv('soc1.csv', header=0, index_col=0)
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
reframed.drop(reframed.columns[[8,9,10,11,12,13]], axis=1, inplace=True)
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
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
model = Sequential()
model.add(LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(x_train, y_train, epochs=20, batch_size=72, validation_data=(x_test, y_test), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

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
file=open("mlstmsoc.txt","wb")
pickle.dump(inv_yhat,file)

file.close()
file=open("mlstmsocpred.txt","wb")
pickle.dump(inv_y,file)
file.close()


df = read_csv('soc1.csv')
X_data=df.drop(['Relative State of Charge'],axis=1)
x_data_orig=X_data[1202:]
x_data_orig=x_data_orig.values.tolist()

file=open("mlstmsocx.txt","wb")
pickle.dump(x_data_orig,file)
file.close()


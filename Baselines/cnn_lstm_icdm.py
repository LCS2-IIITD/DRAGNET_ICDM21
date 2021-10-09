import pickle

import numpy as np
import pandas as pd
import os
import json

from tqdm import tqdm

import matplotlib.pyplot as plt
# %matplot

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten

from keras.layers import Dropout
from keras.layers import Input
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.callbacks import EarlyStopping
import keras
from math import sqrt
from matplotlib import pyplot
import seaborn as sns
sns.set_style('dark')
import tensorflow
from scipy import stats

n_steps = 25
n_features = 5
end = 1
n_output = 1

all_windows = pickle.load(open("./time_series.pkl", "rb"))

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps, n_output, start, end):
    X, y = list(), list()
    # sequences = sequences[0]
    # print(len(sequences))
    arr = sequences[0].reshape((sequences[0].shape[0], 1))
    for i in range(1, len(sequences)-1):
        arr = np.hstack((arr, sequences[i].reshape((sequences[0].shape[0], 1))))
    arr = np.vstack((arr, np.zeros((300 - min(300, arr.shape[0]), 5))))
    # print(arr.shape)
    sequences = arr
    LIMIT = min(300, arr.shape[0])
    for i in range(start, LIMIT):
		# find the end of this pattern
        end_ix = i + n_steps
		# check if we are beyond the dataset
        if end_ix + n_output >= LIMIT:
            break
		# gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix: end_ix+n_output, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def createDataset(all_windows, n_steps, end, idx, n_output):
    bigSize = (all_windows[idx][0].shape[0])
    splitSize = int(bigSize)
    X,y = split_sequences(all_windows[idx], n_steps, n_output, 0, splitSize + n_output)
    return X, y

def getLSTMModel(X, y, n_steps=n_steps,n_output=n_output, n_features = 5):
    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    es = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_output*n_features))
    model.compile(optimizer='adam', loss='mse')
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    model.fit(X, y,  epochs=10, verbose=1,batch_size=32, callbacks=[es])
    return model#, history

def getCNNModel(X, y, n_steps=n_steps, n_output=n_output, n_features=5):
    es = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_steps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(n_output*n_features))
    model.compile(optimizer='adam', loss='mse')
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    #history = 
    model.fit(X, y, epochs=5, verbose=1,batch_size=16, callbacks=[es])
    return model#, history

def getPredictions(X_test, model):
    test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
    yhat = model.predict(test, verbose=2)
    return yhat

def runForIDX(idx, model, n_output):
    X, y = createDataset(all_windows, n_steps, end, idx, n_output)
    return X, y

X = []
y = -1
X_test = -1
y_test = -1
X_list = []
y_list = []
X_test_list = []
y_test_list = []
for i in tqdm(range(int(len(all_windows)*0.8))):
    X, y = runForIDX(i, None, 1)
    X_list.append(X[:300])
    y_list.append(y[:300])

X = np.concatenate(X_list)
y = np.concatenate(y_list)

# FOR 
model = getLSTMModel
trained_model = model(X, y, n_steps, n_output, n_features)

model = getCNNModel
trained_model = model(X, y, n_steps, n_output, n_features)

X_test_list = []
y_test_list = []
for i in tqdm(range(int(len(all_windows)*0.8), len(all_windows))):
    X_test, y_test = runForIDX(i, None, 1)
    X_test_list.append(X_test[:300])
    y_test_list.append(y_test[:300])

preds = []
for i in tqdm(range(len(X_test_list))):
    currX = np.array(X_test_list[i][0])
    for j in range(25, 300):
        # for i in range(26, 300):
        X = currX.reshape((1, 25, 5))
        out = trained_model.predict(X)
        preds.append(out)
        currX = np.vstack((currX, out))[-25:]

truth = np.array(y_test_list)[:, :, 0, 0]

def calc_mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calc_smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

rmses = 0
for i in range(690):
    y_preds = preds[i]
    y_actuals = truth[i]
    for j in range(len(y_actuals)):
        if y_actuals[j] == 0:
            break
    rmses += np.sqrt(np.mean(np.array(y_preds[:j]-y_actuals[:j])**2))

SUM = 0
MEANS = 0
for i in range(690):
    for j in range(len(truth[i])):
        if truth[i][j] == 0:
            break
    SUM += stats.pearsonr(preds[i][:j], truth[i][:j])[0]
    MEANS += np.mean(np.abs((preds[i][:j] - truth[i][:j])))

results = (SUM/690, MEANS/690, rmses/690)

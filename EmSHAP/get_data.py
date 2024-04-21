import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import *

def _dataload(path, x_key, y_key):
    data = pd.read_csv(path)
    X = data[x_key].values
    Y = data[y_key].values.reshape(-1, 1)
    del_list = []
    for i in range(Y.shape[0]):
        if Y[i] == 0:
            del_list.append(i)
    X = np.delete(X, del_list, axis=0)
    Y = np.delete(Y, del_list, axis=0)
    return X[:, :], Y[:, :]


def _normalization(norm_method, data):
    if norm_method == 'minmax':
        data_scaler = MinMaxScaler()
        data = data_scaler.fit_transform(data)
    if norm_method == 'zero_mean':
        pass

    return data


def _get_3D_data(data_x, data_y, n_steps):
    xdataset = []
    ydataset = []
    for i in range(n_steps, data_x.shape[0], n_steps):
        xdataset.append(data_x[i-n_steps:i, :])
        ydataset.append(data_y[i, :])

    return np.array(xdataset), np.array(ydataset)

def _get_batch_3D_data(data_x, data_y, batch_size):
    xdataset = []
    ydataset = []
    for i in range(batch_size, data_x.shape[0]+1, batch_size):
        xdataset.append(data_x[i-batch_size:i, :])
        ydataset.append(data_y[i, :])

    return np.array(xdataset), np.array(ydataset)

def _split_data(data_x, data_y, train_ratio, val_ratio, test_ratio):
    batch_n = data_x.shape[0]
    train_x = data_x[:int(batch_n*train_ratio), :]
    train_y = data_y[:int(batch_n*train_ratio), :]

    val_x = data_x[int(batch_n*train_ratio):int(batch_n*train_ratio)+int(batch_n*val_ratio), :]
    val_y = data_y[int(batch_n * train_ratio):int(batch_n * train_ratio) + int(batch_n * val_ratio), :]

    test_x = data_x[int(batch_n * (train_ratio+val_ratio)):, :]
    test_y = data_y[int(batch_n * (train_ratio+val_ratio)):, :]

    return train_x, train_y, val_x, val_y, test_x, test_y

def _split_data_3D(data_x, data_y, train_ratio, val_ratio, test_ratio):
    batch_n = data_x.shape[0]
    train_x = data_x[:int(batch_n*train_ratio), :, :]
    train_y = data_y[:int(batch_n*train_ratio), :]

    val_x = data_x[int(batch_n*train_ratio):int(batch_n*train_ratio)+int(batch_n*val_ratio), :, :]
    val_y = data_y[int(batch_n * train_ratio):int(batch_n * train_ratio) + int(batch_n * val_ratio), :]

    test_x = data_x[int(batch_n * (train_ratio+val_ratio)):, :, :]
    test_y = data_y[int(batch_n * (train_ratio+val_ratio)):, :]

    return train_x, train_y, val_x, val_y, test_x, test_y

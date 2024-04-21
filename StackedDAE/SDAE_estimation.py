import pandas as pd
from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.datasets import mnist
from keras import backend as K
import numpy as np
from ace.GRUenergy import ACEModel
# import matplotlib.pyplot as plt
import os
import h5py
from config import *
import get_data
from tqdm import tqdm
import tensorflow as tf
import time

# build model
class AutoEncoderLayer():
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.predict_dim = 1
        self.build()

    def build(self):
        self.input = Input(shape=(self.input_dim,))
        self.encode_layer = Dense(self.output_dim, activation='tanh')
        self.encoded = self.encode_layer(self.input)
        self.encoder = Model(self.input, self.encoded)

        self.decode_layer = Dense(self.input_dim, activation='tanh')
        self.decoded = self.decode_layer(self.encoded)

        self.autoencoder = Model(self.input, self.decoded)

        self.predict_layer = Dense(self.predict_dim, activation='tanh')
        self.predicted = self.predict_layer(self.encoded)
        self.predict = Model(self.encoded, self.predicted)


# 构建堆叠DAE
class StackedAutoEncoder():
    def __init__(self, layer_list):
        self.layer_list = layer_list
        self.build()

    def build(self):
        out = self.layer_list[0].encoded

        for i in range(1, num_layers - 1):
            out = self.layer_list[i].encode_layer(out)
            pred = self.layer_list[i].predict_layer(out)
        self.model = Model(self.layer_list[0].input, pred)

def PowerSetsBinary(items):
    N = len(items)
    _set = []
    for i in range(2 ** N): #子集个数，每循环一次一个子集
        combo = []
        for j in range(N): #用来判断二进制下标为j的位置数是否为1
            if(i>>j)%2:
                combo.append(items[j])
        # print(combo)
        _set.append(combo)
    return _set

# @ray.remote
def _estimate_v_hat(x, b, y, _estimate_model, _predict_model):

    x_mask_outputs = np.zeros(shape=x.shape)
    for i in tqdm(range(0, x.shape[0], batch_size)):
    # for i in tqdm(range(2)):
        temp = tf.cast(x[i:i+batch_size, :], tf.float32)
        # aa = _estimate_model.sample(temp, b)
        aa, _ = _estimate_model.impute(temp, b)
        x_mask_outputs[i:i+batch_size, :] = aa
    # x_mask_outputs, _ = get_data._get_3D_data(x_mask_outputs, y, batch_size)
    # x, _ = get_data._get_3D_data(x, y, batch_size)
    _y_mask = _predict_model.predict(x_mask_outputs)
    _y_predict = _predict_model.predict(x)

    return _y_mask, _y_predict

# predict model parameter
num_layers = 5
batch_size = 50
origin_dim = 10
h_dim1 = 6
h_dim2 = 4

encoder_1 = AutoEncoderLayer(origin_dim, h_dim1)
encoder_2 = AutoEncoderLayer(h_dim1, h_dim2)
decoder_3 = AutoEncoderLayer(h_dim2, h_dim1)
decoder_4 = AutoEncoderLayer(h_dim1, origin_dim)
autoencoder_list = [encoder_1, encoder_2, decoder_3, decoder_4]
stacked_ae = StackedAutoEncoder(autoencoder_list)

# predict model load
file = h5py.File('./model/SDAEpredict_dia.h5', 'r')
weight = []
for k in range(len(file.keys())):
    weight.append(file['weight' + str(k)][()])
stacked_ae.model.set_weights(weight)
file.close()

# predict model compile
stacked_ae.model.compile()


# data load
data_x, data_y = get_data._dataload(datapath, x_key, y_key)
data_x = get_data._normalization(norm_method, data_x)
data_y = get_data._normalization(norm_method, data_y)
x_train = data_x[:300, :]
y_train = data_y[:300, :]

x_test = data_x[:, :]
y_test = data_y[:, :]

num_feature = data_x.shape[-1]
full_list = list(range(data_x.shape[-1]))
full_set = PowerSetsBinary(full_list)
print(len(full_set))

# estimate model load
estimate_model = ACEModel(num_feature)
file = h5py.File('./model/batch50_{}.h5'.format(301), 'r')
weight = []
for k in range(len(file.keys())):
    weight.append(file['weight' + str(k)][()])
estimate_model.set_weights(weight)
file.close()

y_mask = []
y_predict = []

iinndd = 0
# y_mask = pd.read_csv('./savedata/y_unmask_{}.csv'.format(iinndd), header=None)
# y_mask = y_mask.values

start_time = time.time()
for len_ind in range(iinndd, len(full_set)):
    if (len_ind != 0) and (len_ind != len(full_set) - 1):
        unmask = np.zeros((batch_size, num_feature))
        ind_list = full_set[len_ind]
        unmask[:, ind_list] = 1
        unmask = tf.cast(unmask, tf.float32)
        # asss = stacked_ae.model.predict(x_test[:100, :])
        # temp_unmask, _ = _estimate_v_hat.remote(x_test, unmask, y_test, estimate_model, stacked_ae.model)
        temp_unmask, _ = _estimate_v_hat(x_test, unmask, y_test, estimate_model, stacked_ae.model)
        # y_mask.append(temp_mask.T)

        y_mask = np.append(y_mask, temp_unmask)
        y_mask_reshape = np.array(y_mask).reshape(-1, x_test.shape[-2])
        y_mask2csv = pd.DataFrame(data=np.array(y_mask).reshape(-1, x_test.shape[-2]), columns=None,
                                  index=None)
        y_mask2csv.to_csv('./savedata/y_unmask_whole{}.csv'.format(len_ind), index=False, header=False)

        print(len_ind)
end_time = time.time()
execution_time = end_time - start_time
print("程序运行时间：", execution_time, "秒")
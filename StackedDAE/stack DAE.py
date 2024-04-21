import pandas as pd
from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.datasets import mnist
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from config import *
import get_data

# 指定gpu
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

##### 设置网络参数 #####
epochs_layer = 300
epochs_whole = 300
batch_size = 50
origin_dim = 10
h_dim1 = 6
h_dim2 = 4

# --predict
# epochs_layer = 100
# epochs_whole = 300
# batch_size = 50
# origin_dim = 8
# h_dim1 = 256
# h_dim2 = 512


##### 准备mnist数据 ######
# (x_train, _), (x_test, _) = mnist.load_data(path='mnist.npz')
# x_train = x_train.astype('float32')/255.
# x_test = x_test.astype('float32')/255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# 给数据添加噪声
# noise_factor = 0.2
# x_train_noisy = x_train + noise_factor * np.random.normal(loc = 0.0, scale = 1.0, size = x_train.shape)
# x_test_noisy = x_test + noise_factor * np.random.normal(loc = 0.0, scale = 1.0, size = x_test.shape)
# x_train_noisy = np.clip(x_train_noisy, 0., 1.)
# x_test_noisy = np.clip(x_test_noisy, 0., 1.)

data_x, data_y = get_data._dataload(datapath, x_key, y_key)

data_x = get_data._normalization(norm_method, data_x)
data_y = get_data._normalization(norm_method, data_y)
x_train = data_x[:300, :]
y_train = data_y[:300, :]

x_test = data_x[300:, :]
y_test = data_y[300:, :]
noise_factor = 0.01
x_train_noisy = x_train + noise_factor * np.random.normal(loc = 0.0, scale = 1.0, size = x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc = 0.0, scale = 1.0, size = x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

##### 构建单个autoencoder #####
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

        # # inp = self.layer_list[0].encoded
        # out = self.layer_list[1].predicted
        # # for i in range(1, num_layers - 1):
        # #     out = self.layer_list[i].encode_layer(out)  
        # # pred = Dense(self.layer_list[1].encoded, 1)
        # self.model = Model(self.layer_list[0].input, out)
        # # self.model = Model(self.layer_list[0].input, self.layer_list[2].predict)

def train_layers(encoder_list=None, layer=None, epochs=None, batch_size=None):
    '''
    预训练：逐层训练，当训练第layer个ae时，使用前（layer-1）个ae训练好的encoder的参数
    :param encoder_list:
    :param layer:
    :param epochs:
    :param batch_size:
    :return:
    '''
    # 对前(layer-1)层用已经训练好的参数进行前向计算，ps:第0层没有前置层
    out = x_train_noisy
    origin = x_train
    if layer != 0:
        for i in range(layer):
            # print("encoder weight", str(i), ":", encoder_list[i].encoder.get_weights()[0])
            out = encoder_list[i].encoder.predict(out)

    encoder_list[layer].autoencoder.summary()
    encoder_list[layer].autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # 训练第layer个ae
    encoder_list[layer].autoencoder.fit(
        out,
        origin if layer == 0 else out,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        verbose=2
    )


def train_whole(sae=None, epochs=None, batch_size=None):
    '''
    用预训练好的参数初始化stacked ae的参数，然后进行全局训练优化
    :param model:
    :param epochs:
    :param batch_size:
    :return:
    '''
    # print("stacked sae weights:")
    # print(sae.model.get_weights())
    sae.model.summary()
    sae.model.compile(optimizer='adam', loss='mean_squared_error')
    sae.model.fit(
        x_train_noisy,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        validation_data=(x_test_noisy, y_test),
        verbose=2
    )


# 5层的stacked ae，实际上要使用4个ae，实例化4个ae
num_layers = 5
encoder_1 = AutoEncoderLayer(origin_dim, h_dim1)
encoder_2 = AutoEncoderLayer(h_dim1, h_dim2)
decoder_3 = AutoEncoderLayer(h_dim2, h_dim1)
decoder_4 = AutoEncoderLayer(h_dim1, origin_dim)
autoencoder_list = [encoder_1, encoder_2, decoder_3, decoder_4]

# 按照顺序对每一层进行预训练
print("Pre training:")
for level in range(num_layers - 1):
    print("level:", level)
    train_layers(encoder_list=autoencoder_list, layer=level, epochs=epochs_layer, batch_size=batch_size)


# 用训练好的4个ae构建stacked dae
stacked_ae = StackedAutoEncoder(autoencoder_list)
print("Whole training:")
# 进行全局训练优化
train_whole(sae=stacked_ae, epochs=epochs_whole, batch_size=batch_size)



y_pred = stacked_ae.model.predict(data_x)

data_y2csv = pd.DataFrame(data=np.array(data_y), columns=None, index=None)
data_y2csv.to_csv('./data/data_y_batch50__dia.csv', index=False, header=False)
y_pred2csv = pd.DataFrame(data=np.array(y_pred), columns=None, index=None)
y_pred2csv.to_csv('./data/y_predict_batch50__dia.csv', index=False, header=False)

plt.plot(y_pred, 'r')
plt.plot(data_y, 'k')
plt.show()

file = h5py.File('./model/SDAEpredict_dia.h5', 'w')
weight = stacked_ae.model.get_weights()
for k in range(len(weight)):
    file.create_dataset('weight' + str(k), data=weight[k])
file.close()



# ##### 显示stacked dae重构后的效果 #####
# decoded_imgs = stacked_ae.model.predict(x_test_noisy)
# n = 10
# plt.figure(figsize=(20, 4))
# for i in range(1, n):
#     # 展示原始图像
#     ax = plt.subplot(2, n, i)
#     plt.imshow(x_test_noisy[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     # 展示自编码器重构后的图像
#     ax = plt.subplot(2, n, i + n)
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
# plt.show()
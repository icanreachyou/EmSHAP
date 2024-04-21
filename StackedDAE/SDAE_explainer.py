import matplotlib.pyplot as plt
import pandas as pd
import copy
import numpy as np
import math
import shap
from sklearn.preprocessing import MinMaxScaler
import get_data
from config import *
import seaborn as sns

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

def _explain(data, full_set_data):
    num_feature = data.shape[-1]

    full_set = list(range(num_feature))
    full_set = PowerSetsBinary(full_set)
    feature_phi = np.zeros(shape=data.shape)
    for feature_ind in range(data.shape[-1]):
        remove_set = copy.deepcopy(full_set)
        for remove_ind in range(len(full_set)):
            if feature_ind in full_set[remove_ind]:
                remove_set.remove(full_set[remove_ind])
        # print(remove_set)

        for set_ind in range(len(remove_set)):
            S = remove_set[set_ind]
            S_plus_i = copy.deepcopy(S)
            S_plus_i.append(feature_ind)
            S_plus_i.sort()
            # print("S", S)
            # print("S_plus_i", S_plus_i)
            S_index = full_set.index(S)
            S_plus_i_index = full_set.index(S_plus_i)
            temp = math.factorial(len(S))*(math.factorial(num_feature-len(S)-1))/math.factorial(num_feature)\
                   *(full_set_data[:, S_plus_i_index]-full_set_data[:, S_index])
            feature_phi[:, feature_ind] = feature_phi[:, feature_ind]+temp

    return feature_phi

def plot_stripplot(data):
    num_feature = data.shape[-1]
    y_class = np.zeros(shape=(data.shape[0], 1))
    x_ = data[:, 0]
    for i in range(1, num_feature):
        y_class = np.vstack((y_class, i * np.ones(shape=(data.shape[0], 1))))
        x_ = np.hstack((x_, data[:, i]))
    print(x_.shape, y_class.shape)
    one_ = np.hstack((x_.reshape(-1, 1), y_class))
    # sns.swarmplot(x=fe)
    # sns.stripplot(x=data[:, 0])
    # sns.violinplot(x=one_[:, 1], y=one_[:, 0], hue=one_[:, 0], inner=None)
    sns.stripplot(y=one_[:, 0], x=one_[:, 1], hue=one_[:, 1], jitter=0.1)
    plt.show()

if __name__ == '__main__':
    data_x, data_y = get_data._dataload(datapath, x_key, y_key)
    full = pd.read_csv('./savedata/y_unmask_whole1022.csv', header=None)
    predict = pd.read_csv('./data/y_predict_batch50__dia.csv', header=None)

    full = full.values
    predict = predict.values
    full = full.T
    # full = full[-1500:, :]

    data_scaler = MinMaxScaler()
    data = data_scaler.fit_transform(data_x)
    data = data[0:, :]

    data_scaler_y = MinMaxScaler()
    data_y = data_scaler_y.fit_transform(data_y)

    _mean = np.mean(predict[:, :])*np.ones(shape=predict.shape)
    print(_mean.shape, full.shape, predict.shape)
    full = np.hstack((_mean[:, :], full[:, :]))
    full = np.hstack((full, predict[:, :]))
    print(full.shape)

    for full_ind in range(full.shape[-1]):
        # tempp = data_scaler_y.inverse_transform(full[:, full_ind].reshape(-1, 1))
        tempp = full[:, full_ind].reshape(-1, 1)
        full[:, full_ind] = tempp.reshape(-1, )

    fe = _explain(data[:, :], full[:, :])
    print(fe.shape)

    # plt.plot(full[:, -1])
    # plt.show()

    for i in range(10):
        plt.subplot(5, 2, i+1)
        plt.plot(fe[:, i])
    plt.show()
    print(np.mean(fe, axis=0))

    fe2csv = pd.DataFrame(data=fe, columns=None,
                                       index=None)
    fe2csv.to_csv('./savedata/dynamic_batch50_explainer_dia.csv', index=False, header=False)
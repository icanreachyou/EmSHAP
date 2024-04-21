import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from config import *
import get_data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

data_x, data_y = get_data._dataload(datapath, x_key, y_key)

# real = data_y[batch_size:]

pre = pd.read_csv('./data/y_predict_batch50__dia.csv', header=None)
pre = pre.values
real = pd.read_csv('./data/data_y_batch50__dia.csv', header=None)
real = real.values

data_scaler_y = MinMaxScaler()
data_y_ = data_scaler_y.fit_transform(data_y)

# real = data_scaler_y.inverse_transform(real)
pre = data_scaler_y.inverse_transform(pre)
real = data_scaler_y.inverse_transform(real)
# plt.figure(figsize=(10, 7))
# plt.plot(data_y[2200:3200], color='k')
# plt.plot(pre[2200:3200], color='r')
# plt.show()

# print(real[3050], real[3660], real[4000])

plt.figure(figsize=(10, 7))
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
# plt.plot(real[:442], color='k', marker='o', markerfacecolor='#FFFFFF', markersize=6)
# plt.plot(pre[:442], color='r', marker='*', markerfacecolor='#FFFFFF', markersize=8)
plt.plot(real[300:442], color='k')
plt.plot(pre[300:442], color='r')
plt.tick_params(labelsize=24)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 18}
plt.legend(['real', 'predict'], prop=font1)

plt.xlim(0, 145)
plt.xticks(np.arange(0, 145, 20), ['', '320', '340', '360', '380', '400', '420', '440'])
# plt.xticks(np.arange(1, 6, 1))
# plt.yticks(np.arange(-6, 7, 2))
plt.xlabel('Patient', fontsize=24, fontname='Times New Roman')
plt.ylabel("Disease progression", fontsize=24, fontname='Times New Roman')
# plt.savefig("predict.png", dpi=600, bbox_inches='tight')
plt.show()

print(pre[:300].mean())
print('===================================')
print(np.sqrt(mean_squared_error(real[0:300], pre[0:300])))
print(r2_score(real[0:300], pre[0:300]))
print('===================================')
print(np.sqrt(mean_squared_error(real[300:442], pre[300:442])))
print(r2_score(real[300:442], pre[300:442]))
print('===================================')
print(pre[396], pre[373], pre[428])
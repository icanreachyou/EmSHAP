import numpy as np
from ace.GRUenergy import ACEModel
from tqdm import tqdm
import get_data
from config import *
from ace.masking import get_add_mask_fn, UniformMaskGenerator, DynamicMaskGenerator
import h5py
import pandas as pd
import time

def train_(x, epochs, ):
    his_loss = []
    his_proposal = []
    his_energy = []
    num_feature = x.shape[-1]
    model = ACEModel(num_feature)
    iinndd = 0
    # file = h5py.File('./checkpoint/batch50_{}.h5'.format(iinndd), 'r')
    # weight = []
    # for k in range(len(file.keys())):
    #   weight.append(file['weight'+str(k)][()])
    # model.set_weights(weight)
    # file.close()
    #
    # his_loss = pd.read_csv('./checkpoint/his_loss_batch50_{}.csv'.format(iinndd), header=None)
    # his_loss = his_loss.values
    # his_proposal = pd.read_csv('./checkpoint/his_proposal_batch50_{}.csv'.format(iinndd), header=None)
    # his_proposal = his_proposal.values
    # his_energy = pd.read_csv('./checkpoint/his_energy_batch50_{}.csv'.format(iinndd), header=None)
    # his_energy = his_energy.values
    start_time = time.time()
    for e in tqdm(range(iinndd, epochs)):
      for i in range(x.shape[0]):
          temp = x[i, :, :]
          ########################################################
          # _fn = get_add_mask_fn(UniformMaskGenerator())
          # _, mask_ = _fn(x[i, :, :])
          _fn = get_add_mask_fn(DynamicMaskGenerator(e, epochs, rmin, rmax))
          _, mask_ = _fn(x[i, :, :])
          ########################################################
          train_x = temp
          # lr_schedule = keras.optimizers.schedules.learning_rate_schedule.PolynomialDecay(
          #     0.01, 100, end_learning_rate=1e-7
          # )
          # optimizer = tf.keras.optimizers.Adam(lr_schedule)

          # model.compile(optimizer)
          # history = model.fit(train_x, mask_)
          results = model.train_step([train_x, mask_])
          if i % 10 == 0:
              print("batch_size:{bs}, loss:{resu}".format(bs=i, resu=results['loss']))
      his_loss = np.append(his_loss, results['loss'].numpy())
      his_proposal = np.append(his_proposal, results['proposal_ll'].numpy())
      his_energy = np.append(his_energy, results['energy_ll'].numpy())
      print("epoch:{e}, loss:{results1}, proposal:{results2}, energy:{results3}"
            .format(e=e, results1=his_loss[-1], results2=his_proposal[-1], results3=his_energy[-1]))
      if e % 100 == 0:
          file = h5py.File('./checkpoint/batch50_{}.h5'.format(e + 1), 'w')
          weight = model.get_weights()
          for k in range(len(weight)):
              file.create_dataset('weight' + str(k), data=weight[k])
          file.close()
          his_loss2csv = pd.DataFrame(data=np.array(his_loss), columns=None,
                                      index=None)
          his_loss2csv.to_csv('./checkpoint/his_loss_batch50_{}.csv'.format(e), index=False, header=False)
          his_proposal2csv = pd.DataFrame(data=np.array(his_proposal), columns=None,
                                          index=None)
          his_proposal2csv.to_csv('./checkpoint/his_proposal_batch50_{}.csv'.format(e), index=False, header=False)
          his_energy2csv = pd.DataFrame(data=np.array(his_energy), columns=None,
                                        index=None)
          his_energy2csv.to_csv('./checkpoint/his_energy_batch50_{}.csv'.format(e), index=False, header=False)
    end_time = time.time()
    execution_time = end_time - start_time
    print("程序运行时间：", execution_time, "秒")
    file = h5py.File('./model/{}.h5'.format('GRU-energy_batch50'), 'w')
    weight = model.get_weights()
    for i in range(len(weight)):
        file.create_dataset('weight' + str(i), data=weight[i])
    file.close()
    # checkpoint = tf.train.Checkpoint(model=model)
    # checkpoint.save('F:\\PhD\\streaming Shapley value\\参考程序\\ace-main\\ace-main\\model\\my_ace_model.ckpt')
    # new_model = MyNet()
    # new_model.load_weights('model_weight')
    return his_loss, his_proposal, his_energy

if __name__ == '__main__':

    data_x, data_y = get_data._dataload(datapath, x_key, y_key)
    data_x = get_data._normalization(norm_method, data_x)
    data_y = get_data._normalization(norm_method, data_y)

    # data_x, _ = get_data._get_3D_data(data_x, data_y, batch_size)
    data_x, _ = get_data._get_3D_data(data_x[:300, :], data_y[:300], batch_size)
    print(data_x.shape)
    train_(data_x, epochs)

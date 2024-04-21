datapath = './data/diabetes2.csv'

x_key = ['age', 'sex', 'bmi', 'map', 'tc', 'ldl', 'hdl', 'tch', 'ltg', 'glu']

y_key = ['y']

norm_method = 'minmax'

batch_size = 50
#
epochs = 301

rmin = 0.2

rmax = 0.8
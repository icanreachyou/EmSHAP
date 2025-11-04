import argparse
import torch
import numpy as np
import pandas as pd
from data_provider.data_factory import data_provider
from utils import split_into_blocks
# from zeus.monitor import ZeusMonitor
parser = argparse.ArgumentParser(description='EmSHAP')
import time
from train_models import *
from utils import *
from evaluate import Evaluation
from metric import SICAUC
from tqdm import tqdm
import matplotlib.pyplot as plt
# from networks3 import *
from model import MLP
# model
parser.add_argument('--model', type=str, default='EmSHAP')
parser.add_argument('--energy_network', type=str, default='MLP')
parser.add_argument('--energy_clip', type=float, default=30.0)
parser.add_argument('--sample_num', type=int, default=20)
parser.add_argument('--proposal_network', type=str, default='GRU',
                    help='MLP/GRU/LSTM/BiGRU/AttGRU')
parser.add_argument('--proposal_unit', type=int, default=16)
parser.add_argument('--residual_blocks', type=int, default=4)
parser.add_argument('--hidden_units', type=int, default=128)
parser.add_argument('--context_units', type=int, default=32)
parser.add_argument('--mixture_components', type=int, default=1,
                    help='mixture of Gaussian distribution, 1 for standard Gaussian distribution')
parser.add_argument('--mask_type', type=str, default='DM',
                    help='DM/Bernoulli/Uniform')
parser.add_argument('--use_proposal_mean', type=bool, default=False)
# data
parser.add_argument('--data', type=str, default='ADTI', help='MNIST/TREC/ADTI/ETT')
parser.add_argument('--root_path', type=str, default='./data/')
parser.add_argument('--X_path', type=str, default='adult_X.csv')
parser.add_argument('--y_path', type=str, default='adult_y.csv')
# parser.add_argument('--data', type=str, default='MNIST', help='MNIST/TREC/ADTI/ETT')
# parser.add_argument('--root_path', type=str, default='./data/mnist_data')
# parser.add_argument('--data', type=str, default='TREC', help='MNIST/TREC/ADTI/ETT')
# parser.add_argument('--root_path', type=str, default='./data')
# parser.add_argument('--data', type=str, default='ETT', help='MNIST/TREC/ADTI/ETT')
# parser.add_argument('--root_path', type=str, default='./data')
# parser.add_argument('--data_path', type=str, default='ETTh1.csv')
parser.add_argument('--test_size', type=float, default=0.2)
parser.add_argument('--num_features', type=int, default=12)
parser.add_argument('--num_seq', type=int, default=1)
parser.add_argument('--shuffle_flag', type=bool, default=False)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--drop_last', type=bool, default=False)

# masking
parser.add_argument('--masking_type', type=str, default='dmg', help='dmg/uniform/bernoulli')
parser.add_argument('--masking_rate_min', type=float, default='0.2')
parser.add_argument('--masking_rate_max', type=float, default='0.8')
parser.add_argument('--masking_rate_bernoulli', type=float, default='0.5')

# training
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=50,
                    help='batch size')
parser.add_argument('--cuda', type=str, default=True)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--lradj', type=str, default='type1')
parser.add_argument('--weight_decay', type=float, default=0.005)
parser.add_argument('--dropout', type=float, default=0.05)
parser.add_argument('--if_energy', type=bool, default=False)
parser.add_argument('--if_seq', type=bool, default=False)
parser.add_argument('--patience', type=float, default=5)
parser.add_argument('--seed', default=3407)
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()

args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

train_data, train_loader = data_provider(args, flag='train')
test_data, test_loader = data_provider(args, flag='test')

start_time = time.time()
proposal_network = train_proposal_model(args, train_loader)
print_memory_usage()
end_time = time.time()
run_time = end_time - start_time
print("running time：", run_time, "秒")

# proposal_network = Gru_proposal_network(args).to(args.device)
# proposal_network = torch.load('F:\code\my_python_code\emshap_whole\emshap_adult\model\early_proposal\\best_network.pth')
# proposal_network = torch.load('./model/BiGRU_proposal_network_ADTI.pth')

proposal_network.load_state_dict(torch.load('./model' + '/' +
                                            '{}_proposal_network_{}.pth'.format(args.proposal_network, args.data),
                                            weights_only=True))

# start_time = time.time()
# energy_network = train_energy_model(args, train_loader, proposal_network)
# print_memory_usage()
# end_time = time.time()
# run_time = end_time - start_time
# print("程序运行时间为：", run_time, "秒")
energy_network = None
# predict_model = torch.load('F:\code\my_python_code\emshap_whole\emshap_mnist\model\cnn_model_196patch.pth')
predict_model = torch.load('./model/mlp_model.pth')
# predict_model = torch.load('F:\code\my_python_code\emshap_whole\emshap_ETT\model\predictive_model.pth').to(args.device)
proposal_network.eval()
# energy_network.eval()
predict_model.eval()


eval_emshap = Evaluation(args, proposal_network, energy_network)

shapley_list = []
for i in tqdm(range(0, len(test_data.labels))):
# for i in tqdm(range(0, 100)):
# for i in tqdm(range(0, 1)):
    x_test = test_data.data[i].reshape(args.num_seq, args.num_features)
    start_time = time.time()
    # monitor.begin_window("training")
    # monitor = ZeusMonitor(gpu_indices=[4])
    # monitor.begin_window("training")
    shapley = eval_emshap.brute_force_shapley_sample(predict_model, x_test, 100, test_data, i)
    # measurement = monitor.end_window("training")
    # print(f"Entire training: {measurement.time} s, {measurement.total_energy} J")
    # measurement = monitor.end_window("training")
    # print(f"Entire test: {measurement.time} s, {measurement.total_energy} J")
    end_time = time.time()
    run_time = end_time - start_time
    print("running time：", run_time, "秒")
    print_memory_usage()
    shapley_list.append(shapley)
shapley_list = np.array(shapley_list)
shapley_list = shapley_list[:, 0, :]
# w_save = pd.DataFrame(data=shapley_list, columns=None, index=None)
# w_save.to_csv("./results/{}_emshap_adult.csv".format(args.proposal_network), index=False, header=False)


# plt.imshow(restore_image(test_data.data[lab].cpu().numpy()), cmap="gray")
# plt.show()
# plt.imshow(np.kron(shapley_list.reshape(14, 14), np.ones((2, 2))), cmap='Oranges')
# plt.show()
print('================ Evaluate Done==================')

reference_ = torch.mean(test_data.data, dim=0).unsqueeze(0).to(args.device)
# reference_ = torch.tensor(-1*np.ones((28, 28)), dtype=torch.float32).unsqueeze(0).to(args.device)
metric_ = SICAUC(args, predict_model)
metric_.metric(test_data, shapley_list, reference_)


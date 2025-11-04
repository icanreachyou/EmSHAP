from data_provider.data_loader import Dataset_MNIST, Dataset_TREC, Dataset_ADTI, Dataset_ETT
from torch.utils.data import DataLoader
import torch
data_dict = {
    'MNIST': Dataset_MNIST,
    'TREC': Dataset_TREC,
    'ADTI': Dataset_ADTI,
    'ETT': Dataset_ETT,
    # 'MIMIC': Data_MIMIC,
}

# 数据提供函数
def data_provider(args, flag='train'):
    Data = data_dict[args.data]
    data_set = Data(args, flag=flag)

    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=True if flag == 'train' else False,
        num_workers=args.num_workers,
        drop_last=args.drop_last
    )
    data_set.data = data_set.data.to(args.device)
    data_set.labels = data_set.labels.to(args.device)
    return data_set, data_loader

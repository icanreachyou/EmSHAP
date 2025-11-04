import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from utils import *
from sklearn.utils import shuffle
from gensim.models.keyedvectors import KeyedVectors
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from torch.utils.data import random_split
import ast
class Dataset_MNIST(Dataset):
    def __init__(self, args, flag='train'):
        self.args = args
        self.flag = flag
        self.root_path = args.root_path
        self.__read_data__()

    def __read_data__(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        if self.flag == 'train':
            dataset = datasets.MNIST(root=self.root_path, train=True, transform=transform, download=True)
        else:
            dataset = datasets.MNIST(root=self.root_path, train=False, transform=transform, download=True)

        images, labels = zip(*dataset)

        blocks = []
        self.args.explain_num = 8
        for img in range(len(images)):
            if labels[img] == self.args.explain_num:
                blocks.append(split_into_blocks(images[img]))
        # image_blocks = [split_into_blocks(img) for img in images]
        self.data = torch.tensor(np.array(blocks), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        blocks = self.data[idx]  # [9, block_h, block_w]
        label = self.labels[idx]
        return blocks, label

class Dataset_TREC(Dataset):
    def __init__(self, args, flag='train'):
        """
        args: 配置参数
        flag: 'train', 'dev', 'test'
        """
        self.args = args
        self.flag = flag
        self.data = self.__read_data__()

    def __read_data__(self):
        data = {}

        def read(flag):
            x, y = [], []
            file_path = os.path.join(self.args.root_path, f"TREC_{flag}.txt")
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    y.append(line.split()[0].split(":")[0])
                    x.append(line.split()[1:])
            x, y = shuffle(x, y)
            if flag == "train":
                dev_idx = len(x) // 10
                data["dev_x"], data["dev_y"] = x[:dev_idx], y[:dev_idx]
                data["train_x"], data["train_y"] = x[dev_idx:], y[dev_idx:]
            else:
                data["test_x"], data["test_y"] = x, y

        read("train")
        read("test")

        # 构建词表、类别索引
        data["vocab"] = sorted(list(set([w for sent in data["train_x"] + data["dev_x"] + data["test_x"] for w in sent])))
        data["classes"] = sorted(list(set(data["train_y"])))
        data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}
        data["idx_to_word"] = {i: w for i, w in enumerate(data["vocab"])}

        params = {
            "MAX_SENT_LEN": max([len(sent) for sent in data["train_x"] + data["dev_x"] + data["test_x"]]),
            "BATCH_SIZE": 50,
            "WORD_DIM": 300,
            "VOCAB_SIZE": len(data["vocab"]),
            "CLASS_SIZE": len(data["classes"]),
            "FILTERS": [3, 4, 5],
            "FILTER_NUM": [100, 100, 100],
            "DROPOUT_PROB": 0.5,
            "NORM_LIMIT": 3,
        }

        print("loading word2vec...")
        word_vectors = KeyedVectors.load_word2vec_format("F:\code\my_python_code\emshap_whole\emshap_TREC\GoogleNews-vectors-negative300.bin", binary=True)

        wv_matrix = []
        for i in range(len(data["vocab"])):
            word = data["idx_to_word"][i]
            if word in word_vectors.vocab:
                wv_matrix.append(word_vectors.word_vec(word))
            else:
                wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))

        # one for UNK and one for zero padding
        wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
        wv_matrix.append(np.zeros(300).astype("float32"))
        wv_matrix = np.array(wv_matrix)
        params["WV_MATRIX"] = wv_matrix

        # 处理 flag 数据
        if self.flag == 'train':
            data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"])
            data_x = [[data["word_to_idx"][w] for w in sent] +
                      [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
                      for sent in data["train_x"]]
            data_y = [data["classes"].index(c) for c in data["train_y"]]
            self.data = torch.tensor(np.array(data_x))
            self.labels = torch.tensor(np.array(data_y).reshape(-1, 1))

        else:
            data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"])
            data_x = [[data["word_to_idx"][w] for w in sent] +
                      [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
                      for sent in data["train_x"]]
            data_y = [data["classes"].index(c) for c in data["train_y"]]
            self.data = torch.tensor(np.array(data_x))
            self.labels = torch.tensor(np.array(data_y).reshape(-1, 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class Dataset_ADTI(Dataset):
    def __init__(self, args, flag='train'):
        self.args = args
        self.flag = flag
        self.root_path = args.root_path
        self.X_path = args.X_path
        self.y_path = args.y_path
        self.test_size = args.test_size
        self.seed = args.seed
        self.__read_data__()

    def __read_data__(self):
        # 读取数据
        X = pd.read_csv(os.path.join(self.root_path, self.X_path)).values
        y = pd.read_csv(os.path.join(self.root_path, self.y_path)).values
        data = np.hstack((X, y))
        # 划分训练集和测试集
        train_data, test_data = train_test_split(data, test_size=self.test_size, random_state=self.seed)

        # 标准化（仅对特征）
        self.scaler = StandardScaler()
        self.scaler.fit(train_data[:, :-1])
        train_data[:, :-1] = self.scaler.transform(train_data[:, :-1])
        test_data[:, :-1] = self.scaler.transform(test_data[:, :-1])

        # 根据 flag 选择对应数据集
        if self.flag == 'train':
            self.data = torch.tensor(train_data[:, :-1], dtype=torch.float32)
            self.labels = torch.tensor(train_data[:, -1], dtype=torch.long)
        else:
            self.data = torch.tensor(test_data[:, :-1], dtype=torch.float32)
            self.labels = torch.tensor(test_data[:, -1], dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        labels = self.labels[idx]
        return sample, labels

class Dataset_ETT(Dataset):
    def __init__(self, args, flag='train'):
        """
        args: 配置参数
        flag: 'train' 或 'test'
        """
        self.args = args
        self.flag = flag
        self.__read_data__()

    def __read_data__(self):
        df = pd.read_csv(os.path.join(self.args.root_path, self.args.data_path))
        data = df[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']].values

        data = data[:2000, :]
        train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

        # 归一化 (只用训练集 fit)
        scaler = MinMaxScaler()
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)

        if self.flag == 'train':
            selected_data = train_data
        else:
            selected_data = test_data

        self.data = torch.tensor(selected_data[:, :-1], dtype=torch.float32)
        self.labels = torch.tensor(selected_data[:, -1], dtype=torch.float32)

        self.data = self.data.unsqueeze(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class Dataset_MIMIC(Dataset):
    def __init__(self, args, flag='train'):
        """
        args: 配置参数
        flag: 'train', 'dev', 'test'
        """
        self.args = args
        self.flag = flag
        self.data, self.labels, self.subject_ids, self.ids = self.__read_data__()

    def __read_data__(self):
        data = {}

        csv_path = r"./mimicdata/mimiciii_clean_processed_50.csv"
        self.df = pd.read_csv(csv_path)
        self.df['icd9_diag'] = self.df['icd9_diag'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        self.df['icd9_proc'] = self.df['icd9_proc'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        self.df['target'] = self.df['target'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        # === 加载 vocab ===
        vocab_file = r"./preprocess/vocab.csv"
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = [line.strip() for line in f]
        self.w2ind = {w: i + 1 for i, w in enumerate(vocab)}  # +1 因为0通常留给padding
        self.ind2w = {i: w for w, i in self.w2ind.items()}

        codes = set()
        for val in self.df['target']:
            codes.update(val)
        self.codes = sorted(list(codes))  # 排序，保证一致性
        self.c2ind = {c: i for i, c in enumerate(codes)}
        self.ind2c = {i: c for c, i in self.c2ind.items()}
        self.code2idx = {code: i for i, code in enumerate(codes)}

        model = Word2Vec.load("./preprocess/processed_50.w2v")
        model.wv.save_word2vec_format("./preprocess/processed_50_google_format.bin", binary=True)
        word_vectors = KeyedVectors.load_word2vec_format("./preprocess/processed_50_google_format.bin", binary=True)

        embed_size = word_vectors.vector_size
        embedding_matrix = np.zeros((len(self.w2ind) + 1, embed_size))

        for word, idx in self.w2ind.items():
            if word in word_vectors:
                embedding_matrix[idx] = word_vectors[word]
            else:
                embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embed_size,))

        embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float)
        # train_data, test_data = train_test_split(self.df, test_size=0.2, shuffle=False)
        if self.flag == 'train':
            row = self.df.iloc[idx]
            self.subject_id = row['subject_id']
            self._id = row['_id']
            self.data = torch.tensor(encode_text(row['text'], self.w2ind), dtype=torch.long)
            self.labels = torch.tensor(encode_labels(row['target'], self.code2idx), dtype=torch.float)
        else:
            row = self.df.iloc[idx]
            self.subject_id = row['subject_id']
            self._id = row['_id']
            self.data = torch.tensor(encode_text(row['text'], self.w2ind), dtype=torch.long)
            self.labels = torch.tensor(encode_labels(row['target'], self.code2idx), dtype=torch.float)
        return self.data, self.labels, self.subject_ids, self.ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.subject_ids[idx], self.ids[idx]

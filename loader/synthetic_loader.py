from math import sin, cos
from abc import ABC, abstractmethod
from torch.utils.data import Subset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
import joblib
import sklearn
import numpy as np


# #########################################################################
# 0. Base Loader
# #########################################################################
class BaseLoader(ABC):
    def __init__(self):
        super().__init__()
        self.train_set = None  # must be of type torch.utils.data.Dataset
        self.test_set = None  # must be of type torch.utils.data.Dataset

    @abstractmethod
    def loaders(self,
                batch_size: int,
                shuffle_train=True,
                shuffle_test=False,
                num_workers: int = 0) -> (DataLoader, DataLoader):
        """Implement data loaders of type torch.utils.data.DataLoader for train_set and test_set."""
        pass

    def __repr__(self):
        return self.__class__.__name__


# #########################################################################
# 1. Gaussian9D Dataset
# #########################################################################
class SyntheticDataset(Dataset):
    def __init__(self,
                 filename: str='synthetic',
                 train: int=0,
                 n: int=20000,
                 load_method: int=0,
                 label_normal: tuple=(0,),
                 label_abnormal: tuple=(1,),
                 ratio_abnormal: float=0.1):
        super(Dataset, self).__init__()

        data = joblib.load(f'../data/{filename}.pkl')
        if train == 1: data = data['train']
        if train == 0: data = data['test']

        if load_method == 0:
            # Load only normal data
            n_normal = n
            X = data[label_normal[0]][:n_normal]
            y = np.zeros(len(X))

        elif load_method == 1:
            # Load only abnormal data
            n_abnormal = int(n * ratio_abnormal)
            X = data[label_abnormal[0]][:n_abnormal]
            y = np.ones(len(X))

        elif load_method == 2:
            # Load for both normal and abnormal data
            n_normal = n
            X_normal = data[label_normal[0]][:n_normal]
            y_normal = np.zeros(len(X_normal))

            n_abnormal = int(n * ratio_abnormal)
            X_abnormal = data[label_abnormal[0]][:n_abnormal]
            y_abnormal = np.ones(len(X_abnormal))

            X = np.vstack((X_normal, X_abnormal))
            y = np.hstack((y_normal, y_abnormal))
            X, y = sklearn.utils.shuffle(X, y)

        self.X, self.y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.int32)

    def __getitem__(self, index):
        sample, target = self.X[index], int(self.y[index])
        return sample, target, index

    def __len__(self):
        return len(self.X)


# #########################################################################
# 2. Gaussian9D Loader
# #########################################################################
class SyntheticLoader(BaseLoader):
    def __init__(self,
                 filename: str='synthetic',
                 train: int=0,
                 n: int=20000,
                 load_method: int=0,
                 label_normal: tuple=(0,),
                 label_abnormal: tuple=(1,),
                 ratio_abnormal: float=0.1):
        super().__init__()

        # Get train set
        self.all_set = SyntheticDataset(filename,
                                        train,
                                        n,
                                        load_method,
                                        label_normal,
                                        label_abnormal,
                                        ratio_abnormal)

    def loaders(self,
                batch_size: int=128,
                shuffle: bool=True,
                num_workers: int=0):
        all_loader = DataLoader(dataset=self.all_set,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=False)

        return all_loader

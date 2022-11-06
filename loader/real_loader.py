from pathlib import Path
from math import sin, cos
from abc import ABC, abstractmethod
from torch.utils.data import Subset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
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
# 1. Dataset Class for Real World Datasets
# #########################################################################
class RealDataset(Dataset):
    def __init__(self,
                 root: str='../data',
                 filename: str='satimage',
                 train: int=0,
                 n_normal_train: int=1200,
                 load_method: int=0,
                 threshold_type: int=0,
                 trained_type: int=0,
                 label_normal: tuple=(3,),
                 label_abnormal: tuple=(4,),
                 ratio_abnormal: float=0.1):

        X, y = torch.load(Path(root) / f'{filename}.pkl')
        # Currently do not support multiple labels
        label_normal = int(label_normal[0])
        label_abnormal = int(label_abnormal[0])
        n_abnormal_train = int(n_normal_train * ratio_abnormal)

        # Dataset for training
        if train == 1:
            # Load only normal data
            if load_method == 0:
                X = X[y == label_normal][:n_normal_train]
                y = np.ones(len(X)) * label_normal

            # Load only abnormal data
            elif load_method == 1:
                X, y = None, None
                print('Invalid load method in training!')

            # Load both normal and abnormal data
            elif load_method == 2:
                X_normal = X[y == label_normal][:n_normal_train]
                y_normal = np.ones(len(X_normal)) * label_normal

                if n_abnormal_train >= len(X[y == label_abnormal]):
                    print('Cannot achieve pre-set abnormal ratio.')
                    print('Return 85% abnormal data for training instead!')
                    n_abnormal_train = int(len(X[y == label_abnormal]) * 0.85)

                X_abnormal = X[y == label_abnormal][:n_abnormal_train]
                y_abnormal = np.ones(len(X_abnormal)) * label_abnormal

                X = np.vstack((X_normal, X_abnormal))
                y = np.hstack((y_normal, y_abnormal))
                X, y = sklearn.utils.shuffle(X, y)

        # Dataset for test
        else:
            # Load only normal data
            if load_method == 0:
                # Use the test part of normal data to set threshold
                if threshold_type == 0:
                    X = X[y == label_normal][n_normal_train:]
                # Use both training & test part to set threshold
                elif threshold_type == 1:
                    X = X[y == label_normal][:]
                y = np.ones(len(X)) * label_normal

            # Load only abnormal data
            elif load_method == 1:
                # When test abnormal comes from a different class
                if trained_type == 0:
                    X = X[y == label_abnormal][:]

                # When test abnormal comes from the same class
                elif trained_type == 1:
                    if n_abnormal_train >= len(X[y == label_abnormal]):
                        print('Cannot achieve pre-set abnormal ratio.')
                        print('Return 15% abnormal data for testing instead!')
                        n_abnormal_train = int(len(X[y == label_abnormal]) * 0.85)
                    X = X[y == label_abnormal][n_abnormal_train:]
                y = np.ones(len(X)) * label_abnormal

            # Load both normal and abnormal data
            elif load_method == 2:
                X, y = None, None
                print('Invalid load method in testing!')

        self.X, self.y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.int32)

    def __getitem__(self, index):
        sample, target = self.X[index], int(self.y[index])
        return sample, target, index

    def __len__(self):
        return len(self.X)


# #########################################################################
# 2. Loader for RealWorld Datasets
# #########################################################################
class RealLoader(BaseLoader):
    def __init__(self,
                 root: str='../data',
                 filename: str='satimage',
                 train: int=0,
                 n_normal_train: int=1200,
                 load_method: int=0,
                 threshold_type: int=0,
                 trained_type: int=0,
                 label_normal: tuple=(3,),
                 label_abnormal: tuple=(4,),
                 ratio_abnormal: float=0.1):
        super().__init__()

        # Get train set
        self.all_set = RealDataset(root,
                                   filename,
                                   train,
                                   n_normal_train,
                                   load_method,
                                   threshold_type,
                                   trained_type,
                                   label_normal,
                                   label_abnormal,
                                   ratio_abnormal)

    def loaders(self,
                batch_size: int=32,
                shuffle: bool=True,
                num_workers: int=0):
        all_loader = DataLoader(dataset=self.all_set,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=False)

        return all_loader

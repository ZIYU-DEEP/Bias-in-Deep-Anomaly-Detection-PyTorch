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
# Helper Functions
# #########################################################################
def sample_(X, n):
    """
    Generate a random sample from X.

    X (np.array): any array
    n (int): the number of the sample needed
    """
    np.random.seed(np.random.randint(0, np.iinfo(np.int32).max))
    index = np.random.choice(X.shape[0], n, replace=False)
    return X[index]


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
class RealPacDataset(Dataset):
    def __init__(self,
                 filename: str='satimage',
                 train: int=0,
                 n_normal_train: int=300,
                 n_normal_val: int=10,
                 n_normal_test: int=208,
                 load_method: int=0,
                 label_normal: tuple=(7,),
                 label_abnormal: tuple=(4,),
                 ratio_abnormal_train: float=0.1,
                 ratio_abnormal_val: float=0.01):

        X, y = joblib.load(f'../../data/{filename}.pkl')
        # Currently do not support multiple labels
        label_normal = int(label_normal[0])
        label_abnormal = int(label_abnormal[0])
        n_abnormal_train = int(np.ceil(n_normal_train * ratio_abnormal_train))
        n_abnormal_test = int(np.ceil(n_normal_test * ratio_abnormal_val))
        n_abnormal_val = int(np.ceil(n_normal_val * ratio_abnormal_val))

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

                X_abnormal = X[y == label_abnormal][:n_abnormal_train]
                y_abnormal = np.ones(len(X_abnormal)) * label_abnormal

                X = np.vstack((X_normal, X_abnormal))
                y = np.hstack((y_normal, y_abnormal))

        # Dataset for test
        if train == 0:
            # Load only normal data
            if load_method == 0:
                X = X[y == label_normal][n_normal_train:n_normal_train + n_normal_test]
                y = np.ones(len(X)) * label_normal

            # Load only abnormal data
            elif load_method == 1:
                X = X[y == label_abnormal][n_abnormal_train:n_abnormal_train + n_abnormal_test]
                y = np.ones(len(X)) * label_abnormal

            # Load both normal and abnormal data
            elif load_method == 2:
                X_normal = X[y == label_normal][n_normal_train:n_normal_train + n_normal_test]
                y_normal = np.ones(len(X_normal)) * label_normal

                X_abnormal = X[y == label_abnormal][n_abnormal_train:n_abnormal_train + n_abnormal_test]
                y_abnormal = np.ones(len(X_abnormal)) * label_abnormal

                X = np.vstack((X_normal, X_abnormal))
                y = np.hstack((y_normal, y_abnormal))

        # Dataset for validation
        if train == - 1:
            # Load only normal data
            if load_method == 0:
                X = sample_(X[y == label_normal][n_normal_train + n_normal_test:], n_normal_val)
                y = np.ones(len(X)) * label_normal

            # Load only abnormal data
            elif load_method == 1:
                X = sample_(X[y == label_abnormal][n_abnormal_train + n_abnormal_test:], n_abnormal_val)
                y = np.ones(len(X)) * label_abnormal

            # Load both normal and abnormal data
            elif load_method == 2:
                X_normal = sample_(X[y == label_normal][n_normal_train + n_normal_test:], n_normal_val)
                y_normal = np.ones(len(X_normal)) * label_normal

                X_abnormal = sample_(X[y == label_abnormal][n_abnormal_train + n_abnormal_test:], n_abnormal_val)
                y_abnormal = np.ones(len(X_abnormal)) * label_abnormal

                X = np.vstack((X_normal, X_abnormal))
                y = np.hstack((y_normal, y_abnormal))

        self.X, self.y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.int32)

    def __getitem__(self, index):
        sample, target = self.X[index], int(self.y[index])
        return sample, target, index

    def __len__(self):
        return len(self.X)


# #########################################################################
# 2. Loader for RealWorld Datasets
# #########################################################################
class RealPacLoader(BaseLoader):
    def __init__(self,
                 filename: str='satimage',
                 train: int=0,
                 n_normal_train: int=300,
                 n_normal_val: int=10,
                 n_normal_test: int=208,
                 load_method: int=0,
                 label_normal: tuple=(7,),
                 label_abnormal: tuple=(4,),
                 ratio_abnormal_train: float=0.1,
                 ratio_abnormal_val: float=0.01):
        super().__init__()

        # Get train set
        self.all_set = RealPacDataset(filename,
                                      train,
                                      n_normal_train,
                                      n_normal_val,
                                      n_normal_test,
                                      load_method,
                                      label_normal,
                                      label_abnormal,
                                      ratio_abnormal_train,
                                      ratio_abnormal_val)

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

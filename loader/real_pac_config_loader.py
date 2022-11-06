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
class RealPacConfigDataset(Dataset):
    def __init__(self,
                 filename: str='satimage',
                 config: str='config_3',
                 train: int=0,
                 n_normal_val: int=100,
                 load_method: int=0,
                 label_normal: tuple=(0,),
                 label_abnormal: tuple=(1,),
                 ratio_abnormal_train: float=0.1,
                 ratio_abnormal_val: float=0.01):
        """
        Be sure that the label_normal and label_abnormal are designated;
        We consider normal to be 0 and abnormal to be 1.
        And we replace the original label as such.
        """
        label_normal = int(label_normal[0])
        label_abnormal = int(label_abnormal[0])

        root = f'../../data/{filename}/{config}'

        X_train_normal, y_train_normal = joblib.load(f'{root}/normal_train.pkl')
        X_val_normal, y_val_normal = joblib.load(f'{root}/normal_val.pkl')
        X_test_normal, y_test_normal = joblib.load(f'{root}/normal_test.pkl')

        X_train_abnormal, y_train_abnormal = joblib.load(f'{root}/abnormal_train.pkl')
        X_val_abnormal, y_val_abnormal = joblib.load(f'{root}/abnormal_val.pkl')
        X_test_abnormal, y_test_abnormal = joblib.load(f'{root}/abnormal_test.pkl')

        n_normal_train = len(X_train_normal)
        n_normal_test = len(X_test_normal)

        n_abnormal_train = int(np.ceil(n_normal_train * ratio_abnormal_train))
        n_abnormal_test = int(np.ceil(n_normal_test * ratio_abnormal_val))
        n_abnormal_val = int(np.ceil(n_normal_val * ratio_abnormal_val))

        # Dataset for training
        if train == 1:
            # Load only normal data
            if load_method == 0:
                X = X_train_normal
                y = np.ones(len(X)) * label_normal

            # Load only abnormal data
            elif load_method == 1:
                X, y = None, None
                print('Invalid load method in training!')

            # Load both normal and abnormal data
            elif load_method == 2:
                X_normal = X_train_normal
                y_normal = np.ones(len(X_normal)) * label_normal

                X_abnormal = X_train_abnormal[:n_abnormal_train]
                y_abnormal = np.ones(len(X_abnormal)) * label_abnormal

                X = np.vstack((X_normal, X_abnormal))
                y = np.hstack((y_normal, y_abnormal))

        # Dataset for validation
        elif train == - 1:
            # Load only normal data
            if load_method == 0:
                X = sample_(X_val_normal, n_normal_val)
                y = np.ones(len(X)) * label_normal

            # Load only abnormal data
            elif load_method == 1:
                X = sample_(X_val_abnormal, n_abnormal_val)
                y = np.ones(len(X)) * label_abnormal

            # Load both normal and abnormal data
            elif load_method == 2:
                X_normal = sample_(X_val_normal, n_normal_val)
                y_normal = np.ones(len(X_normal)) * label_normal

                X_abnormal = sample_(X_val_abnormal, n_abnormal_val)
                y_abnormal = np.ones(len(X_abnormal)) * label_abnormal

                X = np.vstack((X_normal, X_abnormal))
                y = np.hstack((y_normal, y_abnormal))

        # Dataset for test
        else:
            # Load only normal data
            if load_method == 0:
                X = X_test_normal
                y = np.ones(len(X)) * label_normal

            # Load only abnormal data
            elif load_method == 1:
                X = X_test_abnormal[:n_abnormal_test]
                y = np.ones(len(X)) * label_abnormal

            # Load both normal and abnormal data
            elif load_method == 2:
                X_normal = X_test_normal
                y_normal = np.ones(len(X_normal)) * label_normal

                X_abnormal = X_test_abnormal[:n_abnormal_test]
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
class RealPacConfigLoader(BaseLoader):
    def __init__(self,
                 filename: str='satimage',
                 config: str='config_3',
                 train: int=0,
                 n_normal_val: int=100,
                 load_method: int=0,
                 label_normal: tuple=(0,),
                 label_abnormal: tuple=(1,),
                 ratio_abnormal_train: float=0.1,
                 ratio_abnormal_val: float=0.01):
        super().__init__()

        # Get train set
        self.all_set = RealPacConfigDataset(filename,
                                            config,
                                            train,
                                            n_normal_val,
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
                                shuffle=False,
                                num_workers=num_workers,
                                drop_last=False)

        return all_loader

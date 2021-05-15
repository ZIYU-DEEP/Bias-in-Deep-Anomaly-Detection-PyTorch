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
def gen_gaussian9D(random_state,
                   n: int=5000,
                   mix: bool=True,
                   ratio_abnormal: float=0.01,
                   n_features: int=9):
    """
    Generate Gaussian synthetic dataset with normal and/or abnormal data.

    The normal data is generated from a 9-dimensional normal distribution,
    where each dimension follows a N(0, 1).

    The abnormal data is generated under this specific rule:
      - w.p. 0.4, 3 of the 9 dimensions are generated as N(3, 1).
      - w.p. 0.6, 4 of the 9 dimensions are generated as N(3, 1).

    Inputs:
        random_state: (int) the seed to generate data. Be cautious to use
                      different state for training, val and test;
        n: (int) the number of *normal* examples in the set;
        mix: (bool) an indicator for normal only or both normal & abnormal;
        ratio_abnormal: (float) if mix, the ratio of n_abnormal to n_total;

    Returns:
        X: (np.array)
        y: (np.array)
    """
    print('Loading data...')
    np.random.seed(random_state)

    n_normal = n
    n_abnormal = int((ratio_abnormal / (1 - ratio_abnormal)) * n)

    # Generate for normal data
    X_normal = np.random.normal(0, 1, n_normal)
    for _ in range(n_features - 1):
        X_normal_i = np.random.normal(0, 1, n_normal)
        X_normal = np.c_[X_normal, X_normal_i]
    y_normal = np.zeros(X_normal.shape[0])
    print('Normal data loaded...')

    # Generate for abnormal data, first get the base data of N(0, 1)
    if not mix:
        X, y = X_normal, y_normal
    else:
        print('mix:', mix)
        X_abnormal = np.random.normal(0, 1, (n_abnormal, n_features))
        ind_choice = np.random.choice([True, False], (n_abnormal,), p=(0.6, 0.4))

        print('X_abnormal shape:', X_abnormal.shape)
        print('X_normal shape:', X_normal.shape)

        # Get the mask, whose each line indicates which one is abnormal
        def rand_bin_array(shape, k):
            arr = np.zeros(shape); arr[:, :k] = 1
            np.apply_along_axis(np.random.shuffle, 1, arr)
            return arr

        if sum(ind_choice) > 0:
            mask = rand_bin_array((sum(ind_choice), n_features), 3).astype(bool)
            if n_abnormal - sum(ind_choice) > 0:
                mask_ = rand_bin_array((n_abnormal - sum(ind_choice), n_features), 4).astype(bool)
                mask = np.r_[mask, mask_]

        elif n_abnormal - sum(ind_choice) > 0:
            mask = rand_bin_array((n_abnormal - sum(ind_choice), n_features), 4).astype(bool)

        # Replace normal values with abnormal values
        abnormal_arr = np.random.normal(3, 1, (n_abnormal, n_features))
        X_abnormal[mask] = abnormal_arr[mask]
        y_abnormal = np.ones(len(X_abnormal))
        print('Abnormal data loaded...')

        # Concatenate to get the full data
        X = np.vstack((X_normal, X_abnormal))
        y = np.hstack((y_normal, y_abnormal))

    X, y = sklearn.utils.shuffle(X, y)
    print('Concatenated and shuffled...')

    return X, y


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
class Gaussian9DDataset(Dataset):
    def __init__(self,
                 random_state,
                 n: int=5000,
                 mix: bool=True,
                 ratio_abnormal: float=0.01,
                 n_features: int=9):
        super(Dataset, self).__init__()

        # Get the data for training and test
        X, y = gen_gaussian9D(random_state,
                              n,
                              mix,
                              ratio_abnormal,
                              n_features)
        self.X, self.y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.int32)

    def __getitem__(self, index):
        sample, target = self.X[index], int(self.y[index])
        return sample, target, index

    def __len__(self):
        return len(self.X)


# #########################################################################
# 2. Gaussian9D Loader
# #########################################################################
class Gaussian9DLoader(BaseLoader):
    def __init__(self,
                 random_state,
                 n: int=5000,
                 mix: bool=True,
                 ratio_abnormal: float=0.01,
                 n_features: int=9):
        super().__init__()

        # Get train set
        self.all_set = Gaussian9DDataset(random_state,
                                         n,
                                         mix,
                                         ratio_abnormal,
                                         n_features)

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

"""
Title: fmnist_loader.py
Description: The loader classes for the FashionMNIST datasets
"""

from PIL import Image
from abc import ABC, abstractmethod
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torchvision.datasets import FashionMNIST

import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms


# #########################################################################
# 1. Base Dataset
# #########################################################################
class BaseDataset(ABC):
    def __init__(self, root: str):
        super().__init__()

        self.root = root
        self.label_normal = ()
        self.label_abnormal = ()
        self.all_set = None

    @abstractmethod
    def loaders(self,
                batch_size: int=128,
                shuffle=True,
                num_workers: int = 0):
        pass

    def __repr__(self):
        return self.__class__.__name__


# #########################################################################
# 2. FashionMNIST Dataset
# #########################################################################
class FashionMNISTDataset(FashionMNIST):
    """
    Add an index to get item.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
        transform = transforms.ToTensor()
        img = transform(img)
        return img, int(target), index


# #########################################################################
# 3. FashionMNIST Loader for Training
# #########################################################################
class FashionMNISTLoader(BaseDataset):
    def __init__(self,
                 root: str='/net/leksai/data/',
                 filename: str='FashionMNIST',
                 train: int=0,
                 load_method: int=0,
                 label_normal: tuple=(0,),
                 label_abnormal: tuple=(1,),
                 ratio_abnormal: float=0.1,
                 random_state: int=42):
        super().__init__(root)

        # Initialization
        self.root = root + filename
        self.label_normal = label_normal
        self.label_abnormal = label_abnormal
        self.ratio_abnormal = ratio_abnormal

        # Read in initial Full Set
        print('Loading dataset for you!')
        all_set = FashionMNISTDataset(root=self.root, train=train, transform=transforms.ToTensor(), download=True)

        # Get the labels for classes intended to use
        y = all_set.targets.cpu().data.numpy()

        # Get the indices for classes intended to use
        idx = self.get_idx(train, load_method, y, label_normal,
                           label_abnormal, ratio_abnormal, random_state)

        # Get the subset
        self.all_set = Subset(all_set, idx)

    def get_idx(self,
                train,
                load_method,
                y,
                label_normal,
                label_abnormal,
                ratio_abnormal,
                random_state):
        """
        Creat a numpy list of indices of label_ in labels.
        Inputs:
            y (np.array): dataset.targets.cpu().data.numpy()
            label_normal (tuple): e.g. (0,)
            label_abnormal (tuple): e.g. (1,)
            ratio_abnormal (float): e.g. 0.1
            train (bool): True / False
        """
        np.random.seed(random_state)

        idx_normal = np.argwhere(np.isin(y, label_normal)).flatten()
        idx_abnormal = np.argwhere(np.isin(y, label_abnormal)).flatten()

        # Load only for normal data
        if load_method == 0:
            return idx_normal

        # Load only for abnormal data
        if load_method == 1:
            if train:
                print('Invalid load method in training!')
                return None
            else:
                return idx_abnormal

        # Load for both normal and abnormal data
        if load_method == 2:
            if train:
                # Only load designated number of abnormal data in training
                np.random.shuffle(idx_abnormal)
                idx_abnormal_train = idx_abnormal[:int(len(idx_normal) * ratio_abnormal)]
                idx_all = np.hstack((idx_normal, idx_abnormal_train))
                return idx_all
            else:
                print('Invalid load method in testing!')
                return None

    def loaders(self,
                batch_size: int=128,
                shuffle=True,
                num_workers: int = 0) -> (DataLoader, DataLoader):
        all_loader = DataLoader(dataset=self.all_set,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=False)

        return all_loader

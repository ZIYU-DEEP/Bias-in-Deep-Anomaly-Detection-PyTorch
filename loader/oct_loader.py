from PIL import Image
from abc import ABC, abstractmethod
from sklearn.utils import shuffle
from torch.utils.data import Subset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR10

import os
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms

# Be sure that the normal label is 3
idx_to_class= {0: 'CNV',
               1: 'DME',
               2: 'DRUSEN',
               3: 'NORMAL'}

transform_test = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor()])

transform_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()])

# #########################################################################
# 0. Helper Classes and Functions
# #########################################################################
class OCTFolder(ImageFolder):
    """
    A dataset object for Retinal OCT.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)

        if self.transform is None:
            transform = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor()])
        else:
            transform = self.transform

        img = transform(img)
        return img, int(target), index


# #########################################################################
# 1. Base Loader
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
# 2. Loader for Retinal OCT Dataset
# #########################################################################
class OCTLoader(BaseDataset):
    def __init__(self,
                 root: str='../../data/OCT2017/train/',
                 train: int=0,
                 n_normal_train: int=20000,
                 load_method: int=0,
                 trained_type: int=0,
                 label_normal: tuple=(3,),
                 label_abnormal: tuple=(2,),
                 ratio_abnormal: float=0.1,
                 random_state: int=42):
        super().__init__(root)

        # Initialization
        self.root = root
        self.label_normal = label_normal
        self.label_abnormal = label_abnormal
        self.ratio_abnormal = ratio_abnormal

        # Set transform type
        if train:
            transform_ = transform_train
        else:
            transform_ = transform_test

        # Read in full set
        print("Loading for the full data.")
        all_set = OCTFolder(root=root, transform=transform_)

        # Get the labels for the full datasets
        y = np.array(all_set.targets)

        # Get the indices for classes intended to use
        idx = self.get_idx(train, n_normal_train, load_method, y,
                           label_normal, label_abnormal,
                           trained_type, ratio_abnormal, random_state)

        # Get the subset
        self.all_set = Subset(all_set, idx)

    def get_idx(self,
                train,
                n_normal_train,
                load_method,
                y,
                label_normal,
                label_abnormal,
                trained_type,
                ratio_abnormal,
                random_state):
        """
        Creat a numpy list of indices of label_ in labels.
        Inputs:
            y (np.array): dataset.targets.cpu().data.numpy()
            label_normal (tuple): e.g. (3,)
            label_abnormal (tuple): e.g. (1,)
            ratio_abnormal (float): e.g. 0.1
            train (int): 0 or 1
            trained_type (int): indicates if the anomaly is used to train
        """
        np.random.seed(random_state)

        idx_normal = np.argwhere(np.isin(y, label_normal)).flatten()
        idx_abnormal = np.argwhere(np.isin(y, label_abnormal)).flatten()

        np.random.shuffle(idx_normal)
        np.random.shuffle(idx_abnormal)

        idx_normal_train = idx_normal[:n_normal_train]
        idx_normal_test = idx_normal[n_normal_train:]
        idx_abnormal_train = idx_abnormal[:int(len(idx_normal) * ratio_abnormal)]
        idx_abnormal_test = idx_abnormal[int(len(idx_normal) * ratio_abnormal):]

        if train:
            if load_method == 0:
                print('Loading normal data for training!')
                return idx_normal_train

            if load_method == 1:
                print('Invalid load method in training!')
                return None

            if load_method == 2:
                print('Loading normal and abnormal data for training!')
                idx_all = np.hstack((idx_normal_train, idx_abnormal_train))
                return idx_all

        else:
            if load_method == 0:
                print('Loading normal data for testing!')
                return idx_normal_test

            if load_method == 1:
                print('Loading abnormal data for testing!')
                if trained_type == 1:
                    return idx_abnormal_test

                elif trained_type == 0:
                    return idx_abnormal

            if load_method == 2:
                print('Invalid load method in test!')
                return None

    def loaders(self,
                batch_size: int=512,
                shuffle=True,
                num_workers: int = 0) -> (DataLoader, DataLoader):
        all_loader = DataLoader(dataset=self.all_set,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=False)

        return all_loader

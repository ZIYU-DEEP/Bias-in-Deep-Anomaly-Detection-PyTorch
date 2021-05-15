"""
Title: imagenet_loader.py
Description: The loader classes for the ImageNet datasets
"""

from PIL import Image
from abc import ABC, abstractmethod
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torchvision.datasets import FashionMNIST
from torchvision.datasets import ImageNet
from torchvision.datasets import ImageFolder
from sklearn.utils import shuffle

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
# 2. ImageNet Dataset
# #########################################################################
class ImageNetDataset(ImageFolder):
    """
    Add an index to get item.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406),
                                                             (0.229, 0.224, 0.225))])
        img = transform(img)
        return img, int(target), index


# #########################################################################
# 3. ImageNet Loader for Training
# #########################################################################
class ImageNetLoader(BaseDataset):
    def __init__(self,
                 root: str='/mnt/data/huiyingli/imagenet/data/train',
                 train: int=0,
                 load_method: int=0,
                 threshold_type: int=0,
                 trained_type: int=0,
                 label_normal: tuple=(0,),
                 label_abnormal: tuple=(1,),
                 ratio_abnormal: float=0.1,
                 random_state: int=42):
        super().__init__(root)

        # Initialization
        self.root = root
        self.label_normal = label_normal
        self.label_abnormal = label_abnormal
        self.ratio_abnormal = ratio_abnormal


        # Read in initial Full Set
        print('Loading dataset for you!')
        all_set = ImageNetDataset(root=root)

        # Get the labels for classes intended to use
        y = np.array(all_set.targets)

        # Get the indices for classes intended to use
        idx = self.get_idx(train, load_method, threshold_type, trained_type, y,
                           label_normal, label_abnormal, ratio_abnormal, random_state)
        print(f'Total number of idx: {len(idx)}.')

        # Get the subset
        self.all_set = Subset(all_set, idx)

    def get_idx(self,
                train,
                load_method,
                threshold_type,
                trained_type,
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
        idx_normal = np.argwhere(np.isin(y, label_normal)).flatten()
        idx_abnormal = np.argwhere(np.isin(y, label_abnormal)).flatten()

        idx_normal = shuffle(idx_normal, random_state=random_state)
        idx_abnormal = shuffle(idx_abnormal, random_state=random_state)

        idx_normal_train = idx_normal[:int(len(idx_normal) * 0.85)]
        idx_normal_test = idx_normal[int(len(idx_normal) * 0.85):]
        idx_abnormal_train = idx_abnormal[:int(len(idx_normal_train) * ratio_abnormal)]
        idx_abnormal_test = idx_abnormal[int(len(idx_normal_train) * ratio_abnormal):]

        if train:
            if load_method == 0:
                return idx_normal_train

            if load_method == 1:
                print('Invalid load method in training!')
                return None

            elif load_method == 2:
                idx_all = np.hstack((idx_normal_train, idx_abnormal_train))
                idx_all = shuffle(idx_all, random_state=random_state)
                return idx_all
        else:
            if load_method == 0:
                if threshold_type == 0:
                    return idx_normal_test
                elif threshold_type == 1:
                    return idx_normal

            if load_method == 1:
                if trained_type == 0:
                    return idx_abnormal
                else:
                    return idx_abnormal_test

            elif load_method == 2:
                print('Invalid load method in test!')
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

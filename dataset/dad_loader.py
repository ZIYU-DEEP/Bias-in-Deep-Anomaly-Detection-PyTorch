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

idx_to_class= {0: 'normal_driving',
               1: 'adjusting_radio',
               2: 'reaching_behind',
               3: 'talking_with_passenger',
               4: 'messaging_left',
               5: 'messaging_right',
               6: 'drinking',
               7: 'talking_with_phone_left',
               8: 'talking_with_phone_right'}

tester_list = ['Tester1', 'Tester2', 'Tester10', 'Tester11', 'Tester12',
               'Tester13', 'Tester14', 'Tester15', 'Tester16', 'Tester17',
               'Tester18', 'Tester19']

# #########################################################################
# 0. Helper Classes and Functions
# #########################################################################
class Normalize(object):
    """
    github.com/okankop/Driver-Anomaly-Detection/blob/master/spatial_transforms.py

    Normalize an tensor image with mean and standard deviation.

    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std

    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

    def randomize_parameters(self):
        pass


class DADFolder(ImageFolder):
    """
    A dataset object for DAD.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        Normalize([0], [1])])
        img = transform(img)
        return img, int(target), index


class DADSubset(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        label (int) : the label which you will be given all the subset
    """
    def __init__(self, dataset, indices, label):
        self.dataset = dataset
        self.indices = indices
        self.targets = torch.ones(len(dataset)) * label

    def __getitem__(self, index):
        img = self.dataset[self.indices[index]][0]
        target = self.targets[self.indices[index]]
        return img, int(target), index

    def __len__(self):
        return len(self.indices)


def dataset_front(path, label):
    """
    path (str): e.g. '/bigstor/anomaly/DAD/DAD/Tester1/normal_driving_1/';
    label (tuple): e.g. (0,)
    """
    dataset = DADFolder(path)

    if os.path.isdir(path + 'front_depth_strong/'):
        if os.listdir(path + 'front_depth_strong/'):
            val = dataset.class_to_idx['front_depth_strong']
        else:
            val = dataset.class_to_idx['front_depth']
    else:
        val = dataset.class_to_idx['front_depth']

    # try:
    #     val = dataset.class_to_idx['front_depth_strong']
    #     if not os.listdir(path + 'front_depth_strong/'):
    #         val = dataset.class_to_idx['front_depth']
    # except KeyError:
    #     val = dataset.class_to_idx['front_depth']

    # val = dataset.class_to_idx['front_depth']
    y = np.array(dataset.targets)
    idx_front = np.argwhere(np.isin(y, (val,))).flatten()
    dataset = DADSubset(dataset, idx_front, label[0])
    return dataset


def gen_dataset(root, tester_list, label=(0,), n_tester=1):
    dataset_list = []
    dataset_class = idx_to_class[label[0]]

    for tester in tester_list[:n_tester]:

        # For normal data
        if label[0] == 0:
            for i in range(1, 7):
                folder = f'{root}{tester}/normal_driving_{i}/'
                dataset_list.append(dataset_front(folder, label))
        # For abnormal data
        else:
            folder = f'{root}{tester}/{dataset_class}/'
            dataset_list.append(dataset_front(folder, label))

    dataset = ConcatDataset(dataset_list)
    return dataset


def sample_(X, n, random_state=42):
    """
    Generate a random sample from X.

    X (np.array): any array
    n (int): the number of the sample needed
    """
    np.random.seed(random_state)
    index = np.random.choice(X.shape[0], n, replace=False)
    return X[index]

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
# 2. Loader for Driver's Anomaly Dataset
# #########################################################################
class DADLoader(BaseDataset):
    def __init__(self,
                 root: str='/bigstor/anomaly/DAD/DAD/',
                 filename: str='dad',
                 train: int=0,
                 n_normal_train: int=10000,
                 load_method: int=0,
                 trained_type: int=0,
                 label_normal: tuple=(0,),
                 label_abnormal: tuple=(2,),
                 ratio_abnormal: float=0.1,
                 random_state: int=42,
                 n_tester_normal: int=1,
                 n_tester_abnormal: int=1):
        super().__init__(root)

        # Initialization
        normal_class = idx_to_class[label_normal[0]]
        abnormal_class = idx_to_class[label_abnormal[0]]

        self.root = root + filename
        self.label_normal = label_normal
        self.label_abnormal = label_abnormal
        self.ratio_abnormal = ratio_abnormal

        # Read in normal data and abnormal data
        dataset_normal = gen_dataset(root, tester_list, label_normal, n_tester_normal)
        dataset_abnormal = gen_dataset(root, tester_list, label_abnormal, n_tester_abnormal)

        # Get the sample size
        idx_normal = shuffle(list(range(len(dataset_normal))), random_state=random_state)
        idx_abnormal = shuffle(list(range(len(dataset_abnormal))), random_state=random_state)
        n_abnormal_train = int(n_normal_train * ratio_abnormal)

        idx_normal_train = idx_normal[:n_normal_train]
        idx_normal_test = idx_normal[n_normal_train:n_normal_train + 2000]
        idx_abnormal_train = idx_abnormal[:n_abnormal_train]
        idx_abnormal_test = idx_abnormal[n_abnormal_train:]


        if train:
            if load_method == 0:
                print('Loading normal data for training!')
                all_set = Subset(dataset_normal, idx_normal_train)

            if load_method == 1:
                print('Invalid load method in training!')
                return None

            if load_method == 2:
                print('Loading normal and abnormal data for training!')
                set_normal = Subset(dataset_normal, idx_normal_train)
                set_abnormal = Subset(dataset_abnormal, idx_abnormal_train)
                all_set = ConcatDataset([set_normal, set_abnormal])

        else:
            if load_method == 0:
                print('Loading normal data for testing!')
                all_set = Subset(dataset_normal, idx_normal_test)

            if load_method == 1:
                print('Loading abnormal data for testing!')
                if trained_type == 1:
                    if n_abnormal_train >= len(dataset_abnormal):
                        print('Too few data; giving you all instead!')
                        all_set = dataset_abnormal
                    else:
                        all_set = Subset(dataset_abnormal, idx_abnormal_test)

                elif trained_type == 0:
                    all_set = dataset_abnormal

            if load_method == 2:
                print('Invalid load method in test!')
                return None

        # Get the subset
        self.all_set = all_set


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

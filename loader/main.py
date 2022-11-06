"""
Title: main.py
Description: The loading functions.
"""

from .gaussian9d_loader import Gaussian9DLoader
from .gaussian9d_hard_loader import Gaussian9DHardLoader
from .synthetic_loader import SyntheticLoader
from .real_loader import RealLoader
from .kmnist_loader import KMNISTLoader
from .fmnist_loader import FashionMNISTLoader
from .mnist_loader import MNISTLoader
from .imagenet_loader import ImageNetLoader
from .cifar10_loader import CIFAR10Loader
from .real_pac_loader import RealPacLoader
from .real_pac_config_loader import RealPacConfigLoader
from .fmnist_config_loader import FashionMNISTConfigLoader
from .dad_loader import DADLoader
from .oct_loader import OCTLoader
from .oct_resize_loader import OCTResizeLoader


# #########################################################################
# 1. Load Dataset in One Function
# #########################################################################
def load_dataset(loader_name: str='gaussian9d',
                 root: str='/net/leksai/data/',
                 random_state: int=42,
                 filename: str='synthetic-4',
                 train: int=1,
                 n_normal_train: int=1200,
                 n_normal_val: int=10,
                 n_normal_test: int=208,
                 n: int=20000,
                 mix: bool=True,
                 ratio_abnormal: float=0.1,
                 ratio_abnormal_train: float=0.1,
                 ratio_abnormal_val: float=0.01,
                 n_features: int=9,
                 load_method: int=0,
                 threshold_type: int=0,
                 trained_type: int=0,
                 label_normal: tuple=(0,),
                 label_abnormal: tuple=(1,),
                 config: str='config_3',
                 ratio_a: float=0.1,
                 ratio_b: float=0.9,
                 n_tester_normal: int=1,
                 n_tester_abnormal: int=1):

    if loader_name == 'gaussian9d':
        return Gaussian9DLoader(random_state,
                                n,
                                mix,
                                ratio_abnormal,
                                n_features)

    if loader_name == 'gaussian9d_hard':
        return Gaussian9DHardLoader(random_state,
                                    n,
                                    mix,
                                    ratio_abnormal,
                                    n_features)

    if loader_name == 'synthetic':
        return SyntheticLoader(filename,
                               train,
                               n,
                               load_method,
                               label_normal,
                               label_abnormal,
                               ratio_abnormal)

    if loader_name == 'real':
        return RealLoader(root,
                          filename,
                          train,
                          n_normal_train,
                          load_method,
                          threshold_type,
                          trained_type,
                          label_normal,
                          label_abnormal,
                          ratio_abnormal)

    if loader_name == 'fmnist':
        return FashionMNISTLoader(root,
                                  'FashionMNIST',
                                  train,
                                  load_method,
                                  label_normal,
                                  label_abnormal,
                                  ratio_abnormal,
                                  random_state)

    if loader_name == 'kmnist':
        return FashionMNISTLoader(root,
                                  'KMNIST',
                                  train,
                                  load_method,
                                  label_normal,
                                  label_abnormal,
                                  ratio_abnormal,
                                  random_state)

    if loader_name == 'mnist':
        return MNISTLoader(root,
                           'MNIST',
                           train,
                           load_method,
                           label_normal,
                           label_abnormal,
                           ratio_abnormal,
                           random_state)

    if loader_name == 'cifar10':
        return CIFAR10Loader(root,
                             'CIFAR10',
                             train,
                             load_method,
                             label_normal,
                             label_abnormal,
                             ratio_abnormal,
                             random_state)

    if loader_name == 'imagenet':
        return ImageNetLoader('/mnt/data/huiyingli/imagenet/data/train',
                              train,
                              load_method,
                              threshold_type,
                              trained_type,
                              label_normal,
                              label_abnormal,
                              ratio_abnormal,
                              random_state)

    if loader_name == 'real_pac':
        return RealPacLoader(filename,
                             train,
                             n_normal_train,
                             n_normal_val,
                             n_normal_test,
                             load_method,
                             label_normal,
                             label_abnormal,
                             ratio_abnormal_train,
                             ratio_abnormal_val)

    if loader_name == 'real_pac_config':
        return RealPacConfigLoader(filename,
                                   config,
                                   train,
                                   n_normal_val,
                                   load_method,
                                   label_normal,
                                   label_abnormal,
                                   ratio_abnormal_train,
                                   ratio_abnormal_val)

    if loader_name == 'fmnist_config':
        return FashionMNISTConfigLoader(root,
                                        'FashionMNIST',
                                        train,
                                        load_method,
                                        label_normal,
                                        label_abnormal,
                                        ratio_abnormal,
                                        random_state,
                                        ratio_a,
                                        ratio_b)

    if loader_name == 'dad':
        return DADLoader('/bigstor/anomaly/DAD/DAD/',
                         'dad',
                         train,
                         n_normal_train,
                         load_method,
                         trained_type,
                         label_normal,
                         label_abnormal,
                         ratio_abnormal,
                         random_state,
                         n_tester_normal,
                         n_tester_abnormal)

    if loader_name == 'oct':
        return OCTLoader('../../data/OCT2017/train/',
                         train,
                         n_normal_train,
                         load_method,
                         trained_type,
                         label_normal,
                         label_abnormal,
                         ratio_abnormal,
                         random_state)

    if loader_name == 'oct_resize':
        return OCTResizeLoader('../../data/OCT2017/train/',
                               train,
                               n_normal_train,
                               load_method,
                               trained_type,
                               label_normal,
                               label_abnormal,
                               ratio_abnormal,
                               random_state)
    return None

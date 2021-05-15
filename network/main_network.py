"""
Title: main_network.py
Description: Build networks.
"""

from mnist_LeNet import MNISTLeNet, MNISTLeNetAutoencoder
from fmnist_LeNet import FashionMNISTLeNet, FashionMNISTLeNetAutoencoder
from kmnist_LeNet import KMNISTLeNet, KMNISTLeNetAutoencoder
from cifar10_LeNet import CIFAR10LeNet, CIFAR10LeNetAutoencoder
from gaussian3d_net import Guassian3DNet, Gaussian3DNetAutoencoder
from gaussian9d_net import Guassian9DNet, Gaussian9DNetAutoencoder
from imagenet_WideResNet import ImageNetWideResNet, ImageNetWideResNetAutoencoder
from spectrum_net import SpectrumNet, SpectrumAutoencoder
from mlp import *


# ####################m#####################################################
# 1. Build the Network Used for Training
# #########################################################################
def build_network(net_name='fmnist_LeNet_one_class'):

    net_name = net_name.strip()

    # oct_resize
    if net_name in ['oct_resize_one_class', 'oct_resize_hsc']:
        return CIFAR10LeNet(rep_dim=128)

    if net_name in ['oct_resize_rec', 'oct_resize_abc']:
        return CIFAR10LeNetAutoencoder(rep_dim=128)

    # oct
    if net_name in ['oct_one_class', 'oct_hsc']:
        return ImageNetWideResNet(rep_dim=256)

    if net_name in ['oct_rec', 'oct_abc']:
        return ImageNetWideResNetAutoencoder(rep_dim=256)

    # dad
    if net_name in ['dad_one_class', 'dad_hsc']:
        return ImageNetWideResNet(rep_dim=256)

    if net_name in ['dad_rec', 'dad_abc']:
        return ImageNetWideResNetAutoencoder(rep_dim=256)

    # imagenet
    if net_name in ['imagenet_WideResNet_one_class', 'imagenet_WideResNet_hsc']:
        return ImageNetWideResNet(rep_dim=256)

    if net_name in ['imagenet_WideResNet_rec', 'imagenet_WideResNet_abc']:
        return ImageNetWideResNetAutoencoder(rep_dim=256)

    # mnist
    if net_name in ['mnist_LeNet_one_class', 'mnist_LeNet_hsc']:
        return MNISTLeNet(rep_dim=64)

    if net_name in ['mnist_LeNet_rec', 'mnist_LeNet_abc']:
        return MNISTLeNetAutoencoder(rep_dim=64)

    # fmnist
    if net_name in ['fmnist_LeNet_one_class', 'fmnist_LeNet_hsc']:
        return FashionMNISTLeNet(rep_dim=64)

    if net_name in ['fmnist_LeNet_rec', 'fmnist_LeNet_abc']:
        return FashionMNISTLeNetAutoencoder(rep_dim=64)

    # kmnist
    if net_name in ['kmnist_LeNet_one_class', 'kmnist_LeNet_hsc']:
        return KMNISTLeNet(rep_dim=64)

    if net_name in ['kmnist_LeNet_rec', 'kmnist_LeNet_abc']:
        return KMNISTLeNetAutoencoder(rep_dim=64)

    # cifar10
    if net_name in ['cifar10_LeNet_one_class', 'cifar10_LeNet_hsc']:
        return CIFAR10LeNet(rep_dim=128)

    if net_name in ['cifar10_LeNet_rec', 'cifar10_LeNet_abc']:
        return CIFAR10LeNetAutoencoder(rep_dim=128)

    # Gaussian9D Debug
    if net_name in ['gaussian9d_one_class', 'gaussian9d_hsc']:
        return MLP(x_dim=9, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name in ['gaussian9d_rec', 'gaussian9d_abc']:
        return MLP_Autoencoder(x_dim=9, h_dims=[32, 16], rep_dim=8, bias=False)

    # Gaussian9D Debug
    if net_name == 'gaussian9d_one_class_debug':
        return MLP(x_dim=9, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == 'gaussian9d_rec_debug':
        return MLP_Autoencoder(x_dim=9, h_dims=[32, 16], rep_dim=8, bias=False)

    # Synthetic
    if net_name == 'synthetic_one_class':
        return MLP(x_dim=9, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == 'synthetic_rec':
        return MLP_Autoencoder(x_dim=9, h_dims=[32, 16], rep_dim=8, bias=False)

    # imagenet
    if net_name in ['imagenet_WideResNet_one_class', 'imagenet_WideResNet_hsc']:
        return ImageNetWideResNet(rep_dim=256)

    if net_name in ['imagenet_WideResNet_rec', 'imagenet_WideResNet_abc']:
        return ImageNetWideResNetAutoencoder(rep_dim=256)

    # mnist
    if net_name in ['mnist_LeNet_one_class', 'mnist_LeNet_hsc']:
        return MNISTLeNet(rep_dim=64)

    if net_name in ['mnist_LeNet_rec', 'mnist_LeNet_abc']:
        return MNISTLeNetAutoencoder(rep_dim=64)

    # spectrum
    if net_name in ['spectrum_one_class', 'spectrum_hsc']:
        return SpectrumNet(rep_dim=32)

    if net_name in ['spectrum_rec', 'spectrum_abc']:
        return SpectrumAutoencoder(rep_dim=64)

    # Satimage
    if net_name in ['satimage_one_class', 'satimage_hsc']:
        return MLP(x_dim=36, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name in ['satimage_rec', 'satimage_abc']:
        return MLP_Autoencoder(x_dim=36, h_dims=[32, 16], rep_dim=8, bias=False)

    # Covertype
    if net_name in ['covertype_one_class', 'covertype_hsc']:
        return MLP(x_dim=54, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name in ['covertype_rec', 'covertype_abc']:
        return MLP_Autoencoder(x_dim=54, h_dims=[32, 16], rep_dim=8, bias=False)

    # phish_url
    if net_name in ['phish_url_one_class', 'phish_url_hsc']:
        return MLP(x_dim=79, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name in ['phish_url_rec', 'phish_url_abc']:
        return MLP_Autoencoder(x_dim=79, h_dims=[32, 16], rep_dim=8, bias=False)

    # shuttle
    if net_name in ['shuttle_one_class', 'shuttle_hsc']:
        return MLP(x_dim=9, h_dims=[16, 8], rep_dim=4, bias=False)

    if net_name in ['shuttle_rec', 'shuttle_abc']:
        return MLP_Autoencoder(x_dim=9, h_dims=[16, 8], rep_dim=4, bias=False)

    # gas_drift
    if net_name in ['gas_drift_one_class', 'gas_drift_hsc']:
        return MLP(x_dim=128, h_dims=[64, 32], rep_dim=16, bias=False)

    if net_name in ['gas_drift_rec', 'gas_drift_abc']:
        return MLP_Autoencoder(x_dim=128, h_dims=[64, 32], rep_dim=16, bias=False)

    # awid
    if net_name in ['awid_one_class', 'awid_hsc']:
        return MLP(x_dim=13, h_dims=[8, 6], rep_dim=4, bias=False)

    if net_name in ['awid_rec', 'awid_abc']:
        return MLP_Autoencoder(x_dim=13, h_dims=[8, 6], rep_dim=4, bias=False)

    return None


# #########################################################################
# 2. Build the Network Used for Pre-Training (Only for One-Class Model)
# #########################################################################
def build_autoencoder(net_name='fmnist_LeNet_one_class'):

    net_name = net_name.strip()

    if net_name == 'oct_resize_one_class':
        return CIFAR10LeNetAutoencoder(rep_dim=128)

    if net_name == 'oct_one_class':
        return ImageNetWideResNetAutoencoder()

    if net_name == 'dad_one_class':
        return ImageNetWideResNetAutoencoder()

    if net_name == 'imagenet_WideResNet_one_class':
        return ImageNetWideResNetAutoencoder()

    if net_name == 'mnist_LeNet_one_class':
        return MNISTLeNetAutoencoder(rep_dim=64)

    if net_name == 'fmnist_LeNet_one_class':
        return FashionMNISTLeNetAutoencoder(rep_dim=64)

    if net_name == 'kmnist_LeNet_one_class':
        return KMNISTLeNetAutoencoder(rep_dim=64)

    if net_name == 'cifar10_LeNet_one_class':
        return CIFAR10LeNetAutoencoder(rep_dim=128)

    if net_name == 'spectrum_one_class':
        return SpectrumAutoencoder(rep_dim=32)

    if net_name == 'gaussian9d_one_class':
        return Gaussian9DNetAutoencoder(rep_dim=2)

    if net_name == 'gaussian9d_one_class_debug':
        return MLP_Autoencoder(x_dim=9, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == 'synthetic_one_class':
        return MLP_Autoencoder(x_dim=9, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == 'satimage_one_class':
        return MLP_Autoencoder(x_dim=36, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == 'covertype_one_class':
        return MLP_Autoencoder(x_dim=54, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == 'phish_url_one_class':
        return MLP_Autoencoder(x_dim=79, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == 'shuttle_one_class':
        return MLP_Autoencoder(x_dim=9, h_dims=[16, 8], rep_dim=4, bias=False)

    if net_name == 'gas_drift_one_class':
        return MLP_Autoencoder(x_dim=128, h_dims=[64, 32], rep_dim=16, bias=False)

    if net_name == 'awid_one_class':
        return MLP_Autoencoder(x_dim=13, h_dims=[8, 6], rep_dim=4, bias=False)

    return None

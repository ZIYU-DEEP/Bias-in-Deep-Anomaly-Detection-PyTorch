"""
Title: cifar10_LeNet.py
Reference: https://github.com/lukasruff/Deep-SAD-PyTorch/tree/master/src/networks
"""

from base_net import BaseNet
import torch
import torch.nn as nn
import torch.nn.functional as F


# #########################################################################
# 1. Encoder
# #########################################################################
class CIFAR10LeNet(BaseNet):

    def __init__(self, rep_dim=128, bias_terms=False):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder network
        self.conv1 = nn.Conv2d(3, 32, 5, bias=bias_terms, padding=2)
        nn.init.xavier_normal_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=bias_terms)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=bias_terms, padding=2)
        nn.init.xavier_normal_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=bias_terms)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=bias_terms, padding=2)
        nn.init.xavier_normal_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=bias_terms)
        self.fc1 = nn.Linear(128 * 4 * 4, 512, bias=bias_terms)
        nn.init.xavier_normal_(self.fc1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn1d1 = nn.BatchNorm1d(512, eps=1e-04, affine=bias_terms)
        self.fc2 = nn.Linear(512, self.rep_dim, bias=bias_terms)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(int(x.size(0)), -1)
        x = self.fc1(x)
        x = F.leaky_relu(self.bn1d1(x))
        x = self.fc2(x)
        return x

# class CIFAR10LeNet(BaseNet):
#
#     def __init__(self, rep_dim=128):
#         super().__init__()
#
#         self.rep_dim = rep_dim
#         self.pool = nn.MaxPool2d(2, 2)
#
#         self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
#         self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
#         self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
#         self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
#         self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
#         self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
#         self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)
#
#     def forward(self, x):
#         x = x.view(-1, 3, 32, 32)
#         x = self.conv1(x)
#         x = self.pool(F.leaky_relu(self.bn2d1(x)))
#         x = self.conv2(x)
#         x = self.pool(F.leaky_relu(self.bn2d2(x)))
#         x = self.conv3(x)
#         x = self.pool(F.leaky_relu(self.bn2d3(x)))
#         x = x.view(int(x.size(0)), -1)
#         x = self.fc1(x)
#         return x


# #########################################################################
# 2. Decoder
# #########################################################################
class CIFAR10LeNetDecoder(BaseNet):

    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim

        self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (4 * 4)), 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d4 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d5 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d6 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = x.view(int(x.size(0)), int(self.rep_dim / (4 * 4)), 4, 4)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv4(x)
        x = torch.sigmoid(x)
        return x


# #########################################################################
# 3. Autoencoder
# #########################################################################
class CIFAR10LeNetAutoencoder(BaseNet):

    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = CIFAR10LeNet(rep_dim=rep_dim)
        self.decoder = CIFAR10LeNetDecoder(rep_dim=rep_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

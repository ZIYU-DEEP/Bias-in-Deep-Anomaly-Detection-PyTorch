"""
Title: gaussian3d_net.py
Description: The network for 3D Gasussian datasets, suitable for Deep SVDD or Deep SAD.
"""

from base_net import BaseNet
import torch
import torch.nn as nn
import torch.nn.functional as F

# #########################################################################
# 1. Encoder
# #########################################################################
class Guassian9DNet(BaseNet):
    def __init__(self, rep_dim=2):
        super().__init__()

        self.rep_dim = rep_dim
        self.fc1 = nn.Linear(9, 6, bias=True)
        self.fc2 = nn.Linear(6, 3, bias=True)
        self.fc3 = nn.Linear(3, self.rep_dim, bias=True)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


# #########################################################################
# 2. Decoder
# #########################################################################
class Guassian9DNetDecoder(BaseNet):
    def __init__(self, rep_dim=2):
        super().__init__()

        self.rep_dim = rep_dim
        self.fc1 = nn.Linear(self.rep_dim, 3, bias=True)
        self.fc2 = nn.Linear(3, 6, bias=True)
        self.fc3 = nn.Linear(6, 9, bias=True)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


# #########################################################################
# 3. Autoencoder
# #########################################################################
class Gaussian9DNetAutoencoder(BaseNet):
    def __init__(self, rep_dim=2):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = Guassian9DNet(rep_dim=rep_dim)
        self.decoder = Guassian9DNetDecoder(rep_dim=rep_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

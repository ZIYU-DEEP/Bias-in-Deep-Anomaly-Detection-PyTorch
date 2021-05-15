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
class Guassian3DNet(BaseNet):
    def __init__(self, rep_dim=2):
        super().__init__()

        self.rep_dim = rep_dim
        self.fc1 = nn.Linear(3, 2, bias=True)
        self.fc2 = nn.Linear(2, self.rep_dim, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        return x


# #########################################################################
# 2. Decoder
# #########################################################################
class Guassian3DNetDecoder(BaseNet):
    def __init__(self, rep_dim=2):
        super().__init__()

        self.rep_dim = rep_dim
        self.fc1 = nn.Linear(self.rep_dim, 2, bias=True)
        self.fc2 = nn.Linear(2, 3, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        return x


# #########################################################################
# 3. Autoencoder
# #########################################################################
class Gaussian3DNetAutoencoder(BaseNet):
    def __init__(self, rep_dim=2):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = Guassian3DNet(rep_dim=rep_dim)
        self.decoder = Guassian3DNetDecoder(rep_dim=rep_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

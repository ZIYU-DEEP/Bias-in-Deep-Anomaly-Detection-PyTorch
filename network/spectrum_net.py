"""
Title: spectrum_net.py
"""

from base_net import BaseNet
import torch
import torch.nn as nn
import torch.nn.functional as F


# #########################################################################
# 1. Encoder
# #########################################################################
class SpectrumNet(BaseNet):
    def __init__(self, rep_dim=32):
        super().__init__()

        self.rep_dim = rep_dim
        self.lstm1 = nn.LSTM(128, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.fc1 = nn.Linear(32 * 125, self.rep_dim, bias=False)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x.reshape(x.size(0), 32 * 125).contiguous()
        x = self.fc1(x)
        return x


# #########################################################################
# 2. Decoder
# #########################################################################
class SpectrumDecoder(BaseNet):
    def __init__(self, rep_dim=32):
        super().__init__()

        self.rep_dim = rep_dim
        self.lstm1 = nn.LSTM(32, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 128, batch_first=True)
        self.fc1 = nn.Linear(self.rep_dim, 32 * 125)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(x.size(0), 125, 32).contiguous()
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = torch.sigmoid(x)
        return x


# #########################################################################
# 3. Autoencoder
# #########################################################################
class SpectrumAutoencoder(BaseNet):
    def __init__(self, rep_dim=32):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = SpectrumNet(rep_dim=rep_dim)
        self.decoder = SpectrumDecoder(rep_dim=rep_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

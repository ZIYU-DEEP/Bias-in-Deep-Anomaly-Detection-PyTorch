import torch.nn as nn

from base_net import BaseNet
from cbam import CBAM
from torch.nn import init
import torch.nn.functional as F
import torch
import torch.nn as nn



# #########################################################################
# 1. Encoder
# #########################################################################
# Credits to: https://github.com/hendrycks/ss-ood
class ImageNetWideResNet(BaseNet):

    def __init__(self, rep_dim=256):
        self.inplanes = 64
        super().__init__()

        self.rep_dim = rep_dim
        att_type = 'CBAM'
        layers = [2, 2, 2, 2]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.bam1, self.bam2, self.bam3 = None, None, None

        self.layer1 = self._make_layer(BasicBlock, 64, layers[0], att_type=att_type)
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2, att_type=att_type)
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2, att_type=att_type)
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2, att_type=att_type)

        self.fc = nn.Linear(512 * BasicBlock.expansion, self.rep_dim)

        init.kaiming_normal_(self.fc.weight)
        for key in self.state_dict():
            if key.split('.')[-1] == "weight":
                if "conv" in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1, att_type=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=att_type == 'CBAM'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=att_type == 'CBAM'))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if not self.bam1 is None:
            x = self.bam1(x)

        x = self.layer2(x)
        if not self.bam2 is None:
            x = self.bam2(x)

        x = self.layer3(x)
        if not self.bam3 is None:
            x = self.bam3(x)

        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class BasicBlock1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(BasicBlock1, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM(planes, 16)
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# #########################################################################
# 2. Decoder
# #########################################################################
# class ImageNetWideResNetDecoder(BaseNet):
#     def __init__(self, rep_dim=256):
#         super().__init__()
#
#         self.rep_dim = rep_dim
#         self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (4 * 4)), 128, 4, bias=False, padding=0)
#         nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
#         self.bn2d4 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
#         self.deconv2 = nn.ConvTranspose2d(128, 64, 5, bias=False, padding=2)
#         nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
#         self.bn2d5 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
#         self.deconv3 = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
#         nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
#         self.bn2d6 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
#         self.deconv4 = nn.ConvTranspose2d(32, 32, 5, bias=False, padding=2)
#         nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain('leaky_relu'))
#         self.bn2d7 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
#         self.deconv5 = nn.ConvTranspose2d(32, 32, 5, bias=False, padding=2)
#         nn.init.xavier_uniform_(self.deconv5.weight, gain=nn.init.calculate_gain('leaky_relu'))
#         self.bn2d8 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
#         self.deconv6 = nn.ConvTranspose2d(32, 3, 5, bias=False, padding=2)
#         nn.init.xavier_uniform_(self.deconv6.weight, gain=nn.init.calculate_gain('leaky_relu'))
#
#     def forward(self, x):
#         x = x.view(int(x.size(0)), int(self.rep_dim / (4 * 4)), 4, 4)
#         x = F.leaky_relu(x)
#         x = self.deconv1(x)
#         x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
#         x = self.deconv2(x)
#         x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)
#         x = self.deconv3(x)
#         x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
#         x = self.deconv4(x)
#         x = F.interpolate(F.leaky_relu(self.bn2d7(x)), scale_factor=2)
#         x = self.deconv5(x)
#         x = F.interpolate(F.leaky_relu(self.bn2d8(x)), scale_factor=2)
#         x = self.deconv6(x)
#         x = torch.sigmoid(x)
#         return x

class ImageNetWideResNetDecoder(BaseNet):
    def __init__(self, rep_dim=256):
        self.inplanes = 32
        super().__init__()

        self.rep_dim = rep_dim
        att_type = 'CBAM'
        layers = [2, 2, 2, 2]

        self.rep_dim = rep_dim

        self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (4 * 4)), 128, 4, bias=False, padding=0)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d4 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(128, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d5 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(32, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d6 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(32, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain('leaky_relu'))

        self.bn2d7 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv5 = nn.ConvTranspose2d(32, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv5.weight, gain=nn.init.calculate_gain('leaky_relu'))

        self.bn2d8 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv6 = nn.ConvTranspose2d(32, 3, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv6.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.layer1 = self._make_layer(BasicBlock1, 32, layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock1, 32, layers[1], stride=1)
        self.layer3 = self._make_layer(BasicBlock1, 32, layers[2], stride=1)
        self.layer4 = self._make_layer(BasicBlock1, 32, layers[3], stride=1)

    def _make_layer(self, block, planes, blocks, stride=1, att_type=None):
        downsample = None
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=att_type == 'CBAM'))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(int(x.size(0)), int(self.rep_dim / (4 * 4)), 4, 4)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)

        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)
        x = self.layer4(x)

        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        x = self.layer3(x)

        x = self.deconv4(x)
        x = F.interpolate(F.leaky_relu(self.bn2d7(x)), scale_factor=2)
        x = self.layer2(x)

        x = self.deconv5(x)
        x = F.interpolate(F.leaky_relu(self.bn2d8(x)), scale_factor=2)
        x = self.layer1(x)

        x = self.deconv6(x)
        x = torch.sigmoid(x)
        return x

# #########################################################################
# 3. Autoencoder
# #########################################################################
class ImageNetWideResNetAutoencoder(BaseNet):
    def __init__(self, rep_dim=256):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = ImageNetWideResNet(rep_dim=rep_dim)
        self.decoder = ImageNetWideResNetDecoder(rep_dim=rep_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

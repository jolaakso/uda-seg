import torch
from torch import nn

class BypassLayer(nn.Module):
    def __init__(self, channels=2048):
        super().__init__()
        bn_channels = channels // 4
        self.conv1 = nn.Conv2d(channels, bn_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(bn_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(bn_channels, bn_channels, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), bias=False)
        self.bn2 = nn.BatchNorm2d(bn_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = nn.Conv2d(bn_channels, channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        first_layer = self.conv1(x)
        first_layer = self.bn1(first_layer)
        second_layer = self.conv2(first_layer)
        second_layer = self.bn2(second_layer)
        x = self.conv3(first_layer + second_layer)
        x = self.bn3(x)
        return self.relu(x)

class FeatureModifier(nn.Module):
    def __init__(self, channels=2048):
        super().__init__()
        self.neck1 = BypassLayer(2048)
        self.ds1 = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.neck2 = BypassLayer(512)
        self.ds2 = nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.neck3 = BypassLayer(2048)

    def forward(self, x):
        first_layer = self.neck1(x)
        x = self.ds1(first_layer)
        x = self.bn1(x)
        x = self.neck2(x)
        x = self.ds2(x)
        x = self.bn2(x)

        return self.neck3(first_layer + x)

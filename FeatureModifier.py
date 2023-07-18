import torch
from torch import nn

class BypassLayer(nn.Module):
    def __init__(self, channels=2048):
        super().__init__()
        bn_channels = channels
        if channels >= 16:
            bn_channels = channels // 4
        self.conv1 = nn.Conv2d(channels, bn_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(bn_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(bn_channels, bn_channels, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), bias=False)
        self.bn2 = nn.BatchNorm2d(bn_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = nn.Conv2d(bn_channels, channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        first_layer = self.conv1(x)
        first_layer = self.bn1(first_layer)
        second_layer = self.conv2(first_layer)
        second_layer = self.bn2(second_layer)
        x = self.conv3(first_layer + second_layer)
        x = self.bn3(x)
        return x

class FeatureModifier(nn.Module):
    def __init__(self, in_channels=2048, out_channels=2048, sum_initial_layer=True):
        super().__init__()
        self.neck1 = BypassLayer(in_channels)
        self.ds1 = nn.Conv2d(in_channels, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.neck2 = BypassLayer(512)
        self.ds2 = nn.Conv2d(512, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.neck3 = BypassLayer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sum_initial_layer = sum_initial_layer

    def forward(self, x):
        first_layer = self.neck1(x)
        first_layer = self.relu(first_layer)
        x = self.ds1(first_layer)
        x = self.bn1(x)
        x = self.neck2(x)
        x = self.relu(x)
        x = self.ds2(x)
        x = self.bn2(x)

        if not self.sum_initial_layer:
            return self.neck3(x)

        return self.neck3(first_layer + x)

import torch
from torch import nn
import torch.nn.functional as F

class BigConvBlock(nn.Module):
    def __init__(self, width, height, channels):
        super().__init__()
        self.horizontal_scan = nn.Sequential(
            BigConv(width, channels, dilation=2),
            nn.ReLU()
        )
        self.vertical_scan = nn.Sequential(
            BigConv(height, channels, dilation=2),
            nn.ReLU()
        )
        self.fusion = nn.Conv2d(channels * 3, channels, 1)

    def forward(self, x):
        horizontal = self.horizontal_scan(x)
        vertical = self.vertical_scan(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        return self.fusion(torch.cat((horizontal, vertical, x), 1))

class BigConv(nn.Module):
    def __init__(self, width, channels, stride=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, (1, 2 * (width // dilation) + 1), dilation=dilation, stride=stride, padding=(0, width), groups=channels, bias=False)

    def forward(self, x):
        return self.conv(x)

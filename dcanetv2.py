import torch
from torch import nn
from torch.nn import functional as F
from SineTransform import DCALayer, DCATransposeLayer

class DCANetV2(nn.Module):
    def __init__(self, target_h_w):
        super().__init__()
        self.target_h_w = target_h_w
        self.net = nn.Sequential(
            nn.Conv2d(3, 25, 3, stride=2, padding=1),
            nn.CELU(),
            nn.BatchNorm2d(25),
            DCALayer(25, 70, stride=2, latent_size=10, summarize_depth=10, transform_depth=10),
            nn.CELU(),
            nn.BatchNorm2d(70),
            DCALayer(70, 256, stride=2, latent_size=25, summarize_depth=15, transform_depth=15),
            nn.CELU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 35, 3, padding=1),
            nn.CELU(),
            #DCATransposeLayer(256, 152, stride=2, latent_size=40, summarize_depth=30, transform_depth=30),
            #nn.CELU(),
            #nn.BatchNorm2d(152),
            #DCATransposeLayer(152, 70, stride=4, latent_size=25, summarize_depth=15, transform_depth=15),
            #nn.CELU(),
            #nn.BatchNorm2d(70),
            #DCATransposeLayer(70, 35, stride=2, latent_size=21, summarize_depth=9, transform_depth=9),
            #nn.CELU(),
        )

    def forward(self, x):
        x = self.net(x)
        x = F.interpolate(x, self.target_h_w, mode='bilinear')
        return { 'out': x }

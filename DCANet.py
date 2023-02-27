import torch
from torch import nn
from CA import CA2d, CATranspose2d

class DCANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            CA2d(3, 25, moments=5, stride=2, latent_size=5),
            nn.CELU(),
            nn.BatchNorm2d(25),
            CA2d(25, 70, moments=4, stride=2, latent_size=25),
            nn.CELU(),
            nn.BatchNorm2d(70),
            CA2d(70, 152, moments=4, stride=2, latent_size=50),
            nn.CELU(),
            nn.BatchNorm2d(152),
            CA2d(152, 256, moments=3, stride=2, latent_size=75),
            nn.CELU(),
            nn.BatchNorm2d(256),
            CATranspose2d(256, 152, moments=3, stride=2, latent_size=75),
            nn.CELU(),
            nn.BatchNorm2d(152),
            CATranspose2d(152, 70, moments=4, stride=4, latent_size=50),
            nn.CELU(),
            nn.BatchNorm2d(70),
            CATranspose2d(70, 35, moments=5, stride=2, latent_size=25),
            nn.CELU(),
        )

    def forward(self, x):
        return { 'out': self.net(x) }

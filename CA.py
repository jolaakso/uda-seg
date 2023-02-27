import torch
import math
from torch import nn
from BaseCA2d import BaseCA2d

class CA2d(BaseCA2d):
    def __init__(self, input_depth, output_depth, moments=3, stride=1, latent_size=None):
        super().__init__(input_depth, output_depth, moments=moments, latent_size=latent_size)
        self.stride = stride

    def forward(self, x):
        shape = x.shape
        conv_weights = super().forward(x).reshape((shape[0], self.output_depth, self.input_depth))
        x = torch.matmul(conv_weights, x.flatten(2)).unflatten(2, (shape[-2], shape[-1]))
        return nn.functional.interpolate(x, size=(shape[-2] // self.stride, shape[-1] // self.stride))

class CATranspose2d(BaseCA2d):
    def __init__(self, input_depth, output_depth, moments=3, stride=1, latent_size=None):
        super().__init__(input_depth, output_depth, moments=moments, latent_size=latent_size)
        self.stride = stride

    def forward(self, x):
        shape = x.shape
        conv_weights = super().forward(x).reshape((shape[0], self.output_depth, self.input_depth))
        x = torch.matmul(conv_weights, x.flatten(2)).unflatten(2, (shape[-2], shape[-1]))
        return nn.functional.interpolate(x, size=(shape[-2] * self.stride, shape[-1] * self.stride))

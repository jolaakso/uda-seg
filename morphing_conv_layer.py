import torch
from torch import nn

class MorphingConv2d(nn.Module):
    def __init__(self, input_depth):
        super().__init__()
        self.morph_mapper = nn.Conv2d(input_depth, 2, 1)

    def forward(self, x):
        morph_map = self.morph_mapper(x)

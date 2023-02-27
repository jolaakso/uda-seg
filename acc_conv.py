import torch
from torch import nn

class AccConv2d(nn.Module):
    def __init__(self, input_depth, output_depth):
        super().__init__()
        self.combiner = nn.Conv2d(input_depth * 3, output_depth, 1)
        #self.vertical_weights = nn.Parameter(torch.randn(input_depth))
        #self.horizontal_weights = nn.Parameter(torch.randn(input_depth))

    def forward(self, x):
        #bcasted_v_weigths = self.vertical_weights.reshape((-1, 1, 1))
        #bcasted_h_weigths = self.horizontal_weights.reshape((-1, 1, 1))
        v_dist = x.sum(dim=-2, keepdim=True).softmax(dim=-1).broadcast_to(x.shape)#.cumsum(dim=-1)
        h_dist = x.sum(dim=-1, keepdim=True).softmax(dim=-2).broadcast_to(x.shape)#.cumsum(dim=-2)
        return self.combiner(torch.cat((x, v_dist, h_dist), -3))

import torch
import math
from torch import nn

class BaseCA2d(nn.Module):
    def __init__(self, input_depth, output_depth, moments=3, latent_size=None):
        super().__init__()
        self.input_depth = input_depth
        self.output_depth = output_depth
        if not latent_size and input_depth >= 2:
            latent_size = int(math.sqrt(input_depth))
        elif not latent_size and input_depth < 2:
            latent_size = 1
        self.freqs = nn.Parameter(torch.randn(moments - 1))
        self.phases = nn.Parameter(torch.randn(moments - 1))
        self.loc_freqs = nn.Parameter(torch.randn(moments - 1, 2))
        self.loc_phases = nn.Parameter(torch.randn(moments - 1, 2))
        self.conv_param_compress = nn.Linear(input_depth * ((moments - 1) * 2 + 1), latent_size)
        self.conv_param_predictor = nn.Linear(latent_size, output_depth * input_depth)

    def forward(self, x):
        shape = x.shape
        v_loc = torch.ones(shape[0], shape[1], shape[2], 1, dtype=x.dtype).to(x.device).cumsum(dim=-2)
        h_loc = torch.ones(shape[0], shape[1], 1, shape[3], dtype=x.dtype).to(x.device).cumsum(dim=-1)

        mean = x.mean(dim=(-1, -2), keepdim=True)
        moments_tensors = [mean]
        for i in range(self.freqs.numel()):
            loc_freq = self.loc_freqs[i]
            loc_phase = self.loc_phases[i]
            sine_dist = ((x - mean) * self.freqs[i] + self.phases[i]).sin()
            v_loc_sine_dist = (v_loc * loc_freq[0] + loc_phase[0]).sin()
            h_loc_sine_dist = (h_loc * loc_freq[1] + loc_phase[1]).sin()

            moments_tensors.append(sine_dist.mean(dim=(-1, -2), keepdim=True))
            moments_tensors.append((sine_dist * v_loc_sine_dist * h_loc_sine_dist).mean(dim=(-1, -2), keepdim=True))
        nn_input = torch.cat(moments_tensors, -1).flatten(-3)
        # shape == self.output_depth, self.input_depth, 1, 1
        conv_weights = nn.functional.relu(self.conv_param_compress(nn_input), inplace=True)
        conv_weights = self.conv_param_predictor(conv_weights)
        return conv_weights

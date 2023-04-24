import torch
from torch import nn

class FourierNormalization2d(nn.Module):
    def __init__(self, channels, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.means = nn.parameter.Parameter(torch.zeros(channels, height, width), requires_grad=False)
        self.phase_means = nn.parameter.Parameter(torch.zeros(channels, height, width), requires_grad=False)
        self.variances = nn.parameter.Parameter(torch.zeros(channels, height, width), requires_grad=False)
        self.phase_variances = nn.parameter.Parameter(torch.zeros(channels, height, width), requires_grad=False)
        self.covariances = nn.parameter.Parameter(torch.zeros(channels, height, width), requires_grad=False)
        self.iterations = nn.parameter.Parameter(torch.zeros(1), requires_grad=False)
        self.mult_mask = torch.outer(1 - torch.linspace(0.0, 1.0, height), 1 - torch.linspace(0.0, 1.0, width))

    def mean(self):
        return self.means / self.iterations

    def phase_mean(self):
        return self.phase_means / self.iterations

    def var(self):
        return self.variances / self.iterations

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                ffts = torch.fft.rfft2(x, norm="ortho")[:, :, :self.height, :self.width]
                self.iterations.copy_(self.iterations + 1)
                amplitudes = ffts.abs()
                phases = ffts.angle()
                self.means += amplitudes.mean(dim=0)
                self.phase_means += phases.mean(dim=0)
                self.phase_variances += phases.var(dim=0)
                #self.covariances += ((amplitudes - self.means) * (phases - self.phase_means)).mean(dim=0)
                self.variances += amplitudes.var(dim=0)
                # corrs = self.covariances / (self.variances * self.phase_variances).sqrt()

            return x

        ffts = torch.fft.rfft2(x, norm="ortho")
        amplitudes = ffts.abs()
        exp_phases = (1.0j * ffts.angle()).exp()
        mult = self.mult_mask * 0.9
        amplitudes[:, :, :self.height, :self.width] = mult * self.mean() + (1 - mult) * amplitudes[:, :, :self.height, :self.width]

        return torch.fft.irfft2(amplitudes * exp_phases, norm="ortho")

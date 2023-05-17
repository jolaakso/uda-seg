import torch
from torch import nn
import matplotlib.pyplot as plt

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
        self.eps = 1e-12

    def mean(self):
        return self.means / self.iterations

    def phase_mean(self):
        return self.phase_means / self.iterations

    def var(self):
        return self.variances / self.iterations

    def batch_marginals(self, batch_slice):
        y_means = batch_slice.mean(dim=-1, keepdim=True)
        x_means = batch_slice.mean(dim=-2, keepdim=True)

        return x_means * y_means

    def batch_norm(self, batch_amplitude_slice):
        batch_margs = self.batch_marginals(batch_amplitude_slice.mean(dim=0, keepdim=True))
        self_margs = self.batch_marginals(batch_amplitude_slice)

        return (batch_amplitude_slice / (self_margs + self.eps)) * batch_margs

    def reverse(self, amplitudes_slice, full_amplitudes, full_phases):
        _, _, height, width = amplitudes_slice.shape
        full_amplitudes[:, :, :height, :width] = amplitudes_slice

        return torch.fft.irfft2(full_amplitudes * (1.0j * full_phases).exp(), norm="ortho")

    def forward(self, x):
        ffts = torch.fft.rfft2(x, norm="ortho")
        full_amplitudes = ffts.abs()
        full_phases = ffts.angle()
        ffts_slice = ffts[:, :, :self.height, :self.width]

        if self.training:
            with torch.no_grad():
                self.iterations.copy_(self.iterations + 1)
                amplitudes = full_amplitudes[:, :, :self.height, :self.width]
                phases = full_phases[:, :, :self.height, :self.width]
                self.means += amplitudes.mean(dim=0)
                self.phase_means += phases.mean(dim=0)
                self.phase_variances += phases.var(dim=0)
                #self.covariances += ((amplitudes - self.means) * (phases - self.phase_means)).mean(dim=0)
                self.variances += amplitudes.var(dim=0)
                # corrs = self.covariances / (self.variances * self.phase_variances).sqrt()
                norm = self.batch_norm(amplitudes)

                reversed = self.reverse(norm, full_amplitudes, full_phases)

                return reversed


        amplitudes = ffts.abs()
        amplitudes_slice = amplitudes[:, :, :self.height, :self.width]
        y_means = amplitudes_slice.mean(dim=0, keepdim=True).mean(dim=-1, keepdim=True)
        x_means = amplitudes_slice.mean(dim=0, keepdim=True).mean(dim=-2, keepdim=True)
        selfmean = self.mean()[:, :self.height, :self.width]
        stat_y_means = selfmean.mean(dim=-1, keepdim=True)
        stat_x_means = selfmean.mean(dim=-2, keepdim=True)
        #plt.imshow(x[0].permute(1, 2, 0))
        #plt.show()
        #plt.imshow(self.mean().permute(1, 2, 0))
        #plt.show()

        #plt.imshow(amplitudes_slice.mean(dim=0).permute(1, 2, 0))
        #plt.show()

        amplitudes_slice *= (stat_x_means * stat_y_means) / (x_means * y_means + self.eps)
        amplitudes_slice *= stat_x_means.mean(dim=-1, keepdim=True) / amplitudes_slice.mean(dim=0).mean(dim=-2, keepdim=True).mean(dim=-1, keepdim=True)
        exp_phases = (1.0j * ffts.angle()).exp()
        amplitudes[:, :, :self.height, :self.width] = amplitudes_slice

        #plt.imshow(amplitudes_slice.mean(dim=0).permute(1, 2, 0))
        #plt.show()

        #plt.imshow(torch.fft.irfft2(amplitudes * exp_phases, norm="ortho")[0].permute(1, 2, 0))
        #plt.show()
        return torch.fft.irfft2(amplitudes * exp_phases, norm="ortho")

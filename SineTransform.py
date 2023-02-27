import torch
from torch import nn
from torch.nn import functional as F

class BaseDCALayer(nn.Module):
    def __init__(self, input_depth, output_depth, latent_size=4, stride=1, summarize_depth=4, transform_depth=4):
        super().__init__()
        self.summarizer = SineSummarize(input_depth, chord_depth=summarize_depth)
        self.transform = SineTransform(input_depth, chord_depth=transform_depth)
        self.stride = stride
        self.latent = nn.Sequential(
            nn.Linear(self.summarizer.chord_depth * input_depth, latent_size * 2),
            nn.CELU(inplace=True),
            nn.Linear(latent_size * 2, latent_size)
        )

        self.modifiers = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.CELU(),
            nn.Linear(latent_size, self.summarizer.chord_depth * input_depth * 4)
        )

        self.mixer = nn.Linear(latent_size + input_depth, output_depth)

    def interpolate(self, x):
        return x

    def forward(self, x):
        shape = x.shape
        latent_vector = self.latent(self.summarizer(x))
        modifier_vector = self.modifiers(latent_vector).reshape(self.transform.chord_depth, 2, 2, shape[0], shape[1])
        x = self.transform(x, modifier_vector)
        latent_vector = torch.repeat_interleave(latent_vector.unsqueeze(-1), shape[2], dim=-1)
        latent_vector = torch.repeat_interleave(latent_vector.unsqueeze(-1), shape[3], dim=-1)
        x = torch.cat((latent_vector, x), dim=1)
        x = self.mixer(x.flatten(2).permute(0, 2, 1))
        x = x.permute(0, 2, 1).unflatten(2, (shape[-2], shape[-1]))
        return self.interpolate(x)

class DCALayer(BaseDCALayer):
    def __init__(self, input_depth, output_depth, latent_size=4, stride=1, summarize_depth=4, transform_depth=4):
        super().__init__(input_depth, output_depth, latent_size, stride, summarize_depth, transform_depth)

    def interpolate(self, x):
        shape = x.shape
        return F.interpolate(x, size=(shape[-2] // self.stride, shape[-1] // self.stride))

class DCATransposeLayer(BaseDCALayer):
    def __init__(self, input_depth, output_depth, latent_size=4, stride=1, summarize_depth=4, transform_depth=4):
        super().__init__(input_depth, output_depth, latent_size, stride, summarize_depth, transform_depth)

    def interpolate(self, x):
        shape = x.shape
        return F.interpolate(x, size=(shape[-2] * self.stride, shape[-1] * self.stride))

class SineSummarize(nn.Module):
    def __init__(self, input_depth, chord_depth=4):
        super().__init__()
        self.chord_depth = chord_depth
        self.sine_freqs = nn.Parameter(torch.randn(chord_depth, 3, input_depth, 1, 1))
        self.sine_phases = nn.Parameter(torch.randn(chord_depth, 3, input_depth, 1, 1))
        self.sum_weights = nn.Parameter(torch.randn(chord_depth + 1))

    def forward(self, x):
        shape = x.shape
        v_loc = torch.ones(shape[0], shape[1], shape[2], 1, dtype=x.dtype).to(x.device)
        v_loc = F.normalize(v_loc, dim=-2).cumsum(dim=-2)
        h_loc = torch.ones(shape[0], shape[1], 1, shape[3], dtype=x.dtype).to(x.device)
        h_loc = F.normalize(h_loc, dim=-1).cumsum(dim=-1)

        mean = x.mean(dim=(-1, -2), keepdim=True)
        summary_statistics = torch.zeros((self.chord_depth, shape[0], shape[1], 1, 1), dtype=x.dtype, device=x.device)
        summary_statistics[-1] = mean
        for i in range(self.chord_depth):
            accum = torch.zeros_like(x)
            sine_freq = self.sine_freqs[i]
            sine_phase = self.sine_phases[i]
            accum += x * torch.sigmoid(v_loc * sine_freq[0] + sine_phase[0])
            accum += x * torch.sigmoid(h_loc * sine_freq[1] + sine_phase[1])
            accum += torch.sigmoid((x - mean) * sine_freq[2] + sine_phase[2])
            summary_statistics[i] = accum.mean(dim=(-1, -2), keepdim=True)
        summary_statistics = summary_statistics.permute(1, 0, 2, 3, 4)
        return summary_statistics.flatten(1)

class SineTransform(nn.Module):
    def __init__(self, input_depth, chord_depth=4):
        super().__init__()
        self.chord_depth = chord_depth
        #self.sine_freqs = nn.Parameter(torch.randn(chord_depth, 2, input_depth, 1, 1))
        #self.sine_phases = nn.Parameter(torch.randn(chord_depth, 2, input_depth, 1, 1))

    # Modifiers: (chord_depth, 2, 2, batch_size, input_depth)
    def forward(self, x, modifiers):
        modifiers = modifiers.unsqueeze(-1).unsqueeze(-1)
        shape = x.shape
        v_loc = torch.ones(shape[0], shape[1], shape[2], 1, dtype=x.dtype, device=x.device)
        v_loc = F.normalize(v_loc, dim=-2).cumsum(dim=-2)
        h_loc = torch.ones(shape[0], shape[1], 1, shape[3], dtype=x.dtype, device=x.device)
        h_loc = F.normalize(h_loc, dim=-1).cumsum(dim=-1)
        result = torch.zeros_like(x)
        for i in range(self.chord_depth):
            modifier = modifiers[i]
            freq_modifier = modifier[0]
            phase_modifier = modifier[1]
            #sine_freq = self.sine_freqs[i]
            #sine_phase = self.sine_phases[i]
            #result += x * (freq_modifier[0] * v_loc * sine_freq[0] + sine_phase[0] + phase_modifier[0]).sin()
            #result += x * (freq_modifier[1] * h_loc * sine_freq[1] + sine_phase[1] + phase_modifier[1]).sin()
            result += x * torch.sigmoid(freq_modifier[0] * v_loc + phase_modifier[0])
            result += x * torch.sigmoid(freq_modifier[1] * h_loc + phase_modifier[1])
        return result

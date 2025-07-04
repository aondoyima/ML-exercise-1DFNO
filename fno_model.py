# fno_model.py

import torch
import torch.nn as nn
import torch.fft

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = 1 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes))
        self.weights_imag = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes))

    def compl_mul1d(self, input, weights_real, weights_imag):
        # input: (B, in_channels, K)
        # weights: (in_channels, out_channels, K)

        # Output: (B, out_channels, K)
        out_real = torch.einsum("bik, iok -> bok", input.real, weights_real) - \
                torch.einsum("bik, iok -> bok", input.imag, weights_imag)
        out_imag = torch.einsum("bik, iok -> bok", input.real, weights_imag) + \
                torch.einsum("bik, iok -> bok", input.imag, weights_real)

        return out_real + 1j * out_imag

    def forward(self, x):
        B, C, X = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(B, self.out_channels, x_ft.shape[-1], dtype=torch.cfloat, device=x.device)

        modes = min(self.modes, x_ft.shape[-1])
        out_ft[:, :, :modes] = self.compl_mul1d(x_ft[:, :, :modes], self.weights_real, self.weights_imag)
        x = torch.fft.irfft(out_ft, n=X, dim=-1)
        return x

class FNO1D(nn.Module):
    def __init__(self, modes=16, width=64, in_channels=3, out_channels=3):
        super().__init__()
        self.fc0 = nn.Linear(in_channels, width)
        self.conv1 = SpectralConv1d(width, width, modes)
        self.conv2 = SpectralConv1d(width, width, modes)
        self.conv3 = SpectralConv1d(width, width, modes)
        self.conv4 = SpectralConv1d(width, width, modes)
        self.w = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(4)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x, params=None):
        # x: (B, C, X), params: (B, P)
        B, C, X = x.shape
        if params is not None:
            P = params.shape[1]
            p_broadcast = params.unsqueeze(-1).repeat(1, 1, X)  # (B, P, X)
            x = torch.cat([x, p_broadcast], dim=1)  # (B, C+P, X)

        x = x.permute(0, 2, 1)  # (B, X, C+P)
        x = self.fc0(x)         # (B, X, W)
        x = x.permute(0, 2, 1)  # (B, W, X)

        for conv, w in zip([self.conv1, self.conv2, self.conv3, self.conv4], self.w):
            x = conv(x) + w(x)
            x = torch.relu(x)

        x = x.permute(0, 2, 1)  # (B, X, W)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x.permute(0, 2, 1)  # (B, C, X)


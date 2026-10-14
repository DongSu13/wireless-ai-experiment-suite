"""Neural L2-normalized beam predictors from local experiment code.

The real/imaginary channel layout follows y = h^H w x + n; MRT uses h/||h||.
Architectures learn beam vectors and do not enforce a physical RF constraint.
The epsilon-stabilized normalization leaves zero raw output zero; nonzero
outputs have norm slightly below one.
"""
from __future__ import annotations
import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def _validate_n_ant(n_ant: int) -> None:
    if isinstance(n_ant, bool) or not isinstance(n_ant, int) or n_ant < 1:
        raise ValueError("n_ant must be a positive integer")


def _validate_beam_input(x: torch.Tensor, n_ant: int) -> None:
    if x.ndim != 3 or tuple(x.shape[1:]) != (2, n_ant):
        raise ValueError("beam input must have shape (batch, 2, n_ant)")


class BeamformerMLP(nn.Module):

    def __init__(self, n_ant: int=32, hidden: int=256):
        super().__init__()
        _validate_n_ant(n_ant)
        d = n_ant * 2
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.LeakyReLU(0.2, inplace=True), nn.Linear(hidden, hidden), nn.LeakyReLU(0.2, inplace=True), nn.Linear(hidden, d))
        self.n_ant = n_ant

    def forward(self, h_two: torch.Tensor) -> torch.Tensor:
        _validate_beam_input(h_two, self.n_ant)
        b = h_two.shape[0]
        z = self.mlp(h_two.reshape(b, -1))
        y = z.view(b, 2, self.n_ant)
        nrm = torch.sqrt((y ** 2).sum(dim=(1, 2), keepdim=True) + 1e-08)
        return y / nrm

class BeamformerResMLP(nn.Module):

    def __init__(self, n_ant: int=32, hidden: int=256, n_blocks: int=3):
        super().__init__()
        _validate_n_ant(n_ant)
        self.n_ant = n_ant
        d = n_ant * 2
        self.in_proj = nn.Linear(d, hidden)
        self.blocks = nn.ModuleList([_ResBlockLinear(hidden) for _ in range(n_blocks)])
        self.out_proj = nn.Linear(hidden, d)

    def forward(self, h_two: torch.Tensor) -> torch.Tensor:
        _validate_beam_input(h_two, self.n_ant)
        b = h_two.shape[0]
        z = F.leaky_relu(self.in_proj(h_two.reshape(b, -1)), 0.2)
        for blk in self.blocks:
            z = blk(z)
        y = self.out_proj(z).view(b, 2, self.n_ant)
        nrm = torch.sqrt((y ** 2).sum(dim=(1, 2), keepdim=True) + 1e-08)
        return y / nrm

class _ResBlockLinear(nn.Module):

    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.LeakyReLU(0.2, inplace=True), nn.Linear(d, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

class _RNNBeamformer(nn.Module):

    def __init__(self, n_ant: int, rnn_type: str, hidden: int=128, num_layers: int=2):
        super().__init__()
        _validate_n_ant(n_ant)
        self.n_ant = n_ant
        rnn_type = rnn_type.lower()
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(2, hidden, num_layers, batch_first=True, bidirectional=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(2, hidden, num_layers, batch_first=True, bidirectional=True)
        elif rnn_type == 'rnn':
            self.rnn = nn.RNN(2, hidden, num_layers, batch_first=True, bidirectional=True, nonlinearity='tanh')
        else:
            raise ValueError(rnn_type)
        enc = hidden * 2
        self.fc = nn.Sequential(nn.Linear(enc, 256), nn.LeakyReLU(0.2, inplace=True), nn.Linear(256, n_ant * 2))

    def forward(self, h_two: torch.Tensor) -> torch.Tensor:
        _validate_beam_input(h_two, self.n_ant)
        b = h_two.shape[0]
        seq = h_two.permute(0, 2, 1)
        (out, _) = self.rnn(seq)
        z = out.mean(dim=1)
        y = self.fc(z).view(b, 2, self.n_ant)
        nrm = torch.sqrt((y ** 2).sum(dim=(1, 2), keepdim=True) + 1e-08)
        return y / nrm

class BeamformerLSTM(_RNNBeamformer):

    def __init__(self, n_ant: int=32, hidden: int=128, num_layers: int=2):
        super().__init__(n_ant, 'lstm', hidden, num_layers)

class BeamformerGRU(_RNNBeamformer):

    def __init__(self, n_ant: int=32, hidden: int=128, num_layers: int=2):
        super().__init__(n_ant, 'gru', hidden, num_layers)

class BeamformerRNN(_RNNBeamformer):

    def __init__(self, n_ant: int=32, hidden: int=128, num_layers: int=2):
        super().__init__(n_ant, 'rnn', hidden, num_layers)

class _TCNBlockBeam(nn.Module):

    def __init__(self, c: int, k: int, dilation: int):
        super().__init__()
        pad = (k - 1) * dilation // 2
        self.conv = nn.Conv1d(c, c, k, padding=pad, dilation=dilation)
        self.norm = nn.BatchNorm1d(c)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)) + x)

class BeamformerTCN(nn.Module):

    def __init__(self, n_ant: int=32, width: int=64):
        super().__init__()
        _validate_n_ant(n_ant)
        self.n_ant = n_ant
        self.in_proj = nn.Conv1d(2, width, 1)
        self.blocks = nn.ModuleList([_TCNBlockBeam(width, 3, 1), _TCNBlockBeam(width, 3, 2), _TCNBlockBeam(width, 3, 4)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(width, 256), nn.LeakyReLU(0.2, inplace=True), nn.Linear(256, n_ant * 2))

    def forward(self, h_two: torch.Tensor) -> torch.Tensor:
        _validate_beam_input(h_two, self.n_ant)
        b = h_two.shape[0]
        x = self.in_proj(h_two)
        for blk in self.blocks:
            x = blk(x)
        z = self.pool(x).flatten(1)
        y = self.fc(z).view(b, 2, self.n_ant)
        nrm = torch.sqrt((y ** 2).sum(dim=(1, 2), keepdim=True) + 1e-08)
        return y / nrm

class _PosEncBeam(nn.Module):

    def __init__(self, d_model: int, max_len: int=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

class BeamformerTransformer(nn.Module):

    def __init__(self, n_ant: int=32, d_model: int=96, nhead: int=4, num_layers: int=2, dim_ff: int=192):
        super().__init__()
        _validate_n_ant(n_ant)
        self.n_ant = n_ant
        self.in_proj = nn.Linear(2, d_model)
        self.pos = _PosEncBeam(d_model, n_ant + 4)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout=0.05)
        self.enc = nn.TransformerEncoder(layer, num_layers)
        self.fc = nn.Sequential(nn.Linear(d_model, 256), nn.LeakyReLU(0.2, inplace=True), nn.Linear(256, n_ant * 2))

    def forward(self, h_two: torch.Tensor) -> torch.Tensor:
        _validate_beam_input(h_two, self.n_ant)
        b = h_two.shape[0]
        seq = h_two.permute(0, 2, 1)
        z = self.in_proj(seq)
        z = self.pos(z)
        z = self.enc(z.transpose(0, 1)).transpose(0, 1)
        pooled = z.mean(dim=1)
        y = self.fc(pooled).view(b, 2, self.n_ant)
        nrm = torch.sqrt((y ** 2).sum(dim=(1, 2), keepdim=True) + 1e-08)
        return y / nrm
BEAM_ARCH_CHOICES = ('mlp', 'resnet', 'lstm', 'gru', 'rnn', 'tcn', 'transformer')

def build_beam_model(arch: str, n_ant: int=32) -> nn.Module:
    arch = arch.lower().strip()
    if arch == 'mlp':
        return BeamformerMLP(n_ant=n_ant)
    if arch == 'resnet':
        return BeamformerResMLP(n_ant=n_ant)
    if arch == 'lstm':
        return BeamformerLSTM(n_ant=n_ant)
    if arch == 'gru':
        return BeamformerGRU(n_ant=n_ant)
    if arch == 'rnn':
        return BeamformerRNN(n_ant=n_ant)
    if arch == 'tcn':
        return BeamformerTCN(n_ant=n_ant)
    if arch == 'transformer':
        return BeamformerTransformer(n_ant=n_ant)
    raise ValueError(f'Unknown beam architecture {arch!r}; choices: {BEAM_ARCH_CHOICES}')

def twoch_to_complex(y: torch.Tensor) -> torch.Tensor:
    return torch.complex(y[:, 0], y[:, 1])

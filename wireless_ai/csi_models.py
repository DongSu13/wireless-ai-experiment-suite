"""Continuous-latent CSI architectures adapted from local experiment code.

All models return (reconstruction, latent). Latent dimensions are floating
values, not encoded feedback bits; quantized feedback is a separate model.
"""
from __future__ import annotations
import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def _validate_latent_dim(latent_dim: int) -> None:
    if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim < 1:
        raise ValueError("latent_dim must be a positive integer; it counts continuous values, not bits")


def _validate_csi_input(x: torch.Tensor) -> None:
    if x.ndim != 4 or tuple(x.shape[1:]) != (2, 12, 32):
        raise ValueError("CSI input must have shape (batch, 2, 12, 32)")


def twoch_to_complex(y: torch.Tensor) -> torch.Tensor:
    return torch.complex(y[:, 0], y[:, 1])

def complex_to_twoch(h: torch.Tensor) -> torch.Tensor:
    return torch.stack([h.real, h.imag], dim=1)

def _seq_from_twoch(x: torch.Tensor) -> torch.Tensor:
    b = x.size(0)
    return x.permute(0, 2, 1, 3).reshape(b, 12, 64)

class CSIAutoEncoder(nn.Module):

    def __init__(self, latent_dim: int=36):
        super().__init__()
        _validate_latent_dim(latent_dim)
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(nn.Conv2d(2, 64, kernel_size=3, padding=1), nn.GroupNorm(8, 64), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.GroupNorm(16, 128), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.GroupNorm(32, 256), nn.LeakyReLU(0.2, inplace=True))
        self.pool = nn.AdaptiveAvgPool2d((2, 4))
        self.fc_enc = nn.Sequential(nn.Linear(256 * 2 * 4, 512), nn.LeakyReLU(0.2, inplace=True), nn.Linear(512, latent_dim), nn.Sigmoid())
        self.fc_dec = nn.Sequential(nn.Linear(latent_dim, 512), nn.LeakyReLU(0.2, inplace=True), nn.Linear(512, 12 * 32 * 2))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _validate_csi_input(x)
        z = self.enc(x)
        z = self.pool(z).flatten(1)
        return self.fc_enc(z)

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        return self.fc_dec(code).view(-1, 2, 12, 32)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _validate_csi_input(x)
        code = self.encode(x)
        return (self.decode(code), code)

class _ResBlock2d(nn.Module):

    def __init__(self, c: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(max(1, c // 32), c), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(max(1, c // 32), c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

class CSIAutoEncoderResNet(nn.Module):

    def __init__(self, latent_dim: int=36):
        super().__init__()
        _validate_latent_dim(latent_dim)
        self.latent_dim = latent_dim
        self.stem = nn.Sequential(nn.Conv2d(2, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.LeakyReLU(0.2, inplace=True))
        self.res1 = _ResBlock2d(64)
        self.res2 = _ResBlock2d(64)
        self.mid = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(16, 128), nn.LeakyReLU(0.2, inplace=True), _ResBlock2d(128), nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.GroupNorm(32, 256), nn.LeakyReLU(0.2, inplace=True))
        self.pool = nn.AdaptiveAvgPool2d((1, 2))
        self.fc_enc = nn.Sequential(nn.Linear(256 * 1 * 2, 512), nn.LeakyReLU(0.2, inplace=True), nn.Linear(512, latent_dim), nn.Sigmoid())
        self.fc_dec = nn.Sequential(nn.Linear(latent_dim, 512), nn.LeakyReLU(0.2, inplace=True), nn.Linear(512, 12 * 32 * 2))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _validate_csi_input(x)
        z = self.stem(x)
        z = self.res2(self.res1(z))
        z = self.mid(z)
        z = self.pool(z).flatten(1)
        return self.fc_enc(z)

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        return self.fc_dec(code).view(-1, 2, 12, 32)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _validate_csi_input(x)
        code = self.encode(x)
        return (self.decode(code), code)

class _RNNEncoderCSI(nn.Module):

    def __init__(self, latent_dim: int, rnn_type: str, hidden: int=128, num_layers: int=2):
        super().__init__()
        _validate_latent_dim(latent_dim)
        self.latent_dim = latent_dim
        self.proj = nn.Linear(64, hidden)
        rnn_type = rnn_type.lower()
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(hidden, hidden, num_layers, batch_first=True, bidirectional=True)
            enc_dim = hidden * 2
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(hidden, hidden, num_layers, batch_first=True, bidirectional=True)
            enc_dim = hidden * 2
        elif rnn_type == 'rnn':
            self.rnn = nn.RNN(hidden, hidden, num_layers, batch_first=True, bidirectional=True, nonlinearity='tanh')
            enc_dim = hidden * 2
        else:
            raise ValueError(rnn_type)
        self.fc_enc = nn.Sequential(nn.Linear(enc_dim, 512), nn.LeakyReLU(0.2, inplace=True), nn.Linear(512, latent_dim), nn.Sigmoid())
        self.fc_dec = nn.Sequential(nn.Linear(latent_dim, 512), nn.LeakyReLU(0.2, inplace=True), nn.Linear(512, 12 * 32 * 2))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _validate_csi_input(x)
        seq = _seq_from_twoch(x)
        z = F.leaky_relu(self.proj(seq), 0.2)
        (out, _) = self.rnn(z)
        pooled = out.mean(dim=1)
        return self.fc_enc(pooled)

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        return self.fc_dec(code).view(-1, 2, 12, 32)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _validate_csi_input(x)
        code = self.encode(x)
        return (self.decode(code), code)

class CSIAutoEncoderLSTM(_RNNEncoderCSI):

    def __init__(self, latent_dim: int=36, hidden: int=128, num_layers: int=2):
        super().__init__(latent_dim, 'lstm', hidden, num_layers)

class CSIAutoEncoderGRU(_RNNEncoderCSI):

    def __init__(self, latent_dim: int=36, hidden: int=128, num_layers: int=2):
        super().__init__(latent_dim, 'gru', hidden, num_layers)

class CSIAutoEncoderRNN(_RNNEncoderCSI):

    def __init__(self, latent_dim: int=36, hidden: int=128, num_layers: int=2):
        super().__init__(latent_dim, 'rnn', hidden, num_layers)

class _TCNBlock1d(nn.Module):

    def __init__(self, c: int, k: int, dilation: int):
        super().__init__()
        pad = (k - 1) * dilation // 2
        self.conv = nn.Conv1d(c, c, k, padding=pad, dilation=dilation)
        self.norm = nn.GroupNorm(max(1, c // 16), c)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)) + x)

class CSIAutoEncoderTCN(nn.Module):

    def __init__(self, latent_dim: int=36, width: int=96):
        super().__init__()
        _validate_latent_dim(latent_dim)
        self.latent_dim = latent_dim
        self.in_proj = nn.Conv1d(64, width, 1)
        self.blocks = nn.ModuleList([_TCNBlock1d(width, 3, 1), _TCNBlock1d(width, 3, 2), _TCNBlock1d(width, 3, 4)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_enc = nn.Sequential(nn.Linear(width, 512), nn.LeakyReLU(0.2, inplace=True), nn.Linear(512, latent_dim), nn.Sigmoid())
        self.fc_dec = nn.Sequential(nn.Linear(latent_dim, 512), nn.LeakyReLU(0.2, inplace=True), nn.Linear(512, 12 * 32 * 2))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _validate_csi_input(x)
        s = _seq_from_twoch(x).permute(0, 2, 1)
        z = self.in_proj(s)
        for blk in self.blocks:
            z = blk(z)
        z = self.pool(z).flatten(1)
        return self.fc_enc(z)

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        return self.fc_dec(code).view(-1, 2, 12, 32)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _validate_csi_input(x)
        code = self.encode(x)
        return (self.decode(code), code)

class _PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int=32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

class CSIAutoEncoderTransformer(nn.Module):

    def __init__(self, latent_dim: int=36, d_model: int=128, nhead: int=4, num_layers: int=2, dim_ff: int=256):
        super().__init__()
        _validate_latent_dim(latent_dim)
        self.latent_dim = latent_dim
        self.in_proj = nn.Linear(64, d_model)
        self.pos = _PositionalEncoding(d_model, 16)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout=0.05)
        self.enc = nn.TransformerEncoder(layer, num_layers)
        self.fc_enc = nn.Sequential(nn.Linear(d_model, 512), nn.LeakyReLU(0.2, inplace=True), nn.Linear(512, latent_dim), nn.Sigmoid())
        self.fc_dec = nn.Sequential(nn.Linear(latent_dim, 512), nn.LeakyReLU(0.2, inplace=True), nn.Linear(512, 12 * 32 * 2))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _validate_csi_input(x)
        seq = _seq_from_twoch(x)
        z = self.in_proj(seq)
        z = self.pos(z)
        z = self.enc(z.transpose(0, 1)).transpose(0, 1)
        pooled = z.mean(dim=1)
        return self.fc_enc(pooled)

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        return self.fc_dec(code).view(-1, 2, 12, 32)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _validate_csi_input(x)
        code = self.encode(x)
        return (self.decode(code), code)
CSI_ARCH_CHOICES = ('cnn', 'resnet', 'lstm', 'gru', 'rnn', 'tcn', 'transformer')

def build_csi_model(arch: str, latent_dim: int=36) -> nn.Module:
    arch = arch.lower().strip()
    if arch == 'cnn':
        return CSIAutoEncoder(latent_dim=latent_dim)
    if arch == 'resnet':
        return CSIAutoEncoderResNet(latent_dim=latent_dim)
    if arch == 'lstm':
        return CSIAutoEncoderLSTM(latent_dim=latent_dim)
    if arch == 'gru':
        return CSIAutoEncoderGRU(latent_dim=latent_dim)
    if arch == 'rnn':
        return CSIAutoEncoderRNN(latent_dim=latent_dim)
    if arch == 'tcn':
        return CSIAutoEncoderTCN(latent_dim=latent_dim)
    if arch == 'transformer':
        return CSIAutoEncoderTransformer(latent_dim=latent_dim)
    raise ValueError(f'Unknown CSI architecture {arch!r}; choices: {CSI_ARCH_CHOICES}')

"""Continuous-latent NumPy autoencoder with manual gradients and Adam.

latent_dim counts floating values, not a transmitted bit budget.
"""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np

def _positive_dimension(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_matrix(x: np.ndarray, d_in: int) -> None:
    if x.ndim != 2 or x.shape[1] != d_in or not np.isfinite(x).all():
        raise ValueError("input must be a finite (batch, d_in) matrix")


class Adam:

    def __init__(self, params: Dict[str, np.ndarray], lr: float=0.001, beta1: float=0.9, beta2: float=0.999, eps: float=1e-08, wd: float=1e-05):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.wd = wd
        self.m = {k: np.zeros_like(v) for (k, v) in params.items()}
        self.v = {k: np.zeros_like(v) for (k, v) in params.items()}
        self.t = 0

    def step(self, grads: Dict[str, np.ndarray]) -> None:
        self.t += 1
        for k in self.params:
            g = grads[k] + self.wd * self.params[k]
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (g * g)
            mhat = self.m[k] / (1 - self.beta1 ** self.t)
            vhat = self.v[k] / (1 - self.beta2 ** self.t)
            self.params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

class CSIFCAE:

    def __init__(self, d_in: int, hidden: int, latent_dim: int, seed: int=0):
        _positive_dimension("d_in", d_in)
        _positive_dimension("hidden", hidden)
        rng = np.random.RandomState(seed)
        self.d_in = d_in
        self.hidden = hidden
        _positive_dimension("latent_dim", latent_dim)
        self.latent_dim = latent_dim
        self.W1 = rng.randn(d_in, hidden).astype(np.float32) * np.sqrt(2.0 / d_in)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.randn(hidden, latent_dim).astype(np.float32) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(latent_dim, dtype=np.float32)
        self.W3 = rng.randn(latent_dim, d_in).astype(np.float32) * np.sqrt(2.0 / latent_dim)
        self.b3 = np.zeros(d_in, dtype=np.float32)

    def params_dict(self) -> Dict[str, np.ndarray]:
        return {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2, 'W3': self.W3, 'b3': self.b3}

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        _validate_matrix(X, self.W1.shape[0])
        z1 = X @ self.W1 + self.b1
        h = np.maximum(z1, 0.0)
        u = h @ self.W2 + self.b2
        c = 0.5 * (1.0 + np.tanh(0.5 * u))
        y = c @ self.W3 + self.b3
        cache = {'X': X, 'z1': z1, 'h': h, 'u': u, 'c': c}
        return (y, cache)

    def backward(self, cache: Dict[str, np.ndarray], d_out: np.ndarray) -> Dict[str, np.ndarray]:
        X = cache['X']
        z1 = cache['z1']
        h = cache['h']
        c = cache['c']
        dY = d_out
        dc = dY @ self.W3.T
        du = dc * c * (1.0 - c)
        dh = du @ self.W2.T
        dz1 = dh * (z1 > 0).astype(np.float32)
        dW3 = c.T @ dY
        db3 = np.sum(dY, axis=0)
        dW2 = h.T @ du
        db2 = np.sum(du, axis=0)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)
        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}

    def save_npz(self, path: str, best_cos: float) -> None:
        d = self.params_dict()
        d['latent_dim'] = np.array([self.latent_dim], dtype=np.int32)
        d['best_val_cos'] = np.array([best_cos], dtype=np.float32)
        np.savez_compressed(path, **d)

    @classmethod
    def load_npz(cls, path: str) -> Tuple['CSIFCAE', int, float]:
        with np.load(path, allow_pickle=False) as z:
            latent_dim = int(z['latent_dim'][0])
            best = float(z['best_val_cos'][0])
            d_in, hidden = z['W1'].shape
            m = cls(d_in, hidden, latent_dim, seed=0)
            for name, expected in m.params_dict().items():
                value = z[name].astype(np.float32)
                if value.shape != expected.shape or not np.isfinite(value).all():
                    raise ValueError(f"invalid saved parameter {name}")
                setattr(m, name, value)
        return (m, latent_dim, best)

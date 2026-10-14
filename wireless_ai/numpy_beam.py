"""NumPy beam MLP with manual gradients, under y = h^H w x + n.

Normalization uses sqrt(sum(square) + epsilon), including in its derivative.
Zero raw output remains zero; other outputs have norm slightly below one.
"""
from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np
from .numpy_csi import Adam, _positive_dimension, _validate_matrix

class BeamformerMLP:

    def __init__(self, d_in: int, hidden: int, seed: int=0):
        _positive_dimension("d_in", d_in)
        _positive_dimension("hidden", hidden)
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(d_in, hidden).astype(np.float32) * np.sqrt(2.0 / d_in)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.randn(hidden, hidden).astype(np.float32) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(hidden, dtype=np.float32)
        self.W3 = rng.randn(hidden, d_in).astype(np.float32) * np.sqrt(2.0 / hidden)
        self.b3 = np.zeros(d_in, dtype=np.float32)

    def params_dict(self) -> Dict[str, np.ndarray]:
        return {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2, 'W3': self.W3, 'b3': self.b3}

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        _validate_matrix(X, self.W1.shape[0])
        z1 = X @ self.W1 + self.b1
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(z2, 0.0)
        y_raw = h2 @ self.W3 + self.b3
        nrm = np.sqrt(np.sum(y_raw ** 2, axis=1, keepdims=True) + 1e-08)
        y = y_raw / nrm
        cache = {'X': X, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'y_raw': y_raw, 'nrm': nrm, 'y': y}
        return (y, cache)

    def backward(self, cache: Dict[str, Any], grad_y: np.ndarray) -> Dict[str, np.ndarray]:
        y_raw = cache['y_raw']
        nrm = cache['nrm']
        dot = np.sum(grad_y * y_raw, axis=1, keepdims=True)
        grad_y_raw = grad_y / nrm - y_raw * dot / nrm ** 3
        h2 = cache['h2']
        z2 = cache['z2']
        h1 = cache['h1']
        z1 = cache['z1']
        X = cache['X']
        dW3 = h2.T @ grad_y_raw
        db3 = np.sum(grad_y_raw, axis=0)
        dh2 = grad_y_raw @ self.W3.T
        dz2 = dh2 * (z2 > 0).astype(np.float32)
        dW2 = h1.T @ dz2
        db2 = np.sum(dz2, axis=0)
        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * (z1 > 0).astype(np.float32)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)
        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}

    def save_npz(self, path: str, best_gain: float) -> None:
        d = self.params_dict()
        d['best_val_gain'] = np.array([best_gain], dtype=np.float32)
        np.savez_compressed(path, **d)

    @classmethod
    def load_npz(cls, path: str) -> Tuple['BeamformerMLP', float]:
        with np.load(path, allow_pickle=False) as z:
            d_in, hidden = z['W1'].shape
            m = cls(d_in, hidden, seed=0)
            for name, expected in m.params_dict().items():
                value = z[name].astype(np.float32)
                if value.shape != expected.shape or not np.isfinite(value).all():
                    raise ValueError(f"invalid saved parameter {name}")
                setattr(m, name, value)
            best = float(z['best_val_gain'][0])
        return (m, best)

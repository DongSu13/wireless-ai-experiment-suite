"""Fixed sparse 4-bit real/imaginary CSI feedback, fitted on training samples.

The trained mean, selected coefficient locations, and quantizer scales
are fixed shared offline state and are not counted in sample payloads.
This educational baseline is not a standards-compliant Type-II codec.
"""

from __future__ import annotations

from numbers import Integral
from typing import Dict, Tuple

import numpy as np

from .validation.metrics import feedback_bits

BITS_PER_REAL = 4


def _complex_samples(data: np.ndarray) -> np.ndarray:
    values = np.asarray(data)
    if values.ndim < 2 or values.shape[0] == 0 or values.size == 0 or not np.iscomplexobj(values):
        raise ValueError("expected nonempty complex [sample,...] CSI")
    if not np.isfinite(values).all():
        raise ValueError("CSI must contain finite values")
    return values


def fit_fixed_sparse(train: np.ndarray, max_coefficients: int = 64) -> Dict[str, np.ndarray]:
    """Fit mean, ranked locations and scales using only caller-supplied train."""
    train = _complex_samples(train)
    if isinstance(max_coefficients, bool) or not isinstance(max_coefficients, Integral) or max_coefficients <= 0:
        raise ValueError("max_coefficients must be a positive integer")
    flat = train.reshape(train.shape[0], -1)
    mean = flat.mean(axis=0)
    variance = np.mean(np.abs(flat - mean) ** 2, axis=0)
    indices = np.argsort(-variance, kind="stable")[:max_coefficients].astype(np.int32)
    residual = flat[:, indices] - mean[indices]
    scale_real = np.maximum(np.quantile(np.abs(residual.real), 0.999, axis=0), 1e-8).astype(np.float32)
    scale_imag = np.maximum(np.quantile(np.abs(residual.imag), 0.999, axis=0), 1e-8).astype(np.float32)
    return {"shape": np.array(train.shape[1:], dtype=np.int64),
            "mean": mean.astype(np.complex64), "indices": indices,
            "scale_real": scale_real, "scale_imag": scale_imag}


def quantize_dequantize(values: np.ndarray, scales: np.ndarray, bits: int = BITS_PER_REAL) -> np.ndarray:
    if isinstance(bits, bool) or not isinstance(bits, Integral) or not 1 <= bits <= 8:
        raise ValueError("quantizer bits must be an integer from 1 to 8")
    values, scales = np.asarray(values), np.asarray(scales)
    if np.iscomplexobj(values) or np.iscomplexobj(scales) or not np.isfinite(values).all() or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("quantizer requires finite real values and positive scales")
    levels = (1 << bits) - 1
    normalized = np.clip(values / scales, -1.0, 1.0)
    codes = np.rint((normalized + 1.0) * levels / 2.0).astype(np.uint8)
    return ((codes.astype(np.float32) * (2.0 / levels)) - 1.0) * scales


def reconstruct(data: np.ndarray, model: Dict[str, np.ndarray], budget_bits: int) -> Tuple[np.ndarray, int]:
    """Apply fixed trained state without fitting anything on evaluation samples."""
    data = _complex_samples(data)
    if tuple(model["shape"]) != data.shape[1:]:
        raise ValueError("CSI layout does not match the fitted sparse model")
    bits_per_complex = BITS_PER_REAL * 2
    if isinstance(budget_bits, bool) or not isinstance(budget_bits, Integral) or budget_bits <= 0 or budget_bits % bits_per_complex:
        raise ValueError("budget must be a positive integer divisible by 8 bits per complex coefficient")
    k = budget_bits // bits_per_complex
    indices = model["indices"][:k]
    if len(indices) != k:
        raise ValueError("model has too few selected coefficients")
    flat = data.reshape(data.shape[0], -1)
    result = np.broadcast_to(model["mean"], flat.shape).copy()
    residual = flat[:, indices] - model["mean"][indices]
    real = quantize_dequantize(residual.real, model["scale_real"][:k])
    imag = quantize_dequantize(residual.imag, model["scale_imag"][:k])
    result[:, indices] += real + 1j * imag
    return result.reshape(data.shape), feedback_bits(k * 2, BITS_PER_REAL)

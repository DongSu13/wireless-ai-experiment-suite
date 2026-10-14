"""Per-sample CSI and beam metrics shared by independent stage-one studies."""

from __future__ import annotations

from numbers import Integral
from typing import Tuple

import numpy as np


def _paired_arrays(truth: np.ndarray, estimate: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(truth)
    b = np.asarray(estimate)
    if a.shape != b.shape or a.ndim < 2 or a.shape[0] == 0:
        raise ValueError("truth and estimate need equal nonempty [sample, ...] shapes")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("metrics require finite values")
    return a, b


def nmse_per_sample(truth: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    """Sum squared reconstruction error / sum squared truth, per sample."""
    a, b = _paired_arrays(truth, estimate)
    axes = tuple(range(1, a.ndim))
    power = np.sum(np.abs(a) ** 2, axis=axes)
    if np.any(power <= 0):
        raise ValueError("zero-power truth sample has undefined NMSE")
    return np.sum(np.abs(a - b) ** 2, axis=axes) / power


def nmse_db_from_samples(nmse: np.ndarray) -> float:
    """Convert the mean linear NMSE to dB; do not average sample dB values."""
    values = np.asarray(nmse, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("expected nonempty one-dimensional nonnegative NMSE values")
    mean = float(values.mean())
    return float(-np.inf if mean == 0 else 10.0 * np.log10(mean))


def complex_sgcs_per_sample(truth: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    """Mean squared complex cosine across subbands; last axis is antennas."""
    a, b = _paired_arrays(truth, estimate)
    if not np.iscomplexobj(a) or not np.iscomplexobj(b):
        raise ValueError("SGCS requires complex CSI; reconstruct it before scoring")
    numerator = np.abs(np.sum(np.conj(a) * b, axis=-1)) ** 2
    denominator = np.sum(np.abs(a) ** 2, axis=-1) * np.sum(np.abs(b) ** 2, axis=-1)
    if np.any(denominator <= 0):
        raise ValueError("zero-power CSI vector has undefined SGCS")
    scores = numerator / denominator
    return scores.reshape(a.shape[0], -1).mean(axis=1)


def beam_gain_per_sample(channel: np.ndarray, beam: np.ndarray) -> np.ndarray:
    """|hᴴw|²/(||h||²||w||²), matching the repository's signal convention."""
    h, w = _paired_arrays(channel, beam)
    if h.ndim != 2 or not np.iscomplexobj(h) or not np.iscomplexobj(w):
        raise ValueError("beam gain requires complex [sample, antenna] arrays")
    numerator = np.abs(np.sum(np.conj(h) * w, axis=1)) ** 2
    denominator = np.sum(np.abs(h) ** 2, axis=1) * np.sum(np.abs(w) ** 2, axis=1)
    if np.any(denominator <= 0):
        raise ValueError("zero-power channel or beam has undefined gain")
    return numerator / denominator


def illustrative_se_per_sample(snr_db: float, gain: np.ndarray) -> np.ndarray:
    """Single-user log2(1+SNR*alignment); excludes pilot and coding overhead."""
    g = np.asarray(gain, dtype=np.float64)
    if np.any(g < 0) or not np.isfinite(g).all() or not np.isfinite(snr_db):
        raise ValueError("gain must be finite and nonnegative, and SNR finite")
    return np.log2(1.0 + (10.0 ** (snr_db / 10.0)) * g)


def feedback_bits(n_coefficients: int, bits_per_coefficient: int, side_information_bits: int = 0) -> int:
    fields = (n_coefficients, bits_per_coefficient, side_information_bits)
    if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in fields):
        raise ValueError("feedback fields must be nonnegative integers")
    return n_coefficients * bits_per_coefficient + side_information_bits


def paired_improvement_ci(
    ai: np.ndarray, baseline: np.ndarray, *, lower_is_better: bool,
    n_bootstrap: int = 2000, seed: int = 0,
) -> Tuple[float, float, float]:
    """Mean AI improvement and percentile 95% CI from paired resampling."""
    a = np.asarray(ai, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape or a.size == 0 or n_bootstrap < 100:
        raise ValueError("expected paired one-dimensional metrics and >=100 resamples")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("bootstrap metrics must be finite")
    difference = b - a if lower_is_better else a - b
    random = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    for start in range(0, n_bootstrap, 64):
        stop = min(start + 64, n_bootstrap)
        sample_indices = random.integers(0, difference.size, size=(stop - start, difference.size))
        boot_means[start:stop] = difference[sample_indices].mean(axis=1)
    return float(difference.mean()), float(np.quantile(boot_means, 0.025)), float(np.quantile(boot_means, 0.975))

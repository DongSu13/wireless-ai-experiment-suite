"""Simplified correlated multipath MISO channel, not a calibrated standard.

LoS and diffuse components have expected total power n_tx before Rician
mixing and sample-level lognormal shadowing.
"""
from __future__ import annotations
from typing import Any
import numpy as np


def _positive_count(name: str, value: int) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _finite_real(name: str, value: float) -> None:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)):
        raise ValueError(f"{name} must be a finite real number")


def generate_hcsi_down_all(
    n_samples: int, *, seed: int = 0, n_subbands: int = 12, n_tx: int = 32,
    n_taps: int = 8, spatial_rho: float = 0.82, delay_max: float = 1.0,
    tap_decay: float = 0.55, k_factor_db: float | None = 6.0,
    los_angle_spread: float = np.pi / 3, shadowing_std_db: float = 0.8,
) -> np.ndarray:
    """Return complex64 channels shaped (samples, subbands, antennas).

    Tap power is proportional to exp(-tap_decay * tap_index). Delays are
    normalized to subband spacing. None K selects diffuse-only fading;
    otherwise K is the LoS-to-diffuse power ratio independent of n_tx.
    Shadowing scales amplitude by 10 ** (normal_db / 20).
    """
    for name, value in (("n_samples", n_samples), ("n_subbands", n_subbands),
                        ("n_tx", n_tx), ("n_taps", n_taps)):
        _positive_count(name, value)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or not 0 <= seed <= 2 ** 32 - 1):
        raise ValueError("seed must be an integer in [0, 2**32 - 1]")
    for name, value in (("spatial_rho", spatial_rho), ("delay_max", delay_max),
                        ("tap_decay", tap_decay), ("los_angle_spread", los_angle_spread),
                        ("shadowing_std_db", shadowing_std_db)):
        _finite_real(name, value)
    if not 0 <= spatial_rho < 1:
        raise ValueError("spatial_rho must be in [0, 1)")
    if delay_max < 0 or tap_decay < 0 or shadowing_std_db < 0:
        raise ValueError("delay_max, tap_decay and shadowing_std_db must be nonnegative")
    if not 0 <= los_angle_spread <= np.pi / 2:
        raise ValueError("los_angle_spread must be in [0, pi/2]")
    if k_factor_db is not None:
        _finite_real("k_factor_db", k_factor_db)
    rng = np.random.RandomState(int(seed))
    idx = np.arange(n_tx, dtype=np.float64)
    correlation = np.power(float(spatial_rho), np.abs(idx[:, None] - idx[None, :]))
    lower = np.linalg.cholesky(correlation + 1e-9 * np.eye(n_tx))
    powers = np.exp(-tap_decay * np.arange(n_taps, dtype=np.float64))
    powers /= powers.sum()
    delays = np.sort(rng.uniform(0.05 * delay_max, delay_max, (n_samples, n_taps)), axis=1)
    acc = np.zeros((n_subbands, n_samples, n_tx), dtype=np.complex128)
    frequencies = np.arange(n_subbands, dtype=np.float64)[:, None]
    for tap in range(n_taps):
        gaussian = (rng.standard_normal((n_tx, n_samples))
                    + 1j * rng.standard_normal((n_tx, n_samples))) / np.sqrt(2.0)
        diffuse = (lower @ gaussian).T
        phase = np.exp(-1j * 2 * np.pi * frequencies * delays[:, tap][None, :])
        acc += np.sqrt(powers[tap]) * phase[:, :, None] * diffuse[None, :, :]
    if k_factor_db is not None:
        log_k = float(k_factor_db) * (np.log(10.0) / 10.0)
        diffuse_power = np.exp(-np.logaddexp(0.0, log_k))
        los_power = np.exp(-np.logaddexp(0.0, -log_k))
        acc *= np.sqrt(diffuse_power)
        angles = rng.uniform(-los_angle_spread, los_angle_spread, n_samples)
        steering = np.exp(-1j * np.pi * idx[None, :] * np.sin(angles)[:, None])
        los_delay = rng.uniform(0, 0.25 * delay_max, n_samples)
        phase = np.exp(-1j * 2 * np.pi * frequencies * los_delay[None, :])
        acc += np.sqrt(los_power) * phase[:, :, None] * steering[None, :, :]
    channels = acc.transpose(1, 0, 2)
    if shadowing_std_db > 0:
        with np.errstate(over="ignore", invalid="ignore"):
            channels *= (10.0 ** (rng.normal(0, shadowing_std_db, n_samples) / 20.0))[:, None, None]
    with np.errstate(over="ignore", invalid="ignore"):
        result = channels.astype(np.complex64)
    if not np.isfinite(result).all():
        raise ValueError("channel parameters produce nonfinite complex64 values")
    return result


def default_channel_kwargs() -> dict[str, Any]:
    """Default synthetic channel settings, independent of local datasets."""
    return {"n_subbands": 12, "n_tx": 32, "n_taps": 8, "spatial_rho": 0.82,
            "delay_max": 1.0, "tap_decay": 0.55, "k_factor_db": 6.0,
            "los_angle_spread": float(np.pi / 3), "shadowing_std_db": 0.8}

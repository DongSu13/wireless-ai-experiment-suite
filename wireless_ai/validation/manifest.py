"""Explicit CSI data-layout contracts for stage-one experiments."""

from __future__ import annotations

from typing import Dict

import numpy as np


COST2100_FEATURES = 2048
COST2100_HEIGHT = 32
COST2100_WIDTH = 32


def cost2100_ht_to_complex(ht: np.ndarray) -> np.ndarray:
    """Invert the original CsiNet HT layout: [Re, Im] + 0.5 offset.

    Returns the 32x32 truncated angle-delay representation, not raw OFDM CSI.
    Convention reference: C.-K. Wen, W.-T. Shih and S. Jin, "Deep Learning
    for Massive MIMO CSI Feedback", IEEE Wireless Communications Letters,
    2018, https://ieeexplore.ieee.org/document/8322184/
    Public implementation reference:
    https://github.com/sydney222/Python_CsiNet/blob/master/CS-CsiNet_onlytest.py
    """
    values = np.asarray(ht)
    if values.ndim != 2 or values.shape[1] != COST2100_FEATURES or values.shape[0] == 0:
        raise ValueError("COST2100 HT must have shape [sample, 2048]")
    if np.iscomplexobj(values) or not np.isfinite(values).all():
        raise ValueError("COST2100 HT must contain finite real numbers")
    centered = values.astype(np.float32, copy=False) - np.float32(0.5)
    real = centered[:, :1024].reshape(-1, COST2100_HEIGHT, COST2100_WIDTH)
    imag = centered[:, 1024:].reshape(-1, COST2100_HEIGHT, COST2100_WIDTH)
    return np.asarray(real + 1j * imag, dtype=np.complex64)


def cost2100_layout_manifest() -> Dict[str, object]:
    return {
        "source_variable": "HT", "source_shape": ["sample", 2048],
        "source_dtype": "float64 in MAT; converted to float32 for model input",
        "complex_conversion": "reshape [sample,2,32,32], subtract 0.5 from both channels",
        "result_axes": ["sample", "angle", "delay"],
        "result_shape": ["sample", 32, 32],
        "domain": "preprocessed truncated angle-delay CSI",
    }

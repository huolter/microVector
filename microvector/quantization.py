"""Vector quantization utilities."""

from typing import cast

import numpy as np


def binary_quantization(vector: np.ndarray, num_bits: int) -> np.ndarray:
    """
    Scalar quantization: maps float values into 2^num_bits integer buckets.

    Args:
        vector: 1-D float array.
        num_bits: Number of bits (1–16). Common values: 4, 8.

    Returns:
        Quantized vector as float32.

    Note:
        When min == max (constant vector), all values map to bucket 0.
    """
    if num_bits < 1 or num_bits > 16:
        raise ValueError(f"num_bits must be between 1 and 16, got {num_bits}")
    if vector.ndim != 1:
        raise ValueError(f"vector must be 1-D, got shape {vector.shape}")

    num_buckets = 2**num_bits
    v_min, v_max = float(np.min(vector)), float(np.max(vector))

    if v_min == v_max:
        return cast(np.ndarray, np.zeros(vector.shape, dtype=np.float32))

    ranges = np.linspace(v_min, v_max, num=num_buckets + 1, endpoint=True)
    quantized = np.digitize(vector, ranges, right=True) - 1
    quantized = np.clip(quantized, 0, num_buckets - 1)
    return cast(np.ndarray, quantized.astype(np.float32))

"""Tests for quantization utilities."""

import numpy as np
import pytest

from microvector.quantization import binary_quantization


class TestBinaryQuantization:
    def test_output_dtype_is_float32(self) -> None:
        v = np.array([0.1, 0.5, 0.9, 0.3])
        result = binary_quantization(v, num_bits=4)
        assert result.dtype == np.float32

    def test_shape_preserved(self) -> None:
        v = np.random.rand(10)
        result = binary_quantization(v, num_bits=8)
        assert result.shape == v.shape

    def test_output_values_in_valid_range(self) -> None:
        v = np.random.rand(64)
        result = binary_quantization(v, num_bits=4)
        assert np.all(result >= 0)
        assert np.all(result < 2**4)

    def test_constant_vector_returns_zeros(self) -> None:
        v = np.ones(4) * 5.0
        result = binary_quantization(v, num_bits=4)
        assert np.all(result == 0.0)

    def test_num_bits_1(self) -> None:
        v = np.array([0.0, 1.0, 0.5])
        result = binary_quantization(v, num_bits=1)
        assert np.all(result < 2)

    def test_num_bits_16(self) -> None:
        v = np.random.rand(4)
        result = binary_quantization(v, num_bits=16)
        assert np.all(result < 2**16)

    def test_invalid_num_bits_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="num_bits"):
            binary_quantization(np.ones(4), num_bits=0)

    def test_invalid_num_bits_too_large_raises(self) -> None:
        with pytest.raises(ValueError, match="num_bits"):
            binary_quantization(np.ones(4), num_bits=17)

    def test_2d_vector_raises(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            binary_quantization(np.ones((2, 4)), num_bits=4)

    def test_increasing_bits_increases_resolution(self) -> None:
        v = np.linspace(0, 1, 100)
        r4 = binary_quantization(v, num_bits=4)
        r8 = binary_quantization(v, num_bits=8)
        assert len(np.unique(r8)) >= len(np.unique(r4))

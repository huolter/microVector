"""Tests for similarity functions."""

import numpy as np
import pytest

from microvector.similarity import (
    cosine_similarity,
    dot_product_similarity,
    euclidean_similarity,
    get_metric,
)


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors_score_minus_one(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_norm_first_arg_returns_zero(self) -> None:
        zero = np.zeros(3)
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(zero, v) == 0.0

    def test_zero_norm_second_arg_returns_zero(self) -> None:
        zero = np.zeros(3)
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, zero) == 0.0

    def test_both_zero_returns_zero(self) -> None:
        zero = np.zeros(3)
        assert cosine_similarity(zero, zero) == 0.0

    def test_result_clipped_to_valid_range(self) -> None:
        v = np.array([1.0, 0.0])
        score = cosine_similarity(v, v)
        assert -1.0 <= score <= 1.0

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            cosine_similarity(np.ones(3), np.ones(4))

    def test_known_value(self) -> None:
        a = np.array([1.0, 1.0])
        b = np.array([1.0, 0.0])
        expected = 1.0 / np.sqrt(2.0)
        assert cosine_similarity(a, b) == pytest.approx(expected)


class TestEuclideanSimilarity:
    def test_identical_vectors_score_one(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        assert euclidean_similarity(v, v) == pytest.approx(1.0)

    def test_score_is_always_positive(self) -> None:
        a = np.array([0.0, 0.0])
        b = np.array([10.0, 10.0])
        assert euclidean_similarity(a, b) > 0.0

    def test_score_less_than_one_for_different_vectors(self) -> None:
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        assert euclidean_similarity(a, b) < 1.0

    def test_closer_vector_scores_higher(self) -> None:
        query = np.array([0.0, 0.0])
        near = np.array([1.0, 0.0])
        far = np.array([5.0, 0.0])
        assert euclidean_similarity(query, near) > euclidean_similarity(query, far)

    def test_known_value(self) -> None:
        a = np.array([0.0])
        b = np.array([1.0])
        assert euclidean_similarity(a, b) == pytest.approx(0.5)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            euclidean_similarity(np.ones(3), np.ones(4))


class TestDotProductSimilarity:
    def test_orthogonal_is_zero(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert dot_product_similarity(a, b) == pytest.approx(0.0)

    def test_known_value(self) -> None:
        a = np.array([2.0, 3.0])
        b = np.array([4.0, 5.0])
        assert dot_product_similarity(a, b) == pytest.approx(23.0)

    def test_negative_dot_product(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert dot_product_similarity(a, b) == pytest.approx(-1.0)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            dot_product_similarity(np.ones(3), np.ones(4))


class TestGetMetric:
    def test_returns_cosine(self) -> None:
        fn = get_metric("cosine")
        assert fn is cosine_similarity

    def test_returns_euclidean(self) -> None:
        fn = get_metric("euclidean")
        assert fn is euclidean_similarity

    def test_returns_dot(self) -> None:
        fn = get_metric("dot")
        assert fn is dot_product_similarity

    def test_unknown_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown metric"):
            get_metric("manhattan")

    def test_error_message_lists_valid_options(self) -> None:
        with pytest.raises(ValueError, match="cosine"):
            get_metric("bad")

"""Similarity and distance functions for vector search."""

from __future__ import annotations

from typing import Callable

import numpy as np


def _validate_shapes(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.

    Returns a value in [-1.0, 1.0] where 1.0 means identical direction.
    Returns 0.0 if either vector has zero norm (no direction to compare).

    Fixes original bug: crashed with ZeroDivisionError on zero-norm vectors.
    """
    _validate_shapes(a, b)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    raw = np.dot(a, b) / (norm_a * norm_b)
    return float(np.clip(raw, -1.0, 1.0))


def euclidean_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Euclidean similarity between two vectors: 1 / (1 + distance).

    Maps distance [0, ∞) → similarity (0, 1]. Identical vectors score 1.0.

    Fixes original bug: returned raw negative distance, making far vectors
    rank as more similar than near ones.
    """
    _validate_shapes(a, b)
    dist = float(np.linalg.norm(a - b))
    return 1.0 / (1.0 + dist)


def dot_product_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Raw dot product similarity.

    Not normalized to [0, 1]. Best used with pre-normalized embeddings
    (e.g., unit-norm vectors from OpenAI or HuggingFace models).
    """
    _validate_shapes(a, b)
    return float(np.dot(a, b))


METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "cosine": cosine_similarity,
    "euclidean": euclidean_similarity,
    "dot": dot_product_similarity,
}


def get_metric(name: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return the similarity function for the given metric name."""
    if name not in METRICS:
        raise ValueError(
            f"Unknown metric '{name}'. Choose from: {sorted(METRICS.keys())}"
        )
    return METRICS[name]

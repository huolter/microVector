"""Shared pytest fixtures."""

import numpy as np
import pytest

from microvector import MicroVectorDB

DIMENSION = 4


@pytest.fixture
def empty_db() -> MicroVectorDB:
    return MicroVectorDB()


@pytest.fixture
def db_with_nodes() -> MicroVectorDB:
    """Database with 3 orthogonal unit vectors for predictable similarity scores."""
    db = MicroVectorDB()
    db.add_node(np.array([1.0, 0.0, 0.0, 0.0]), "alpha", metadata={"tag": "a"})
    db.add_node(np.array([0.0, 1.0, 0.0, 0.0]), "beta", metadata={"tag": "b"})
    db.add_node(np.array([0.0, 0.0, 1.0, 0.0]), "gamma", metadata={"tag": "c"})
    return db


@pytest.fixture
def zero_vector() -> np.ndarray:
    return np.zeros(DIMENSION)


@pytest.fixture
def unit_vector() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0])

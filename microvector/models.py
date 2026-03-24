"""Data models for microvector."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Node:
    """A stored vector with its associated document and optional metadata."""

    index: int
    vector: np.ndarray
    document: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self.index == other.index and np.array_equal(self.vector, other.vector)


@dataclass(frozen=True)
class SearchResult:
    """An immutable search result returned by query methods."""

    index: int
    document: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: SearchResult) -> bool:
        return self.score < other.score

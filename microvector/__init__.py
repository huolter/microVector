"""
microvector — A lightweight, production-grade in-memory vector database.

Quick start::

    import numpy as np
    from microvector import MicroVectorDB

    db = MicroVectorDB(dimension=3)
    db.add_node(np.array([1.0, 0.0, 0.0]), "hello world")

    results = db.search_top_k(np.array([1.0, 0.0, 0.0]), k=1)
    print(results[0].document)   # "hello world"
    print(results[0].score)      # ~1.0
"""

__version__ = "0.2.0"

__all__ = [
    "MicroVectorDB",
    "SearchResult",
    "Node",
    "MicroVectorError",
    "DimensionMismatchError",
    "EmptyDatabaseError",
    "NodeNotFoundError",
]

from .core import MicroVectorDB
from .exceptions import (
    DimensionMismatchError,
    EmptyDatabaseError,
    MicroVectorError,
    NodeNotFoundError,
)
from .models import Node, SearchResult

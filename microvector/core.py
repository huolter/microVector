"""Core MicroVectorDB class."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .exceptions import DimensionMismatchError, EmptyDatabaseError, NodeNotFoundError
from .models import Node, SearchResult
from .quantization import binary_quantization
from .similarity import get_metric


class MicroVectorDB:
    """
    A lightweight, in-memory vector database.

    Indexes are assigned monotonically and are never reused after deletion.
    All similarity scores follow the convention: higher = more similar.

    Example:
        >>> import numpy as np
        >>> from microvector import MicroVectorDB
        >>> db = MicroVectorDB(dimension=3)
        >>> db.add_node(np.array([1.0, 0.0, 0.0]), "hello")
        0
        >>> results = db.search_top_k(np.array([1.0, 0.0, 0.0]), k=1)
        >>> results[0].document
        'hello'
    """

    def __init__(self, dimension: int) -> None:
        """
        Initialize a new MicroVectorDB.

        Args:
            dimension: The fixed dimension of all vectors in this database.

        Raises:
            ValueError: If dimension < 1.
        """
        if dimension < 1:
            raise ValueError(f"dimension must be >= 1, got {dimension}")
        self._dimension = dimension
        self._nodes: dict[int, Node] = {}
        self._next_index: int = 0

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    def _validate_vector(self, vector: np.ndarray) -> None:
        """Validate that a vector is a 1-D numpy array of the correct dimension."""
        if not isinstance(vector, np.ndarray):
            raise TypeError(f"vector must be np.ndarray, got {type(vector).__name__}")
        if vector.ndim != 1:
            raise ValueError(f"vector must be 1-D, got shape {vector.shape}")
        if vector.shape[0] != self._dimension:
            raise DimensionMismatchError(self._dimension, vector.shape[0])

    # ------------------------------------------------------------------ #
    # Mutation                                                              #
    # ------------------------------------------------------------------ #

    def add_node(
        self,
        vector: np.ndarray,
        document: str,
        metadata: dict[str, Any] | None = None,
        num_bits: int | None = None,
    ) -> int:
        """
        Add a single node to the database.

        Args:
            vector: The embedding vector. Must match the database dimension.
            document: The text or identifier associated with this vector.
            metadata: Optional dict of arbitrary key-value pairs.
            num_bits: If set, apply n-bit scalar quantization before storing.

        Returns:
            The index assigned to the new node.

        Raises:
            TypeError: If vector is not a numpy array.
            DimensionMismatchError: If vector dimension doesn't match the database.
        """
        self._validate_vector(vector)
        v = (
            binary_quantization(vector, num_bits)
            if num_bits is not None
            else vector.copy()
        )
        node = Node(
            index=self._next_index,
            vector=v,
            document=document,
            metadata=metadata or {},
        )
        self._nodes[self._next_index] = node
        self._next_index += 1
        return node.index

    def add_nodes(
        self,
        vectors: np.ndarray | list[np.ndarray],
        documents: list[str],
        metadata: list[dict[str, Any] | None] | None = None,
    ) -> list[int]:
        """
        Add multiple nodes in a single call.

        All inputs are validated before any insertion occurs, so either all
        nodes are added or none are (fail-fast, not partial).

        Args:
            vectors: A 2-D numpy array (n, dim) or list of 1-D arrays.
            documents: List of document strings, one per vector.
            metadata: Optional list of metadata dicts, one per vector.

        Returns:
            List of assigned indexes, in the same order as the inputs.

        Raises:
            ValueError: If vectors and documents have different lengths.
            DimensionMismatchError: If any vector has wrong dimension.
        """
        if isinstance(vectors, np.ndarray):
            if vectors.ndim != 2:
                raise ValueError(
                    "Batch vectors array must be 2-D (n_vectors, dimension)"
                )
            vector_list: list[np.ndarray] = [
                vectors[i] for i in range(vectors.shape[0])
            ]
        else:
            vector_list = list(vectors)

        if len(vector_list) != len(documents):
            raise ValueError(
                f"vectors and documents must have the same length: "
                f"{len(vector_list)} != {len(documents)}"
            )
        if metadata is not None and len(metadata) != len(vector_list):
            raise ValueError(
                f"metadata list must match vectors length: "
                f"{len(metadata)} != {len(vector_list)}"
            )

        # Validate all before inserting any
        for v in vector_list:
            self._validate_vector(v)

        metas: list[dict[str, Any] | None] = (
            metadata if metadata is not None else [None] * len(vector_list)
        )
        return [
            self.add_node(v, doc, meta)
            for v, doc, meta in zip(vector_list, documents, metas)
        ]

    def get_node(self, index: int) -> Node:
        """
        Retrieve a node by its index.

        Raises:
            NodeNotFoundError: If no node with that index exists.
        """
        if index not in self._nodes:
            raise NodeNotFoundError(index)
        return self._nodes[index]

    def update_node(
        self,
        index: int,
        vector: np.ndarray | None = None,
        document: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Update one or more fields of an existing node.

        Only provided fields are changed; omitted fields are left as-is.

        Raises:
            NodeNotFoundError: If no node with that index exists.
            DimensionMismatchError: If the new vector has wrong dimension.
        """
        node = self.get_node(index)
        if vector is not None:
            self._validate_vector(vector)
            node.vector = vector.copy()
        if document is not None:
            node.document = document
        if metadata is not None:
            node.metadata = metadata

    def remove_node(self, index: int) -> None:
        """
        Remove a node by its index.

        The index is permanently retired — it will not be reused for future nodes.

        Raises:
            NodeNotFoundError: If no node with that index exists.
        """
        if index not in self._nodes:
            raise NodeNotFoundError(index)
        del self._nodes[index]

    # ------------------------------------------------------------------ #
    # Search                                                                #
    # ------------------------------------------------------------------ #

    def search_top_k(
        self,
        query_vector: np.ndarray,
        k: int,
        metric: str = "cosine",
        filter_fn: Callable[[Node], bool] | None = None,
    ) -> list[SearchResult]:
        """
        Find the top-k most similar nodes to a query vector.

        Args:
            query_vector: The query embedding. Must match the database dimension.
            k: Number of results to return. If k > number of nodes, all nodes
               are returned (no error).
            metric: Distance metric — 'cosine', 'euclidean', or 'dot'.
            filter_fn: Optional predicate to pre-filter nodes before scoring.
                       Only nodes where filter_fn(node) is True are considered.

        Returns:
            List of SearchResult objects sorted by score descending.
            Each result includes: index, document, score, metadata.

        Raises:
            EmptyDatabaseError: If the database has no nodes.
            DimensionMismatchError: If query_vector dimension doesn't match.
            ValueError: If k < 1 or metric is unknown.
        """
        if not self._nodes:
            raise EmptyDatabaseError()
        self._validate_vector(query_vector)
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        sim_fn = get_metric(metric)
        candidates = (
            (n for n in self._nodes.values() if filter_fn(n))
            if filter_fn is not None
            else self._nodes.values()
        )

        results: list[SearchResult] = []
        for node in candidates:
            score = sim_fn(query_vector, node.vector)
            results.append(
                SearchResult(
                    index=node.index,
                    document=node.document,
                    score=score,
                    metadata=node.metadata,
                )
            )

        results.sort(reverse=True)
        return results[:k]

    def search_by_threshold(
        self,
        query_vector: np.ndarray,
        min_score: float,
        metric: str = "cosine",
    ) -> list[SearchResult]:
        """
        Return all nodes with similarity score >= min_score, sorted descending.

        Args:
            query_vector: The query embedding.
            min_score: Minimum score threshold (inclusive).
            metric: Distance metric — 'cosine', 'euclidean', or 'dot'.

        Returns:
            List of SearchResult objects with score >= min_score, sorted descending.

        Raises:
            EmptyDatabaseError: If the database has no nodes.
            DimensionMismatchError: If query_vector dimension doesn't match.
        """
        if not self._nodes:
            raise EmptyDatabaseError()
        self._validate_vector(query_vector)

        sim_fn = get_metric(metric)
        results: list[SearchResult] = []
        for node in self._nodes.values():
            score = sim_fn(query_vector, node.vector)
            if score >= min_score:
                results.append(
                    SearchResult(
                        index=node.index,
                        document=node.document,
                        score=score,
                        metadata=node.metadata,
                    )
                )
        results.sort(reverse=True)
        return results

    # ------------------------------------------------------------------ #
    # Persistence                                                           #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """
        Save the database to a ``.mvdb/`` directory.

        The directory contains:
        - ``index.json``: Node metadata, documents, and manifest.
        - ``vectors.npy``: All vectors as a 2-D numpy array.

        This format is safe (no arbitrary code execution unlike pickle),
        portable, and human-inspectable.

        Args:
            path: Destination path. The ``.mvdb`` extension is added automatically
                  if not already present.
        """
        path = Path(path)
        if not path.name.endswith(".mvdb"):
            path = path.with_suffix(".mvdb")
        path.mkdir(parents=True, exist_ok=True)

        indices = sorted(self._nodes.keys())
        manifest = {
            "version": "1.0",
            "dimension": self._dimension,
            "next_index": self._next_index,
            "count": len(indices),
            "nodes": [
                {
                    "index": i,
                    "document": self._nodes[i].document,
                    "metadata": self._nodes[i].metadata,
                    "vector_row": row,
                }
                for row, i in enumerate(indices)
            ],
        }

        if indices:
            vector_matrix = np.stack([self._nodes[i].vector for i in indices])
        else:
            vector_matrix = np.empty((0, self._dimension), dtype=np.float32)

        with open(path / "index.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        np.save(str(path / "vectors.npy"), vector_matrix)

    @classmethod
    def load(cls, path: str | Path) -> MicroVectorDB:
        """
        Load a database from a ``.mvdb/`` directory.

        Args:
            path: Path to the ``.mvdb`` directory (extension added if omitted).

        Returns:
            A fully restored MicroVectorDB instance.

        Raises:
            FileNotFoundError: If the path does not exist.
        """
        path = Path(path)
        if not path.name.endswith(".mvdb"):
            path = path.with_suffix(".mvdb")

        if not path.exists():
            raise FileNotFoundError(f"No database found at {path}")

        with open(path / "index.json", encoding="utf-8") as f:
            manifest = json.load(f)

        vectors = np.load(str(path / "vectors.npy"))
        db = cls(dimension=manifest["dimension"])
        db._next_index = manifest["next_index"]

        for node_meta in manifest["nodes"]:
            row = node_meta["vector_row"]
            node = Node(
                index=node_meta["index"],
                vector=vectors[row],
                document=node_meta["document"],
                metadata=node_meta["metadata"],
            )
            db._nodes[node.index] = node

        return db

    # ------------------------------------------------------------------ #
    # Dunder methods                                                        #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        """Return the number of nodes currently in the database."""
        return len(self._nodes)

    def __contains__(self, index: object) -> bool:
        """Check whether an index exists in the database."""
        return index in self._nodes

    def __repr__(self) -> str:
        return (
            f"MicroVectorDB(dimension={self._dimension}, "
            f"nodes={len(self._nodes)}, "
            f"next_index={self._next_index})"
        )

    def stats(self) -> dict[str, Any]:
        """
        Return a summary of the database state.

        Returns:
            Dict with keys: count, dimension, next_index, index_gaps, indices.
        """
        return {
            "count": len(self._nodes),
            "dimension": self._dimension,
            "next_index": self._next_index,
            "index_gaps": self._next_index - len(self._nodes),
            "indices": sorted(self._nodes.keys()),
        }

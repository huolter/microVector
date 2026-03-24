# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.2.0] — 2026-03-24

### Changed

- `MicroVectorDB(dimension)` is now optional. If omitted, the dimension is
  inferred automatically from the first vector inserted and locked for all
  subsequent inserts. Explicit `dimension=` still works as before.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-03-24

Complete rewrite from prototype to production-grade library.

### Fixed

- **Critical: `IndentationError` in `add_node`** — the original module could not be imported at all due to a syntax error in the method definition.
- **Euclidean distance returned negative values** — the original returned `-np.linalg.norm(...)`, making far vectors rank as *more* similar than near ones. Fixed to `1 / (1 + distance)`, mapping to `(0, 1]` where higher always means more similar.
- **Cosine similarity crashed on zero-norm vectors** — division by zero when either vector was all zeros. Now returns `0.0` (no direction, no similarity).
- **Vector dimension never validated** — `self._dimension` was stored in `__init__` but never checked. Any vector of any size was silently accepted. Now raises `DimensionMismatchError` with clear expected/got values.
- **Search results dropped the document text** — `search_top_k` returned `(index, score)` tuples, forcing callers to do a separate lookup. Now returns `SearchResult` objects with `index`, `document`, `score`, and `metadata`.
- **Index counter could diverge after removal** — internal index tracking was done via a list with no gap management. Replaced with `dict[int, Node]` keyed by index; `_next_index` increments monotonically and retired indexes are never reused.
- **Quantization was untested and had edge cases** — constant vectors (min == max) would cause `linspace` issues. Added input validation, edge-case handling, and a full test suite to verify correctness.

### Added

- **Package structure** — reorganized from a single `microvector.py` file into a proper `microvector/` package with `core.py`, `similarity.py`, `quantization.py`, `models.py`, `exceptions.py`, and `__init__.py`.
- **`SearchResult` dataclass** — immutable result type with `index`, `document`, `score`, `metadata` fields. Supports sorting.
- **`Node` dataclass** — replaces internal dict representation.
- **Custom exceptions** — `MicroVectorError` (base), `DimensionMismatchError`, `EmptyDatabaseError`, `NodeNotFoundError`. All carry structured attributes for programmatic inspection.
- **`add_nodes()`** — batch insert with fail-fast validation (all inputs validated before any insertion).
- **`get_node()`** — retrieve a node by index.
- **`update_node()`** — update vector, document, and/or metadata in place.
- **`search_by_threshold()`** — return all nodes with score >= a minimum threshold.
- **`filter_fn` parameter on `search_top_k`** — pre-filter nodes before scoring with any Python predicate.
- **`dot` distance metric** — raw dot product, best for pre-normalized embeddings.
- **`__len__`, `__contains__`, `__repr__`** — standard Python dunder methods.
- **`stats()`** — returns count, dimension, next_index, index_gaps, indices.
- **`metadata` support** — arbitrary key-value dicts stored per node, returned in search results.
- **Safe persistence** — replaced `pickle` with JSON manifest + numpy `.npy` file in a `.mvdb/` directory. Safe against arbitrary code execution, portable, human-inspectable.
- **`pyproject.toml`** — proper packaging with hatchling, PyPI classifiers, optional dev dependencies.
- **Full test suite** — 107 tests, 98.5% coverage across all modules.
- **CI/CD** — GitHub Actions workflow: lint (black + ruff + mypy) then test across Python 3.9, 3.10, 3.11, 3.12.

### Changed

- `save_to_disk` / `read_from_disk` → `save` / `MicroVectorDB.load` (classmethod). New format is a `.mvdb/` directory instead of a single pickle file.
- `search_top_k` now accepts `metric=` keyword (was `distance_metric=`) and returns `list[SearchResult]` instead of `list[tuple]`.

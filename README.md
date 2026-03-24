# MicroVector

A lightweight, production-grade in-memory vector database for Python.

```
pip install microvector
```

No external services. No complex setup. Just numpy and your embeddings.

---

## Features

- **Fast similarity search** — cosine, euclidean, and dot product metrics
- **Rich results** — every search result includes the document text and metadata, not just an index
- **Batch operations** — insert thousands of vectors in one call
- **Metadata filtering** — filter candidates before scoring with any Python predicate
- **Safe persistence** — JSON + numpy format (no pickle, no security risk)
- **Full type hints** — works great with mypy and IDEs
- **Zero required dependencies** beyond numpy

---

## Quick Start

```python
import numpy as np
from microvector import MicroVectorDB

# Create a database for 768-dimensional embeddings (e.g. OpenAI ada-002)
db = MicroVectorDB(dimension=768)

# Add documents with their embeddings
db.add_node(embed("Paris is the capital of France"), "Paris is the capital of France")
db.add_node(embed("The Eiffel Tower is in Paris"),   "The Eiffel Tower is in Paris")
db.add_node(embed("Python is a programming language"), "Python is a programming language")

# Search — results include document text, not just indexes
results = db.search_top_k(embed("What city is the Eiffel Tower in?"), k=2)
for r in results:
    print(f"{r.score:.3f}  {r.document}")
# 0.921  The Eiffel Tower is in Paris
# 0.887  Paris is the capital of France

# Save and reload
db.save("my_knowledge_base")
db = MicroVectorDB.load("my_knowledge_base")
```

---

## Installation

**Requires Python 3.9+ and numpy >= 1.21.**

```bash
pip install microvector
```

Or from source:

```bash
git clone https://github.com/huolter/microVector
cd microVector
pip install -e ".[dev]"
```

---

## API Reference

### `MicroVectorDB(dimension)`

Create a new database. All vectors must have this exact dimension.

```python
db = MicroVectorDB(dimension=512)
```

---

### `add_node(vector, document, metadata=None, num_bits=None) → int`

Add a single vector. Returns the assigned index.

```python
idx = db.add_node(
    vector=np.array([...]),
    document="The text this vector represents",
    metadata={"source": "wikipedia", "date": "2024-01"},
    num_bits=8,   # optional: apply 8-bit scalar quantization
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `vector` | `np.ndarray` | 1-D array matching the database dimension |
| `document` | `str` | Text or identifier associated with this vector |
| `metadata` | `dict` | Optional key-value pairs stored alongside the vector |
| `num_bits` | `int` | Optional quantization (1–16 bits). Reduces memory at the cost of precision. |

---

### `add_nodes(vectors, documents, metadata=None) → list[int]`

Batch insert. All inputs are validated before any insertion occurs.

```python
import numpy as np

vectors = np.random.rand(1000, 512)
docs = [f"Document {i}" for i in range(1000)]
indices = db.add_nodes(vectors, docs)
```

---

### `search_top_k(query_vector, k, metric='cosine', filter_fn=None) → list[SearchResult]`

Find the top-k most similar vectors. Returns results sorted by score descending.

```python
results = db.search_top_k(query, k=5)
results = db.search_top_k(query, k=5, metric="euclidean")

# Filter before scoring — only consider documents tagged "news"
results = db.search_top_k(
    query, k=5,
    filter_fn=lambda node: node.metadata.get("type") == "news"
)
```

**Metrics:**

| Metric | Range | Notes |
|--------|-------|-------|
| `cosine` | [-1, 1] | Best for text embeddings. Direction-based, scale-invariant. |
| `euclidean` | (0, 1] | `1 / (1 + dist)`. Identical vectors = 1.0. |
| `dot` | (-∞, +∞) | Fastest. Best for pre-normalized unit vectors. |

**`SearchResult` fields:**

```python
result.index     # int   — node's assigned index
result.document  # str   — the document text
result.score     # float — similarity score (higher = more similar)
result.metadata  # dict  — metadata stored with this node
```

---

### `search_by_threshold(query_vector, min_score, metric='cosine') → list[SearchResult]`

Return all nodes with score >= `min_score`, sorted descending.

```python
results = db.search_by_threshold(query, min_score=0.8)
```

---

### `get_node(index) → Node`

Retrieve a node by index.

```python
node = db.get_node(42)
print(node.document, node.metadata)
```

---

### `update_node(index, vector=None, document=None, metadata=None)`

Update fields of an existing node. Omit fields to leave them unchanged.

```python
db.update_node(42, document="Updated text")
db.update_node(42, metadata={"status": "reviewed"})
```

---

### `remove_node(index)`

Remove a node. Its index is permanently retired (never reused).

```python
db.remove_node(42)
```

---

### `save(path)` / `MicroVectorDB.load(path)`

Persist to and restore from a `.mvdb/` directory. Safe format: JSON + numpy binary.

```python
db.save("my_db")              # creates my_db.mvdb/
db = MicroVectorDB.load("my_db")
```

The `.mvdb/` directory contains:
- `index.json` — node metadata and documents (human-readable)
- `vectors.npy` — all vectors as a 2-D numpy array

---

### Utility methods

```python
len(db)           # number of nodes
42 in db          # check if index exists
repr(db)          # MicroVectorDB(dimension=512, nodes=1000, next_index=1000)
db.stats()        # {'count': 1000, 'dimension': 512, 'next_index': 1000, ...}
```

---

## Exceptions

All exceptions inherit from `MicroVectorError`.

```python
from microvector import (
    MicroVectorError,
    DimensionMismatchError,  # wrong vector dimension
    EmptyDatabaseError,      # search on empty database
    NodeNotFoundError,       # get/update/remove non-existent index
)
```

---

## Design Notes

**Why in-memory?** MicroVector is designed for small-to-medium datasets (up to ~100k vectors) where the simplicity of pure-Python outweighs the need for a dedicated service. It starts instantly, needs no configuration, and is trivially embeddable in any Python application.

**Why not pickle?** Pickle can execute arbitrary code when loading untrusted files. MicroVector uses JSON + numpy binary format which is safe, portable, and inspectable with any text editor.

**Index monotonicity.** Deleted indexes are never reused. This prevents stale external references from silently pointing to new data.

---

## Roadmap

- [ ] Approximate nearest neighbor (HNSW)
- [ ] Voronoi cell indexing
- [ ] FAISS optional backend for large datasets
- [ ] Async search support
- [ ] Benchmarks

---

## License

MIT

# MicroVector

MicroVector is a lightweight Python tool for managing a small-scale vector database. It is designed to handle vectors associated with documents, providing functionalities such as adding, removing, searching, and persisting the data to disk. All in-memory and with flat index.

MicroVector follows the SS paradigm: Stupid Simple. :-) 

## Features

- **Add Node:** Add a vector and its associated document to the database.
- **Remove Node:** Remove a node from the database based on its index.
- **Search Top K:** Retrieve the top K similar nodes based on a query vector.
- **Save to Disk:** Persist the database to a file using pickle.
- **Read from Disk:** Load the database from a previously saved file.

## Usage

```python
# Example Usage

from microvector import MicroVectorDB
import numpy as np

# Initialize MicroVectorDB with a specified vector dimension
vector_db = MicroVectorDB(dimension=50)

# Add nodes
vector_db.add_node(np.random.rand(50), "Document A")
vector_db.add_node(np.random.rand(50), "Document B")

# Remove a node by index
vector_db.remove_node(0)

# Search for the top 5 similar nodes based on a query vector
query_vector = np.random.rand(50)
top_results = vector_db.search_top_k(query_vector, k=5)

# Save the database to disk
vector_db.save_to_disk("vector_db.pkl")

# Read the database from disk
vector_db.read_from_disk("vector_db.pkl")
```

## Next

- Quantization
- Voroni Cells
- Hierarchical Navigable Small-World (HNSW)
- Examples
- Benchmarks on speed and memory ussage

## Links and references

- https://thedataquarry.com/posts/vector-db-3/
- https://www.pinecone.io/learn/series/faiss/faiss-tutorial/
- https://arxiv.org/abs/1603.09320
- https://www.youtube.com/watch?v=PNVJvZEkuXo
- https://www.youtube.com/watch?v=t9mRf2S5vDI 


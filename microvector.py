import numpy as np
import pickle
from typing import List, Union


class MicroVectorDB:
    def __init__(self, dimension: int):
        """
        Initialize MicroVectorDB.

        Parameters:
        - dimension (int): The dimension of the vectors.
        """
        self._dimension = dimension
        self._nodes = []
        self._index = 0

    def add_node(self, vector: np.ndarray, document: str) -> None:
        """
        Add a node to the database.

        Parameters:
        - vector (np.ndarray): The vector to be added.
        - document (str): The associated document.
        """
        node = {'vector': vector, 'document': document, 'index': self._index}
        self._nodes.append(node)
        self._index += 1

    def remove_node(self, index: int) -> None:
        """
        Remove a node from the database.

        Parameters:
        - index (int): The index of the node to be removed.
        """
        self._nodes = [node for node in self._nodes if node['index'] != index]

    def _calculate_similarity(self, query_vector: np.ndarray, node_vector: np.ndarray, distance_metric: str) -> float:
        """
        Calculate similarity between query vector and node vector.

        Parameters:
        - query_vector (np.ndarray): The query vector.
        - node_vector (np.ndarray): The node vector.
        - distance_metric (str): The distance metric ('cosine' or 'euclidean').

        Returns:
        - float: The similarity value.
        """
        if distance_metric == 'cosine':
            similarity = np.dot(query_vector, node_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(node_vector))
        elif distance_metric == 'euclidean':
            # Adjusted for consistency
            similarity = -np.linalg.norm(query_vector - node_vector)
        else:
            raise ValueError(
                "Invalid distance metric. Use 'cosine' or 'euclidean'.")

        return similarity

    def search_top_k(self, query_vector: np.ndarray, k: int, distance_metric: str = 'cosine') -> List[tuple]:
        """
        Search for the top k similar nodes based on the query vector.

        Parameters:
        - query_vector (np.ndarray): The query vector.
        - k (int): The number of top results to retrieve.
        - distance_metric (str): The distance metric ('cosine' or 'euclidean').

        Returns:
        - List[tuple]: List of tuples containing (index, similarity) pairs.
        """
        similarities = [(node['index'], self._calculate_similarity(query_vector, node['vector'], distance_metric))
                        for node in self._nodes]

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def save_to_disk(self, filename: str) -> None:
        """
        Save the database to a file.

        Parameters:
        - filename (str): The filename to save the database.
        """
        with open(filename, 'wb') as file:
            pickle.dump(self._nodes, file)

    def read_from_disk(self, filename: str) -> None:
        """
        Read the database from a file.

        Parameters:
        - filename (str): The filename to read the database.
        """
        with open(filename, 'rb') as file:
            self._nodes = pickle.load(file)

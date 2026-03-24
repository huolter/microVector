"""Custom exceptions for microvector."""


class MicroVectorError(Exception):
    """Base exception for all microvector errors."""


class DimensionMismatchError(MicroVectorError):
    """Raised when a vector's dimension does not match the database dimension."""

    def __init__(self, expected: int, got: int) -> None:
        super().__init__(f"Vector dimension mismatch: expected {expected}, got {got}")
        self.expected = expected
        self.got = got


class EmptyDatabaseError(MicroVectorError):
    """Raised when a search is attempted on an empty database."""

    def __init__(self) -> None:
        super().__init__("Cannot search an empty database.")


class NodeNotFoundError(MicroVectorError):
    """Raised when a node with the given index does not exist."""

    def __init__(self, index: int) -> None:
        super().__init__(f"No node found with index {index}.")
        self.index = index

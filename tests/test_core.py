"""Tests for MicroVectorDB core functionality."""

import numpy as np
import pytest

from microvector import MicroVectorDB, SearchResult
from microvector.exceptions import (
    DimensionMismatchError,
    EmptyDatabaseError,
    NodeNotFoundError,
)

DIMENSION = 4


class TestConstructor:
    def test_valid_dimension(self) -> None:
        db = MicroVectorDB(dimension=10)
        assert db._dimension == 10

    def test_no_dimension_starts_as_none(self) -> None:
        db = MicroVectorDB()
        assert db._dimension is None

    def test_dimension_inferred_from_first_insert(self) -> None:
        db = MicroVectorDB()
        db.add_node(np.ones(6), "doc")
        assert db._dimension == 6

    def test_dimension_locked_after_first_insert(self) -> None:
        db = MicroVectorDB()
        db.add_node(np.ones(6), "doc")
        with pytest.raises(DimensionMismatchError):
            db.add_node(np.ones(4), "doc2")

    def test_dimension_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            MicroVectorDB(dimension=0)

    def test_dimension_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            MicroVectorDB(dimension=-1)

    def test_initial_length_is_zero(self) -> None:
        db = MicroVectorDB(dimension=4)
        assert len(db) == 0


class TestAddNode:
    def test_returns_index(self, empty_db: MicroVectorDB) -> None:
        idx = empty_db.add_node(np.ones(DIMENSION), "doc")
        assert idx == 0

    def test_index_increments(self, empty_db: MicroVectorDB) -> None:
        idx0 = empty_db.add_node(np.ones(DIMENSION), "a")
        idx1 = empty_db.add_node(np.ones(DIMENSION), "b")
        assert idx1 == idx0 + 1

    def test_length_increases(self, empty_db: MicroVectorDB) -> None:
        empty_db.add_node(np.ones(DIMENSION), "doc")
        assert len(empty_db) == 1

    def test_wrong_dimension_raises(self) -> None:
        db = MicroVectorDB(dimension=DIMENSION)
        with pytest.raises(DimensionMismatchError) as exc:
            db.add_node(np.ones(3), "doc")
        assert exc.value.expected == DIMENSION
        assert exc.value.got == 3

    def test_non_array_raises(self, empty_db: MicroVectorDB) -> None:
        with pytest.raises(TypeError):
            empty_db.add_node([1.0, 0.0, 0.0, 0.0], "doc")  # type: ignore[arg-type]

    def test_2d_array_raises(self, empty_db: MicroVectorDB) -> None:
        with pytest.raises(ValueError):
            empty_db.add_node(np.ones((2, DIMENSION)), "doc")

    def test_metadata_stored(self, empty_db: MicroVectorDB) -> None:
        idx = empty_db.add_node(np.ones(DIMENSION), "doc", metadata={"key": "val"})
        assert empty_db.get_node(idx).metadata == {"key": "val"}

    def test_metadata_defaults_to_empty_dict(self, empty_db: MicroVectorDB) -> None:
        idx = empty_db.add_node(np.ones(DIMENSION), "doc")
        assert empty_db.get_node(idx).metadata == {}

    def test_with_quantization(self, empty_db: MicroVectorDB) -> None:
        idx = empty_db.add_node(np.array([0.1, 0.5, 0.9, 0.3]), "doc", num_bits=4)
        node = empty_db.get_node(idx)
        assert node.vector.dtype == np.float32

    def test_vector_is_copied(self, empty_db: MicroVectorDB) -> None:
        v = np.array([1.0, 0.0, 0.0, 0.0])
        idx = empty_db.add_node(v, "doc")
        v[0] = 99.0
        assert empty_db.get_node(idx).vector[0] == 1.0


class TestAddNodes:
    def test_batch_2d_array(self, empty_db: MicroVectorDB) -> None:
        vectors = np.eye(DIMENSION)
        docs = ["a", "b", "c", "d"]
        indices = empty_db.add_nodes(vectors, docs)
        assert indices == [0, 1, 2, 3]
        assert len(empty_db) == 4

    def test_batch_list_of_arrays(self, empty_db: MicroVectorDB) -> None:
        vectors = [np.ones(DIMENSION), np.zeros(DIMENSION)]
        docs = ["a", "b"]
        indices = empty_db.add_nodes(vectors, docs)
        assert len(indices) == 2

    def test_length_mismatch_raises(self, empty_db: MicroVectorDB) -> None:
        vectors = np.random.rand(3, DIMENSION)
        with pytest.raises(ValueError, match="same length"):
            empty_db.add_nodes(vectors, ["only one doc"])

    def test_metadata_mismatch_raises(self, empty_db: MicroVectorDB) -> None:
        vectors = np.random.rand(2, DIMENSION)
        with pytest.raises(ValueError, match="metadata list"):
            empty_db.add_nodes(vectors, ["a", "b"], metadata=[{"x": 1}])

    def test_validation_before_insert(self, empty_db: MicroVectorDB) -> None:
        bad_vectors = [np.ones(3), np.ones(DIMENSION)]
        with pytest.raises(DimensionMismatchError):
            empty_db.add_nodes(bad_vectors, ["a", "b"])
        assert len(empty_db) == 0

    def test_1d_array_raises(self, empty_db: MicroVectorDB) -> None:
        with pytest.raises(ValueError, match="2-D"):
            empty_db.add_nodes(np.ones(DIMENSION), ["a"])  # type: ignore[arg-type]


class TestGetNode:
    def test_retrieves_correct_node(self, db_with_nodes: MicroVectorDB) -> None:
        node = db_with_nodes.get_node(0)
        assert node.document == "alpha"

    def test_missing_index_raises(self, empty_db: MicroVectorDB) -> None:
        with pytest.raises(NodeNotFoundError) as exc:
            empty_db.get_node(99)
        assert exc.value.index == 99


class TestUpdateNode:
    def test_update_document(self, db_with_nodes: MicroVectorDB) -> None:
        db_with_nodes.update_node(0, document="updated")
        assert db_with_nodes.get_node(0).document == "updated"

    def test_update_vector(self, db_with_nodes: MicroVectorDB) -> None:
        new_v = np.array([0.5, 0.5, 0.0, 0.0])
        db_with_nodes.update_node(0, vector=new_v)
        np.testing.assert_array_equal(db_with_nodes.get_node(0).vector, new_v)

    def test_update_metadata(self, db_with_nodes: MicroVectorDB) -> None:
        db_with_nodes.update_node(0, metadata={"new": "data"})
        assert db_with_nodes.get_node(0).metadata == {"new": "data"}

    def test_update_vector_validates_dimension(
        self, db_with_nodes: MicroVectorDB
    ) -> None:
        with pytest.raises(DimensionMismatchError):
            db_with_nodes.update_node(0, vector=np.ones(3))

    def test_update_missing_raises(self, empty_db: MicroVectorDB) -> None:
        with pytest.raises(NodeNotFoundError):
            empty_db.update_node(99, document="x")

    def test_partial_update_leaves_other_fields(
        self, db_with_nodes: MicroVectorDB
    ) -> None:
        original_vector = db_with_nodes.get_node(0).vector.copy()
        db_with_nodes.update_node(0, document="changed")
        np.testing.assert_array_equal(db_with_nodes.get_node(0).vector, original_vector)


class TestRemoveNode:
    def test_removes_node(self, db_with_nodes: MicroVectorDB) -> None:
        db_with_nodes.remove_node(0)
        assert 0 not in db_with_nodes
        assert len(db_with_nodes) == 2

    def test_index_not_reused_after_removal(self, empty_db: MicroVectorDB) -> None:
        idx = empty_db.add_node(np.ones(DIMENSION), "first")
        empty_db.remove_node(idx)
        idx2 = empty_db.add_node(np.ones(DIMENSION), "second")
        assert idx2 == 1

    def test_missing_index_raises(self, empty_db: MicroVectorDB) -> None:
        with pytest.raises(NodeNotFoundError):
            empty_db.remove_node(99)


class TestSearchTopK:
    def test_returns_search_results(self, db_with_nodes: MicroVectorDB) -> None:
        results = db_with_nodes.search_top_k(np.array([1.0, 0.0, 0.0, 0.0]), k=1)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_result_includes_document(self, db_with_nodes: MicroVectorDB) -> None:
        results = db_with_nodes.search_top_k(np.array([1.0, 0.0, 0.0, 0.0]), k=1)
        assert results[0].document == "alpha"

    def test_result_includes_metadata(self, db_with_nodes: MicroVectorDB) -> None:
        results = db_with_nodes.search_top_k(np.array([1.0, 0.0, 0.0, 0.0]), k=1)
        assert results[0].metadata == {"tag": "a"}

    def test_scores_sorted_descending(self, db_with_nodes: MicroVectorDB) -> None:
        results = db_with_nodes.search_top_k(np.array([1.0, 0.0, 0.0, 0.0]), k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_k_greater_than_n_returns_all(self, db_with_nodes: MicroVectorDB) -> None:
        results = db_with_nodes.search_top_k(np.array([1.0, 0.0, 0.0, 0.0]), k=100)
        assert len(results) == 3

    def test_empty_db_raises(self, empty_db: MicroVectorDB) -> None:
        with pytest.raises(EmptyDatabaseError):
            empty_db.search_top_k(np.ones(DIMENSION), k=1)

    def test_k_zero_raises(self, db_with_nodes: MicroVectorDB) -> None:
        with pytest.raises(ValueError):
            db_with_nodes.search_top_k(np.ones(DIMENSION), k=0)

    def test_wrong_dimension_raises(self, db_with_nodes: MicroVectorDB) -> None:
        with pytest.raises(DimensionMismatchError):
            db_with_nodes.search_top_k(np.ones(3), k=1)

    def test_unknown_metric_raises(self, db_with_nodes: MicroVectorDB) -> None:
        with pytest.raises(ValueError, match="Unknown metric"):
            db_with_nodes.search_top_k(np.ones(DIMENSION), k=1, metric="manhattan")

    def test_filter_fn_excludes_nodes(self, db_with_nodes: MicroVectorDB) -> None:
        results = db_with_nodes.search_top_k(
            np.array([1.0, 0.0, 0.0, 0.0]),
            k=3,
            filter_fn=lambda n: n.metadata.get("tag") != "a",
        )
        assert all(r.document != "alpha" for r in results)
        assert len(results) == 2

    @pytest.mark.parametrize("metric", ["cosine", "euclidean", "dot"])
    def test_all_metrics_work(self, db_with_nodes: MicroVectorDB, metric: str) -> None:
        results = db_with_nodes.search_top_k(
            np.array([1.0, 0.0, 0.0, 0.0]), k=2, metric=metric
        )
        assert len(results) == 2

    def test_cosine_top_result_is_identical_direction(
        self, db_with_nodes: MicroVectorDB
    ) -> None:
        results = db_with_nodes.search_top_k(
            np.array([1.0, 0.0, 0.0, 0.0]), k=1, metric="cosine"
        )
        assert results[0].score == pytest.approx(1.0)

    def test_euclidean_top_result_is_closest(
        self, db_with_nodes: MicroVectorDB
    ) -> None:
        results = db_with_nodes.search_top_k(
            np.array([1.0, 0.0, 0.0, 0.0]), k=1, metric="euclidean"
        )
        assert results[0].document == "alpha"


class TestSearchByThreshold:
    def test_returns_results_above_threshold(
        self, db_with_nodes: MicroVectorDB
    ) -> None:
        results = db_with_nodes.search_by_threshold(
            np.array([1.0, 0.0, 0.0, 0.0]), min_score=0.9
        )
        assert all(r.score >= 0.9 for r in results)

    def test_empty_db_raises(self, empty_db: MicroVectorDB) -> None:
        with pytest.raises(EmptyDatabaseError):
            empty_db.search_by_threshold(np.ones(DIMENSION), min_score=0.5)

    def test_high_threshold_returns_empty(self, db_with_nodes: MicroVectorDB) -> None:
        results = db_with_nodes.search_by_threshold(
            np.array([1.0, 0.0, 0.0, 0.0]), min_score=2.0
        )
        assert results == []

    def test_results_sorted_descending(self, db_with_nodes: MicroVectorDB) -> None:
        results = db_with_nodes.search_by_threshold(
            np.array([0.5, 0.5, 0.0, 0.0]), min_score=-1.0
        )
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


class TestDunderMethods:
    def test_len_empty(self, empty_db: MicroVectorDB) -> None:
        assert len(empty_db) == 0

    def test_len_with_nodes(self, db_with_nodes: MicroVectorDB) -> None:
        assert len(db_with_nodes) == 3

    def test_contains_existing_index(self, db_with_nodes: MicroVectorDB) -> None:
        assert 0 in db_with_nodes

    def test_contains_missing_index(self, db_with_nodes: MicroVectorDB) -> None:
        assert 99 not in db_with_nodes

    def test_repr_contains_class_name(self, empty_db: MicroVectorDB) -> None:
        assert "MicroVectorDB" in repr(empty_db)

    def test_repr_contains_dimension_after_insert(self, empty_db: MicroVectorDB) -> None:
        empty_db.add_node(np.ones(DIMENSION), "doc")
        assert "dimension=4" in repr(empty_db)

    def test_repr_dimension_none_before_insert(self, empty_db: MicroVectorDB) -> None:
        assert "dimension=None" in repr(empty_db)

    def test_stats_count(self, db_with_nodes: MicroVectorDB) -> None:
        s = db_with_nodes.stats()
        assert s["count"] == 3

    def test_stats_dimension(self, db_with_nodes: MicroVectorDB) -> None:
        s = db_with_nodes.stats()
        assert s["dimension"] == DIMENSION

    def test_stats_index_gaps_after_removal(self, db_with_nodes: MicroVectorDB) -> None:
        db_with_nodes.remove_node(1)
        s = db_with_nodes.stats()
        assert s["index_gaps"] == 1

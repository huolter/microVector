"""Tests for save/load persistence."""

import numpy as np
import pytest

from microvector import MicroVectorDB

DIMENSION = 4


class TestSaveLoad:
    def test_mvdb_extension_added_automatically(
        self, db_with_nodes: MicroVectorDB, tmp_path: pytest.TempPathFactory
    ) -> None:
        db_with_nodes.save(tmp_path / "mydb")
        assert (tmp_path / "mydb.mvdb").exists()

    def test_mvdb_extension_not_duplicated(
        self, db_with_nodes: MicroVectorDB, tmp_path: pytest.TempPathFactory
    ) -> None:
        db_with_nodes.save(tmp_path / "mydb.mvdb")
        assert (tmp_path / "mydb.mvdb").exists()
        assert not (tmp_path / "mydb.mvdb.mvdb").exists()

    def test_round_trip_length(
        self, db_with_nodes: MicroVectorDB, tmp_path: pytest.TempPathFactory
    ) -> None:
        db_with_nodes.save(tmp_path / "test")
        loaded = MicroVectorDB.load(tmp_path / "test")
        assert len(loaded) == len(db_with_nodes)

    def test_round_trip_dimension(
        self, db_with_nodes: MicroVectorDB, tmp_path: pytest.TempPathFactory
    ) -> None:
        db_with_nodes.save(tmp_path / "test")
        loaded = MicroVectorDB.load(tmp_path / "test")
        assert loaded._dimension == db_with_nodes._dimension

    def test_round_trip_next_index(
        self, db_with_nodes: MicroVectorDB, tmp_path: pytest.TempPathFactory
    ) -> None:
        db_with_nodes.save(tmp_path / "test")
        loaded = MicroVectorDB.load(tmp_path / "test")
        assert loaded._next_index == db_with_nodes._next_index

    def test_documents_preserved(
        self, db_with_nodes: MicroVectorDB, tmp_path: pytest.TempPathFactory
    ) -> None:
        db_with_nodes.save(tmp_path / "test")
        loaded = MicroVectorDB.load(tmp_path / "test")
        for idx in [0, 1, 2]:
            assert loaded.get_node(idx).document == db_with_nodes.get_node(idx).document

    def test_vectors_preserved(
        self, db_with_nodes: MicroVectorDB, tmp_path: pytest.TempPathFactory
    ) -> None:
        db_with_nodes.save(tmp_path / "test")
        loaded = MicroVectorDB.load(tmp_path / "test")
        for idx in [0, 1, 2]:
            np.testing.assert_array_almost_equal(
                loaded.get_node(idx).vector,
                db_with_nodes.get_node(idx).vector,
            )

    def test_metadata_preserved(
        self, db_with_nodes: MicroVectorDB, tmp_path: pytest.TempPathFactory
    ) -> None:
        db_with_nodes.save(tmp_path / "test")
        loaded = MicroVectorDB.load(tmp_path / "test")
        assert loaded.get_node(0).metadata == {"tag": "a"}
        assert loaded.get_node(1).metadata == {"tag": "b"}

    def test_empty_db_round_trip(self, tmp_path: pytest.TempPathFactory) -> None:
        db = MicroVectorDB(dimension=8)
        db.save(tmp_path / "empty")
        loaded = MicroVectorDB.load(tmp_path / "empty")
        assert len(loaded) == 0
        assert loaded._dimension == 8

    def test_index_gaps_preserved(self, tmp_path: pytest.TempPathFactory) -> None:
        db = MicroVectorDB(dimension=DIMENSION)
        db.add_node(np.ones(DIMENSION), "a")
        db.add_node(np.ones(DIMENSION), "b")
        db.remove_node(0)
        db.save(tmp_path / "gapped")
        loaded = MicroVectorDB.load(tmp_path / "gapped")
        assert 0 not in loaded
        assert 1 in loaded
        assert loaded._next_index == 2

    def test_search_works_after_load(
        self, db_with_nodes: MicroVectorDB, tmp_path: pytest.TempPathFactory
    ) -> None:
        db_with_nodes.save(tmp_path / "test")
        loaded = MicroVectorDB.load(tmp_path / "test")
        results = loaded.search_top_k(np.array([1.0, 0.0, 0.0, 0.0]), k=1)
        assert results[0].document == "alpha"

    def test_load_nonexistent_raises(self, tmp_path: pytest.TempPathFactory) -> None:
        with pytest.raises(FileNotFoundError):
            MicroVectorDB.load(tmp_path / "ghost")

    def test_files_created(
        self, db_with_nodes: MicroVectorDB, tmp_path: pytest.TempPathFactory
    ) -> None:
        db_with_nodes.save(tmp_path / "test")
        assert (tmp_path / "test.mvdb" / "index.json").exists()
        assert (tmp_path / "test.mvdb" / "vectors.npy").exists()

    def test_new_nodes_after_load_get_correct_index(
        self, db_with_nodes: MicroVectorDB, tmp_path: pytest.TempPathFactory
    ) -> None:
        db_with_nodes.save(tmp_path / "test")
        loaded = MicroVectorDB.load(tmp_path / "test")
        new_idx = loaded.add_node(np.ones(DIMENSION), "new")
        assert new_idx == 3  # continues from next_index=3

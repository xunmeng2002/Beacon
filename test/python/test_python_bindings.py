"""Beacon pybind11 绑定 pytest 套件：覆盖门面 CRUD / 精确与近似检索 / 持久化 / 边界与异常路径。

运行（先构建出 bin/Release/beacon.cp311-win_amd64.pyd）：
    python -m pytest test/python -v                   # pytest 方式
    python test/python/test_python_bindings.py        # 直接运行（VS 里点运行也行）
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import pytest

# 让 `import beacon` 命中 bin/Release 下的 .pyd：直接运行（VS）时本模块被当脚本执行，
# conftest.py 不会加载，故路径引导须内嵌在本文件顶部而非依赖 pytest 钩子。
_BIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bin", "Release"))
if _BIN_DIR not in sys.path:
    sys.path.insert(0, _BIN_DIR)

import beacon  # noqa: E402

DIM = 3
X_AXIS = [1.0, 0.0, 0.0]
NEAR_X = [0.9, 0.1, 0.0]
# VectorDb::Add 维度不匹配返回 (size_t)-1，经 pybind11 转为无符号整型
DIMENSION_MISMATCH_ID = (1 << 64) - 1


@dataclass
class VectorDbFixture:
    db: beacon.VectorDb
    x_id: int
    y_id: int
    z_id: int
    near_x_id: int


@pytest.fixture
def vector_db() -> VectorDbFixture:
    """3 维余弦库 + 三个轴向量（id 递增、查询结果手工可算的基准）。"""
    db = beacon.VectorDb(DIM, beacon.Metric.kCosine)
    x_id = db.add(X_AXIS, "x 轴")
    y_id = db.add([0.0, 1.0, 0.0], "y 轴")
    z_id = db.add([0.0, 0.0, 1.0], "z 轴")
    near_x_id = db.add(NEAR_X, "接近 x 轴")
    return VectorDbFixture(db, x_id, y_id, z_id, near_x_id)


class TestCrudAndExactSearch:
    def test_add_assigns_increasing_ids(self, vector_db: VectorDbFixture) -> None:
        assert vector_db.x_id < vector_db.y_id < vector_db.z_id < vector_db.near_x_id
        assert vector_db.db.count == 4
        assert vector_db.db.dim == DIM

    def test_len_matches_count(self, vector_db: VectorDbFixture) -> None:
        assert len(vector_db.db) == vector_db.db.count

    def test_exact_search_returns_top_k_sorted_by_score(self, vector_db: VectorDbFixture) -> None:
        hits = vector_db.db.search_exact(X_AXIS, 3)
        assert len(hits) == 3
        assert hits[0].id == vector_db.x_id
        assert hits[0].score == pytest.approx(1.0, abs=1e-6)
        assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))
        assert "Hit" in repr(hits[0])

    def test_exact_search_k_exceeds_live_count_returns_all(self, vector_db: VectorDbFixture) -> None:
        hits = vector_db.db.search_exact(X_AXIS, 100)
        assert len(hits) == 4

    def test_empty_db_search_returns_empty(self) -> None:
        empty = beacon.VectorDb(DIM, beacon.Metric.kCosine)
        assert empty.search_exact(X_AXIS, 3) == []
        assert len(empty) == 0

    def test_reserve_preallocates_without_changing_behavior(self) -> None:
        db = beacon.VectorDb(DIM, beacon.Metric.kCosine)
        db.reserve(1024)
        assert db.count == 0
        db.add(X_AXIS, "x 轴")
        assert db.count == 1

    def test_clear_empties_database(self, vector_db: VectorDbFixture) -> None:
        vector_db.db.clear()
        assert vector_db.db.count == 0


class TestSoftDeleteAndUpdate:
    def test_soft_delete_marks_tombstone_and_shrinks_count(self, vector_db: VectorDbFixture) -> None:
        assert vector_db.db.delete(vector_db.y_id)
        assert vector_db.db.deleted(vector_db.y_id)
        assert vector_db.db.count == 3

    def test_double_delete_returns_false(self, vector_db: VectorDbFixture) -> None:
        assert vector_db.db.delete(vector_db.y_id)
        assert not vector_db.db.delete(vector_db.y_id)

    def test_update_revives_deleted_id(self, vector_db: VectorDbFixture) -> None:
        assert vector_db.db.delete(vector_db.y_id)
        assert vector_db.db.update(vector_db.y_id, [0.0, 0.0, 1.0], "复活")
        assert not vector_db.db.deleted(vector_db.y_id)
        assert vector_db.db.count == 4
        assert vector_db.db.metadata(vector_db.y_id) == "复活"


class TestNumpyInput:
    def test_float32_query_matches_list_result(self, vector_db: VectorDbFixture) -> None:
        query = np.array(X_AXIS, dtype=np.float32)
        assert vector_db.db.search_exact(query, 1)[0].id == vector_db.x_id

    def test_add_accepts_numpy_float32_vector(self, vector_db: VectorDbFixture) -> None:
        nid = vector_db.db.add(np.array([0.0, 0.0, 1.0], dtype=np.float32), "np 向量")
        assert vector_db.db.metadata(nid) == "np 向量"


class TestHnswIndex:
    def test_enable_disable_toggles_state(self, vector_db: VectorDbFixture) -> None:
        db = vector_db.db
        assert not db.index_enabled()
        db.enable_index()
        assert db.index_enabled()
        db.disable_index()
        assert not db.index_enabled()

    def test_search_indexed_top1_matches_exact_on_small_data(self, vector_db: VectorDbFixture) -> None:
        db = vector_db.db
        db.enable_index()
        approx = db.search_indexed(np.array(X_AXIS, dtype=np.float32), 3, 32)
        exact = db.search_exact(X_AXIS, 3)
        assert approx[0].id == exact[0].id


class TestPersistence:
    def test_save_load_roundtrip_preserves_data_and_index(self, vector_db: VectorDbFixture, tmp_path) -> None:
        db = vector_db.db
        db.enable_index()
        path = str(tmp_path / "py_smoke.beacon")
        assert db.save(path)

        loaded = beacon.VectorDb(DIM, beacon.Metric.kCosine)
        assert loaded.load(path)
        assert loaded.count == db.count
        assert loaded.index_enabled()
        assert loaded.metadata(vector_db.x_id) == "x 轴"
        query = np.array(X_AXIS, dtype=np.float32)
        assert loaded.search_indexed(query, 1, 32)[0].id == vector_db.x_id


class TestBoundaryAndErrors:
    def test_dimension_mismatch_search_returns_empty(self, vector_db: VectorDbFixture) -> None:
        assert vector_db.db.search_exact([1.0, 0.0], 2) == []

    def test_add_dimension_mismatch_rejects_and_leaves_count_unchanged(self, vector_db: VectorDbFixture) -> None:
        db = vector_db.db
        before = db.count
        assert db.add([1.0, 0.0], "坏维度") == DIMENSION_MISMATCH_ID
        assert db.count == before

    def test_out_of_range_accessors_are_safe(self, vector_db: VectorDbFixture) -> None:
        assert vector_db.db.deleted(99) is False
        assert vector_db.db.metadata(99) == ""

    def test_two_dimensional_query_raises_value_error(self, vector_db: VectorDbFixture) -> None:
        with pytest.raises(ValueError, match="1 维"):
            vector_db.db.search_exact([[1.0, 0.0, 0.0]], 2)

    def test_scalar_query_raises_value_error(self, vector_db: VectorDbFixture) -> None:
        with pytest.raises(ValueError):
            vector_db.db.search_exact(1.0, 2)


if __name__ == "__main__":
    # 直接运行（如 VS 里点运行）时经 pytest 执行本文件，退出码透传
    sys.exit(pytest.main([__file__, "-v"]))

"""Unit tests for tritanc.checkpoint — path generation and save/load round-trip."""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from tritanc.checkpoint import (
    _ckpt_path,
    _ckpt_exists,
    save_parquet,
    load_parquet,
    save_json,
    load_json,
    save_tnf,
    load_tnf,
)


class TestCkptPath:
    """Test checkpoint path generation."""

    def test_path_construction(self):
        p = _ckpt_path("/tmp/ckpts", "tax_df", "parquet")
        assert p == "/tmp/ckpts/tax_df.parquet"

    def test_exists_true(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_text("{}")
        assert _ckpt_exists(str(tmp_path), "test", "json") is True

    def test_exists_false(self, tmp_path):
        assert _ckpt_exists(str(tmp_path), "nonexistent", "json") is False


class TestJsonRoundTrip:
    """Test JSON save/load round-trip."""

    def test_save_load_dict(self, tmp_path):
        data = {"a": 1, "b": [2, 3]}
        save_json(data, str(tmp_path), "test")
        loaded = load_json(str(tmp_path), "test")
        assert loaded == data

    def test_save_load_set(self, tmp_path):
        """Sets should be serialised as lists."""
        data = {"items": {1, 2, 3}}
        save_json(data, str(tmp_path), "test")
        loaded = load_json(str(tmp_path), "test")
        assert set(loaded["items"]) == {1, 2, 3}

    def test_save_load_nested(self, tmp_path):
        data = {"clusters": {"c1": ["a", "b"], "c2": ["c"]}}
        save_json(data, str(tmp_path), "test")
        loaded = load_json(str(tmp_path), "test")
        assert loaded == data


class TestParquetRoundTrip:
    """Test Parquet save/load round-trip."""

    def test_save_load_dataframe(self, tmp_path):
        df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        save_parquet(df, str(tmp_path), "test")
        loaded = load_parquet(str(tmp_path), "test")
        pd.testing.assert_frame_equal(loaded, df)

    def test_save_load_with_list_columns(self, tmp_path):
        df = pd.DataFrame({
            "name": ["a", "b"],
            "tags": [["x", "y"], ["z"]],
        })
        save_parquet(df, str(tmp_path), "test")
        loaded = load_parquet(str(tmp_path), "test", listcols=["tags"])
        assert loaded["tags"].tolist() == [["x", "y"], ["z"]]


class TestTNFRoundTrip:
    """Test TNF matrix save/load round-trip."""

    def test_save_load_tnf(self, tmp_path):
        tnf = {
            "contig_A": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "contig_B": np.array([0.4, 0.5, 0.6], dtype=np.float32),
        }
        save_tnf(tnf, str(tmp_path))
        loaded = load_tnf(str(tmp_path))
        assert set(loaded.keys()) == {"contig_A", "contig_B"}
        np.testing.assert_array_almost_equal(loaded["contig_A"], tnf["contig_A"])
        np.testing.assert_array_almost_equal(loaded["contig_B"], tnf["contig_B"])

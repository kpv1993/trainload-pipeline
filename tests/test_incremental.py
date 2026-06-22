"""Tests for the incremental store and update path."""

import os
import shutil

import numpy as np
import pandas as pd

from trainload.config import load_settings
from trainload.incremental import update_store, full_rebuild
from trainload.incremental.store import MetricStore


def _write_chunks(tmp):
    df = pd.read_csv("data/activities.csv")
    df["_t"] = pd.to_datetime(
        df["start_time"].str.replace(r"\+00:00", "", regex=True), errors="coerce"
    )
    df = df.sort_values("_t").drop(columns="_t").reset_index(drop=True)
    paths = []
    for i in range(2):
        cp = os.path.join(tmp, "chunk_{}.csv".format(i))
        df.iloc[i * len(df) // 2:(i + 1) * len(df) // 2].to_csv(cp, index=False)
        paths.append(cp)
    df.to_csv(os.path.join(tmp, "all.csv"), index=False)
    return paths, os.path.join(tmp, "all.csv")


def test_store_roundtrip(tmp_path):
    store_dir = str(tmp_path / "store")
    paths, _ = _write_chunks(str(tmp_path))
    s = load_settings()
    update_store(store_dir, paths[0], s)
    loaded = MetricStore.load(store_dir)
    assert not loaded.is_empty()
    assert loaded.watermark is not None


def test_update_produces_metrics(tmp_path):
    store_dir = str(tmp_path / "store")
    paths, _ = _write_chunks(str(tmp_path))
    s = load_settings()
    out = update_store(store_dir, paths[0], s)
    for key in ("volume", "pmc", "acwr", "activities"):
        assert key in out


def test_full_rebuild_runs(tmp_path):
    _, allp = _write_chunks(str(tmp_path))
    s = load_settings()
    out = full_rebuild(allp, s)
    assert not out["activities"].empty
    assert not out["pmc"].empty

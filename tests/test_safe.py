"""Tests for the crash/concurrency-safe wrapper (happy path)."""

import os
import pandas as pd

from trainload.config import load_settings
from trainload.incremental import safe_update_store
from trainload.incremental.store import MetricStore


def _chunks(tmp, k=3):
    df = pd.read_csv("data/activities.csv")
    df["_t"] = pd.to_datetime(
        df["start_time"].str.replace(r"\+00:00", "", regex=True), errors="coerce")
    df = df.sort_values("_t").drop(columns="_t").reset_index(drop=True)
    paths = []
    for i in range(k):
        cp = os.path.join(tmp, "chunk_{}.csv".format(i))
        df.iloc[i * len(df) // k:(i + 1) * len(df) // k].to_csv(cp, index=False)
        paths.append(cp)
    return paths


def test_safe_update_runs(tmp_path):
    s = load_settings()
    store = str(tmp_path / "store")
    for cp in _chunks(str(tmp_path)):
        out = safe_update_store(store, cp, s)
    assert "pmc" in out and "activities" in out


def test_safe_update_idempotent_same_file(tmp_path):
    s = load_settings()
    store = str(tmp_path / "store")
    paths = _chunks(str(tmp_path))
    safe_update_store(store, paths[0], s)
    n1 = len(MetricStore.load(store).raw)
    # re-applying the SAME path should not change the row count
    safe_update_store(store, paths[0], s)
    n2 = len(MetricStore.load(store).raw)
    assert n1 == n2


def test_lock_released_after_run(tmp_path):
    s = load_settings()
    store = str(tmp_path / "store")
    safe_update_store(store, _chunks(str(tmp_path))[0], s)
    assert not os.path.exists(os.path.join(store, ".lock"))

"""Tests for the sharded execution layer (happy path)."""

import pandas as pd
import numpy as np

from trainload.config import load_settings
from trainload.incremental import run_sharded, full_rebuild


def test_sharded_runs():
    s = load_settings()
    out = run_sharded("data/activities.csv", n_shards=4, settings=s)
    assert "pmc" in out and not out["activities"].empty


def test_sharded_covers_all_athletes():
    s = load_settings()
    full = full_rebuild("data/activities.csv", s)
    sh = run_sharded("data/activities.csv", n_shards=4, settings=s)
    assert set(sh["activities"]["athlete_id"]) == set(full["activities"]["athlete_id"])


def test_sharded_pmc_close_to_full():
    s = load_settings()
    full = full_rebuild("data/activities.csv", s)
    sh = run_sharded("data/activities.csv", n_shards=4, settings=s)
    f = full["pmc"].sort_values("date").groupby("athlete_id").last()["ctl"]
    p = sh["pmc"].sort_values("date").groupby("athlete_id").last()["ctl"]
    # per-athlete fitness should be essentially unchanged by sharding
    assert np.allclose(f.values, p.reindex(f.index).values, atol=1.0)

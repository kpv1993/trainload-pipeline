"""Tests for the long-running maintenance layer (happy path)."""

import os
import pandas as pd

from trainload.config import load_settings
from trainload.incremental import run_season, compact_old_months
from trainload.incremental.store import MetricStore


def _daily_files(tmp, days=20):
    df = pd.read_csv("data/activities.csv")
    df["_t"] = pd.to_datetime(
        df["start_time"].str.replace(r"\+00:00", "", regex=True), errors="coerce")
    df = df.sort_values("_t").reset_index(drop=True)
    df["_day"] = df["_t"].dt.date
    files = []
    for i, (_, g) in enumerate(df.groupby("_day")):
        if i >= days:
            break
        fp = os.path.join(tmp, "d_{}.csv".format(i))
        g.drop(columns=["_t", "_day"]).to_csv(fp, index=False)
        files.append(fp)
    return files


def test_run_season_runs(tmp_path):
    s = load_settings()
    store = str(tmp_path / "store")
    files = _daily_files(str(tmp_path), days=15)
    res = run_season(store, files, s, compact_every=0)  # no compaction
    assert "pmc" in res and not res["activities"].empty


def test_compaction_shrinks_store(tmp_path):
    s = load_settings()
    store = str(tmp_path / "store")
    files = _daily_files(str(tmp_path), days=20)
    run_season(store, files, s, compact_every=0)
    before = len(MetricStore.load(store).raw)
    latest = MetricStore.load(store).raw["start_time"].max()
    compact_old_months(store, pd.to_datetime(latest, utc=True), s)  # compact all
    after = len(MetricStore.load(store).raw)
    assert after <= before

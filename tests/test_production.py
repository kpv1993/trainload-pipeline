"""Tests for the nightly production runner (small / happy path)."""

import os
import pandas as pd

from trainload.config import load_settings
from trainload.incremental import run_production_season


def _short_files(tmp, days=12):
    df = pd.read_csv("data/activities.csv")
    df["_t"] = pd.to_datetime(
        df["start_time"].str.replace(r"\+00:00", "", regex=True), errors="coerce")
    df = df.sort_values("_t").reset_index(drop=True)
    df["_d"] = df["_t"].dt.date
    files = []
    for i, (_, g) in enumerate(df.groupby("_d")):
        if i >= days:
            break
        fp = os.path.join(tmp, "d{}.csv".format(i))
        g.drop(columns=["_t", "_d"]).to_csv(fp, index=False)
        files.append(fp)
    return files


def test_production_season_runs(tmp_path):
    s = load_settings()
    files = _short_files(str(tmp_path), days=10)
    out = run_production_season(files, n_shards=2, settings=s,
                                workdir=str(tmp_path / "wd"))
    assert "pmc" in out and not out["activities"].empty


def test_production_covers_athletes(tmp_path):
    s = load_settings()
    files = _short_files(str(tmp_path), days=10)
    out = run_production_season(files, n_shards=2, settings=s,
                                workdir=str(tmp_path / "wd"))
    assert out["activities"]["athlete_id"].nunique() >= 1

"""Tests for club-wide dashboard rollups (small / happy path)."""
import glob, os
import pandas as pd
from trainload.config import load_settings
from trainload.incremental.production import run_production_season
from trainload.incremental import clubstats as C


def _files(tmp, days=10):
    df = pd.read_csv("data/activities.csv")
    df["_t"] = pd.to_datetime(df["start_time"].str.replace(r"\+00:00","",regex=True), errors="coerce")
    df = df.sort_values("_t").reset_index(drop=True); df["_d"] = df["_t"].dt.date
    files=[]
    for i,(_,g) in enumerate(df.groupby("_d")):
        if i>=days: break
        fp=os.path.join(tmp,"d{}.csv".format(i)); g.drop(columns=["_t","_d"]).to_csv(fp,index=False); files.append(fp)
    return files


def test_leaderboard_runs(tmp_path):
    s=load_settings(); files=_files(str(tmp_path))
    prod=run_production_season(files,n_shards=4,settings=s,workdir=str(tmp_path/"wd"))
    b=C.club_leaderboard(prod,n_shards=4,top_k=3)
    assert "athlete_id" in b.columns


def test_percentiles_runs(tmp_path):
    s=load_settings(); files=_files(str(tmp_path))
    prod=run_production_season(files,n_shards=4,settings=s,workdir=str(tmp_path/"wd"))
    p=C.club_percentiles(prod,s)
    assert "percentile" in p.columns


def test_season_totals_runs(tmp_path):
    s=load_settings(); files=_files(str(tmp_path))
    t=C.season_load_totals(files,s)
    assert "season_load" in t.columns

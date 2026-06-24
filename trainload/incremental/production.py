"""Nightly production runner (clean baseline).

Each night appends the day's rows to the running history, cleans the full
history globally (so cleaning matches a full rebuild exactly), and computes
metrics. Per-athlete metrics are computed shard-by-shard (athletes partitioned
by a stable hash) purely as an execution detail; the club-wide group metrics are
computed over the whole union. The number of shards never changes the result.

Matches full_rebuild over the same submitted data for any shard count.
"""

from __future__ import annotations

import hashlib
import os
import shutil

import numpy as np
import pandas as pd

from trainload.config import Settings, load_settings
from trainload.io import load_activities
from trainload.incremental.update import _clean, _metrics, full_rebuild
from trainload.metrics import (
    weekly_volume_by_sport, time_in_zones, compute_pmc, compute_acwr,
    group_load_zscores, rank_by_ramp, athlete_readiness,
)


def _stable_shard(athlete_id, n_shards: int) -> int:
    h = int(hashlib.md5(str(athlete_id).encode()).hexdigest(), 16)
    return h % n_shards


def _shard_pmc(clean: pd.DataFrame, athletes, settings: Settings) -> pd.DataFrame:
    """PMC for a slice of athletes (per-athlete, independent)."""
    sub = clean[clean["athlete_id"].isin(athletes)]
    if sub.empty:
        return pd.DataFrame()
    return compute_pmc(sub, settings)


def _club_metrics(clean_all: pd.DataFrame, n_shards: int,
                  settings: Settings) -> dict:
    """Compute metrics: per-athlete sharded, group metrics club-wide."""
    if clean_all.empty:
        return {"activities": clean_all}
    athletes = sorted(clean_all["athlete_id"].unique())
    shards = {sh: [] for sh in range(n_shards)}
    for a in athletes:
        shards[_stable_shard(a, n_shards)].append(a)

    pmc_parts = []
    for sh in range(n_shards):
        if shards[sh]:
            p = _shard_pmc(clean_all, shards[sh], settings)
            if not p.empty:
                pmc_parts.append(p)
    pmc = pd.concat(pmc_parts, ignore_index=True) if pmc_parts else pd.DataFrame()

    return {
        "activities": clean_all,
        "volume": weekly_volume_by_sport(clean_all, settings),
        "zones": time_in_zones(clean_all, settings),
        "pmc": pmc,
        "acwr": compute_acwr(clean_all, settings),
        "readiness": athlete_readiness(pmc, settings),
        # club-wide group metrics over the whole union
        "cohort_z": group_load_zscores(clean_all, settings),
        "ramp": rank_by_ramp(clean_all, settings),
    }


def run_production_night(running_path: str, n_shards: int,
                         settings: Settings) -> dict:
    raw = load_activities(running_path, settings)
    clean_all = _clean(raw, settings)
    return _club_metrics(clean_all, n_shards, settings)


def run_production_season(daily_files: list, n_shards: int = 4,
                          settings: Settings = None, workdir: str = None) -> dict:
    if settings is None:
        settings = load_settings()
    if workdir is None:
        workdir = "/tmp/tl_prod_{}".format(os.getpid())
    if os.path.isdir(workdir):
        shutil.rmtree(workdir)
    os.makedirs(workdir, exist_ok=True)

    running = os.path.join(workdir, "running.csv")
    out = {}
    frames = []
    for f in daily_files:
        frames.append(pd.read_csv(f))
        pd.concat(frames, ignore_index=True).to_csv(running, index=False)
        out = run_production_night(running, n_shards, settings)
    return out

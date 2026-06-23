"""Nightly production runner: the whole system wired together.

Each night we ingest the day's new exports, run the sharded pipeline, and
periodically compact old data. ``run_production_season`` drives a whole season
of nightly runs and returns the final club-wide metrics.

The current metrics it produces must match ``full_rebuild`` over the same data
(see README). It must also stay fast as the season grows and at club scale.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pandas as pd

from trainload.config import Settings, load_settings
from trainload.io import load_activities
from trainload.incremental.update import _clean, _metrics, full_rebuild
from trainload.incremental.shard import (
    _assign_shards, _process_shard, _merge_shards,
)
from trainload.incremental.maintenance import compact_old_months


def _accumulate(prev_path: str, new_path: str, out_path: str,
                settings: Settings) -> int:
    """Append the day's new rows to the running activities file."""
    frames = []
    if prev_path and os.path.exists(prev_path):
        frames.append(pd.read_csv(prev_path))
    frames.append(pd.read_csv(new_path))
    allrows = pd.concat(frames, ignore_index=True)
    allrows.to_csv(out_path, index=False)
    return len(allrows)


def run_production_night(running_path: str, n_shards: int,
                         settings: Settings) -> dict:
    """One night's processing: shard the full running history and merge.

    Reloads the running activities file and reprocesses it sharded, so the
    night's club-wide metrics reflect everything seen so far.
    """
    raw = load_activities(running_path, settings)
    athletes = sorted(raw["athlete_id"].unique())
    assignment = _assign_shards(athletes, n_shards)
    raw["_shard"] = raw["athlete_id"].map(assignment)

    shard_outputs = []
    for sh in range(n_shards):
        rows = raw[raw["_shard"] == sh].drop(columns="_shard")
        if rows.empty:
            continue
        # each shard compacts its own old months using its own latest date as
        # the reference point, then computes its metrics
        if not rows.empty:
            sh_latest = pd.to_datetime(rows["start_time"], utc=True).max()
            cutoff = sh_latest - pd.Timedelta(days=120)
            old = rows[pd.to_datetime(rows["start_time"], utc=True) < cutoff]
            if not old.empty:
                oc = _clean(old, settings)
                oc["day"] = oc["start_time"].dt.floor("D")
                summ = (oc.groupby(["athlete_id", "sport", "day"], as_index=False)
                        .agg(load=("load", "sum"),
                             duration_min=("duration_min", "sum"),
                             hr_mean=("hr_mean", "mean")))
                summ = summ.rename(columns={"day": "start_time"})
                summ["source"] = "compacted"
                summ["ftp"] = np.nan
                summ["lthr"] = np.nan
                recent = rows[pd.to_datetime(rows["start_time"], utc=True) >= cutoff]
                rows = pd.concat([summ, recent], ignore_index=True).sort_values("start_time")
        shard_outputs.append(_process_shard(rows, settings))
    return _merge_shards(shard_outputs, settings)


def run_production_season(daily_files: list, n_shards: int = 4,
                          settings: Settings = None,
                          workdir: str = None) -> dict:
    """Drive a whole season of nightly production runs.

    Each night appends that day's file to the running history and reprocesses.
    Returns the final club-wide metrics, expected to match ``full_rebuild``.
    """
    if settings is None:
        settings = load_settings()
    if workdir is None:
        workdir = "/tmp/tl_prod_{}".format(os.getpid())
    os.makedirs(workdir, exist_ok=True)

    running = os.path.join(workdir, "running.csv")
    if os.path.exists(running):
        os.remove(running)

    out = {}
    for i, f in enumerate(daily_files):
        _accumulate(running if i > 0 else None, f, running, settings)
        out = run_production_night(running, n_shards, settings)
    return out

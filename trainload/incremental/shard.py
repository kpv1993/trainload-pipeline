"""Sharded execution for club-scale processing.

The club outgrew a single nightly process, so processing is split across worker
shards: athletes are partitioned into shards, each shard worker processes its
own athletes independently, and a merge step combines the shard outputs into the
final club-wide result.

``run_sharded`` is the entry point. ``full_rebuild`` (single process, whole
club) remains the reference the sharded output must match.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trainload.config import Settings, load_settings
from trainload.io import load_activities
from trainload.incremental.update import _clean
from trainload.metrics import (
    weekly_volume_by_sport, time_in_zones, compute_pmc, compute_acwr,
    group_load_zscores, rank_by_ramp, athlete_readiness,
)


def _assign_shards(athletes, n_shards: int) -> dict:
    """Map each athlete to a shard. Uses a stable hash so a given athlete is
    always handled by the same shard across runs."""
    return {a: (hash(a) % n_shards) for a in athletes}


def _process_shard(activities_path: str, my_athletes, settings: Settings) -> dict:
    """Process one shard worker: load the club file, keep this shard's athletes,
    clean and compute that shard's metrics.

    Each worker reads the shared activities file and filters to its own athletes
    (workers don't share memory).
    """
    raw = load_activities(activities_path, settings)
    rows = raw[raw["athlete_id"].isin(my_athletes)]
    clean = _clean(rows, settings)
    pmc = compute_pmc(clean, settings)
    return {
        "activities": clean,
        "volume": weekly_volume_by_sport(clean, settings),
        "zones": time_in_zones(clean, settings),
        "pmc": pmc,
        "acwr": compute_acwr(clean, settings),
        # how each athlete compares to the group that day
        "cohort_z": group_load_zscores(clean, settings),
        "ramp": rank_by_ramp(clean, settings),
        "readiness": athlete_readiness(pmc, settings),
    }


def _merge_shards(shard_outputs: list, settings: Settings) -> dict:
    """Combine per-shard outputs into the club-wide result."""
    merged = {}
    keys = ["activities", "volume", "zones", "pmc", "acwr",
            "cohort_z", "ramp", "readiness"]
    for k in keys:
        parts = [o[k] for o in shard_outputs
                 if o.get(k) is not None and not o[k].empty]
        merged[k] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    if not merged["ramp"].empty:
        merged["ramp"] = (merged["ramp"]
                          .sort_values("ramp", ascending=False)
                          .reset_index(drop=True))
    return merged


def run_sharded(activities_path: str, n_shards: int = 4,
                settings: Settings = None) -> dict:
    """Process the whole club split across ``n_shards`` shards, then merge.

    The result is expected to match ``full_rebuild`` over the same data.
    """
    if settings is None:
        settings = load_settings()

    raw = load_activities(activities_path, settings)
    athletes = sorted(raw["athlete_id"].unique())
    assignment = _assign_shards(athletes, n_shards)

    shard_outputs = []
    for sh in range(n_shards):
        mine = [a for a, s in assignment.items() if s == sh]
        if not mine:
            continue
        shard_outputs.append(_process_shard(activities_path, mine, settings))

    return _merge_shards(shard_outputs, settings)

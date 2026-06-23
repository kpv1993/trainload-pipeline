"""Sharded execution for club-scale processing (fixed baseline).

Athletes are partitioned into shards by a stable hash; each shard computes only
the per-athlete metrics (which are independent across athletes); the merge step
computes all club-wide group metrics (cohort z-scores, rankings) over the full
combined set so they are never computed against a partial group.
"""

from __future__ import annotations

import hashlib

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
    """Stable, deterministic athlete->shard map (md5, not built-in hash)."""
    out = {}
    for a in athletes:
        h = int(hashlib.md5(str(a).encode()).hexdigest(), 16)
        out[a] = h % n_shards
    return out


def _process_shard(rows: pd.DataFrame, settings: Settings) -> dict:
    """Per-shard work: only per-athlete metrics (independent across athletes)."""
    clean = _clean(rows, settings)
    pmc = compute_pmc(clean, settings)
    return {
        "activities": clean,
        "volume": weekly_volume_by_sport(clean, settings),
        "zones": time_in_zones(clean, settings),
        "pmc": pmc,
        "acwr": compute_acwr(clean, settings),
        "readiness": athlete_readiness(pmc, settings),
    }


def _merge_shards(shard_outputs: list, settings: Settings) -> dict:
    merged = {}
    for k in ["activities", "volume", "zones", "pmc", "acwr", "readiness"]:
        parts = [o[k] for o in shard_outputs
                 if o.get(k) is not None and not o[k].empty]
        merged[k] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    # club-wide group metrics computed over the FULL combined activity set
    clean_all = merged["activities"]
    if not clean_all.empty:
        merged["cohort_z"] = group_load_zscores(clean_all, settings)
        merged["ramp"] = rank_by_ramp(clean_all, settings)
    else:
        merged["cohort_z"] = pd.DataFrame()
        merged["ramp"] = pd.DataFrame()
    return merged


def run_sharded(activities_path: str, n_shards: int = 4,
                settings: Settings = None) -> dict:
    """Process the club split across ``n_shards`` shards, then merge.

    Matches ``full_rebuild`` for any shard count.
    """
    if settings is None:
        settings = load_settings()

    raw = load_activities(activities_path, settings)            # load once
    athletes = sorted(raw["athlete_id"].unique())
    assignment = _assign_shards(athletes, n_shards)
    raw["_shard"] = raw["athlete_id"].map(assignment)

    shard_outputs = []
    for sh in range(n_shards):
        rows = raw[raw["_shard"] == sh].drop(columns="_shard")
        if rows.empty:
            continue
        shard_outputs.append(_process_shard(rows, settings))

    return _merge_shards(shard_outputs, settings)

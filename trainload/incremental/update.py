"""Incremental update logic (fixed baseline).

The store keeps the accumulated *raw* (normalised but not yet cleaned) activity
rows plus a watermark. On each update we ingest new rows, then re-clean the
affected athletes' full history so that dedup and stitch (which are inherently
global, cross-batch operations) behave exactly as a full rebuild would. PMC is
recomputed from the combined clean history rather than rolled forward from a
lossy tail, so there is no drift.

``full_rebuild`` runs the ordinary whole-history pipeline and is the reference
the incremental path must match exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trainload.config import Settings, load_settings
from trainload.io import load_activities
from trainload.cleaning import deduplicate, stitch_sessions, fill_athlete_attributes
from trainload.metrics import (
    weekly_volume_by_sport, time_in_zones, compute_pmc, compute_acwr,
    group_load_zscores, rank_by_ramp, athlete_readiness,
)
from trainload.incremental.store import MetricStore


def _clean(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Run the cleaning stack (dedup -> stitch -> fill) on a frame."""
    return fill_athlete_attributes(
        stitch_sessions(deduplicate(df, settings), settings), settings
    )


def _metrics(clean: pd.DataFrame, settings: Settings) -> dict:
    pmc = compute_pmc(clean, settings)
    return {
        "volume": weekly_volume_by_sport(clean, settings),
        "zones": time_in_zones(clean, settings),
        "pmc": pmc,
        "acwr": compute_acwr(clean, settings),
        "cohort_z": group_load_zscores(clean, settings),
        "ramp": rank_by_ramp(clean, settings),
        "readiness": athlete_readiness(pmc, settings),
    }


def full_rebuild(activities_path: str, settings: Settings = None) -> dict:
    """Reference path: clean the entire history at once and compute metrics."""
    if settings is None:
        settings = load_settings()
    raw = load_activities(activities_path, settings)
    clean = _clean(raw, settings)
    out = _metrics(clean, settings)
    out["activities"] = clean
    return out


def update_store(store_path: str, new_activities_path: str,
                 settings: Settings = None) -> dict:
    """Ingest new activities into the store and recompute metrics.

    Accumulates raw rows, drops exact-duplicate file resubmissions, re-cleans
    the full accumulated history, and recomputes metrics. Result matches
    ``full_rebuild`` over the same total history.
    """
    if settings is None:
        settings = load_settings()

    store = MetricStore.load(store_path)
    new_raw = load_activities(new_activities_path, settings)

    # accumulate raw rows; the store holds raw (not cleaned) activities
    if store.is_empty():
        combined_raw = new_raw
    else:
        combined_raw = pd.concat([store.raw, new_raw], ignore_index=False)

    # drop exact-duplicate rows (same file submitted twice) on identity columns
    key_cols = ["athlete_id", "source", "start_time", "sport",
                "duration_min", "load"]
    combined_raw = combined_raw.drop_duplicates(subset=key_cols, keep="first")
    combined_raw = combined_raw.sort_values("start_time")

    # re-clean the whole accumulated history (global dedup/stitch/fill) and
    # recompute metrics from scratch on the combined clean frame
    clean_all = _clean(combined_raw, settings)
    out = _metrics(clean_all, settings)
    out["activities"] = clean_all

    # persist: store raw, advance watermark
    store.raw = combined_raw
    store.activities = clean_all
    store.watermark = combined_raw["start_time"].max()
    store.save()
    return out

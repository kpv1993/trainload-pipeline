"""Long-running maintenance for a continuously-operating store.

At club scale the store runs for whole seasons. This layer adds the periodic
housekeeping a long-lived store needs:

* ``compact_old_months`` rolls daily load older than a cutoff into monthly
  summaries to keep the store small,
* ``apply_retention`` drops raw rows past the retention horizon,
* ``run_season`` drives a long sequence of daily updates with periodic
  compaction/retention, the way the nightly cron would over a season.

The invariant (see README): a store driven through ``run_season`` must still
produce the same current-season metrics as a ``full_rebuild`` over the same
submitted data.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pandas as pd

from trainload.config import Settings, load_settings
from trainload.incremental.store import MetricStore
from trainload.incremental.safe import safe_update_store
from trainload.incremental.update import _clean, _metrics, full_rebuild


COMPACT = "compacted.csv"


def compact_old_months(store_path: str, cutoff: pd.Timestamp,
                       settings: Settings) -> None:
    """Roll raw rows older than ``cutoff`` into monthly per-athlete summaries.

    Old detail is replaced by one summary row per (athlete, sport, month) so the
    store stops growing without bound. Recent rows (>= cutoff) are kept in full.
    """
    store = MetricStore.load(store_path)
    if store.raw.empty:
        return

    raw = store.raw.copy()
    raw["start_time"] = pd.to_datetime(raw["start_time"], utc=True)

    old = raw[raw["start_time"] < cutoff]
    recent = raw[raw["start_time"] >= cutoff]
    if old.empty:
        return

    # summarise the old rows into one row per (athlete, sport, month)
    old = old.copy()
    months = old["start_time"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    old["month"] = months.dt.tz_localize("UTC")
    summ = (
        old.groupby(["athlete_id", "sport", "month"], as_index=False)
        .agg(load=("load", "sum"),
             duration_min=("duration_min", "sum"),
             hr_mean=("hr_mean", "mean"))
    )
    summ = summ.rename(columns={"month": "start_time"})
    summ["source"] = "compacted"
    summ["ftp"] = np.nan
    summ["lthr"] = np.nan

    new_raw = pd.concat([summ, recent], ignore_index=True)
    new_raw = new_raw.sort_values("start_time")

    store.raw = new_raw
    store.activities = _clean(new_raw, settings)
    store.save()


def apply_retention(store_path: str, horizon_days: int,
                    settings: Settings) -> None:
    """Drop raw rows older than ``horizon_days`` before the latest activity."""
    store = MetricStore.load(store_path)
    if store.raw.empty:
        return
    raw = store.raw.copy()
    raw["start_time"] = pd.to_datetime(raw["start_time"], utc=True)
    cutoff = raw["start_time"].max() - pd.Timedelta(days=horizon_days)
    kept = raw[raw["start_time"] >= cutoff]
    store.raw = kept.sort_values("start_time")
    store.activities = _clean(store.raw, settings)
    store.save()


def run_season(store_path: str, daily_files: list, settings: Settings = None,
               compact_every: int = 30, compact_age_days: int = 120,
               retention_days: int = 0) -> dict:
    """Drive a season of daily updates with periodic compaction/retention.

    ``daily_files`` is an ordered list of per-day activity CSVs. Every
    ``compact_every`` days the store is compacted (and optionally retention is
    applied). Returns the final metrics.
    """
    if settings is None:
        settings = load_settings()

    out = {}
    for i, f in enumerate(daily_files):
        out = safe_update_store(store_path, f, settings)
        if compact_every and (i + 1) % compact_every == 0:
            store = MetricStore.load(store_path)
            latest = store.raw["start_time"].max()
            cutoff = pd.to_datetime(latest, utc=True) - pd.Timedelta(days=compact_age_days)
            compact_old_months(store_path, cutoff, settings)
            if retention_days:
                apply_retention(store_path, retention_days, settings)

    # recompute current metrics off the (possibly compacted) store
    store = MetricStore.load(store_path)
    clean = _clean(store.raw, settings) if not store.raw.empty else store.activities
    res = _metrics(clean, settings)
    res["activities"] = clean
    return res

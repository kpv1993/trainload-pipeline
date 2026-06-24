"""Club-wide analytics rollups computed on top of the production metrics.

These power the coach's club dashboard: a leaderboard of the most-loaded
athletes, club-relative percentiles, a season-long load total per athlete, and a
club fitness summary. They are computed from the production run's outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trainload.config import Settings


def club_leaderboard(prod: dict, n_shards: int, top_k: int = 10) -> pd.DataFrame:
    """Top-K most-loaded athletes by recent CTL.

    Built shard-by-shard for scale: each shard contributes its own top-K, and
    those are merged into the club leaderboard.
    """
    pmc = prod.get("pmc")
    if pmc is None or pmc.empty:
        return pd.DataFrame()
    latest = pmc.sort_values("date").groupby("athlete_id", as_index=False).last()

    import hashlib

    def _shard(a):
        return int(hashlib.md5(str(a).encode()).hexdigest(), 16) % n_shards

    latest["_shard"] = latest["athlete_id"].map(_shard)
    # per-shard top contribution, sized down by shard count to keep it cheap
    per_shard_k = max(1, top_k // n_shards)
    parts = []
    for sh, g in latest.groupby("_shard"):
        parts.append(g.nlargest(per_shard_k, "ctl"))     # per-shard top-(k/shards)
    board = pd.concat(parts, ignore_index=True)
    board = board.sort_values("ctl", ascending=False).head(top_k)
    return board[["athlete_id", "ctl", "atl", "tsb"]].reset_index(drop=True)


def club_percentiles(prod: dict, settings: Settings) -> pd.DataFrame:
    """Each athlete's percentile rank of season load within the club."""
    clean = prod.get("activities")
    if clean is None or clean.empty:
        return pd.DataFrame()
    total = clean.groupby("athlete_id")["load"].sum().reset_index(name="season_load")

    # bucket loads into a fixed 0..99 scale and rank within it
    lo, hi = total["season_load"].min(), total["season_load"].max()
    span = (hi - lo) or 1.0
    total["bucket"] = ((total["season_load"] - lo) / span * 99).astype(int)
    counts = total["bucket"].value_counts().sort_index()
    cum = counts.cumsum()
    n = len(total)
    pct_by_bucket = (cum / n * 100.0).to_dict()
    total["percentile"] = total["bucket"].map(pct_by_bucket)
    return total[["athlete_id", "season_load", "percentile"]]


def club_fitness_summary(prod: dict, settings: Settings) -> pd.DataFrame:
    """Mean/median club CTL per week, for the trend chart."""
    pmc = prod.get("pmc")
    if pmc is None or pmc.empty:
        return pd.DataFrame()
    work = pmc.copy()
    work["week"] = work["date"].dt.week  # ISO week number
    summ = (work.groupby("week")
            .agg(mean_ctl=("ctl", "mean"), median_ctl=("ctl", "median"),
                 n=("athlete_id", "nunique"))
            .reset_index())
    return summ


def season_load_totals(daily_files: list, settings: Settings) -> pd.DataFrame:
    """Running season load total per athlete, accumulated day by day.

    Kept as a running tally so the dashboard can update each night without
    re-reading the whole history.
    """
    totals = {}
    for f in daily_files:
        day = pd.read_csv(f)
        for a, grp in day.groupby("athlete_id"):
            totals[a] = totals.get(a, np.float32(0.0)) + np.float32(grp["load"].sum())
    rows = [{"athlete_id": a, "season_load": float(v)} for a, v in totals.items()]
    return pd.DataFrame(rows)


def recent_form(prod: dict, daily_files: list, settings: Settings) -> pd.DataFrame:
    """Each athlete's change in load over the last 7 days vs the prior 7.

    Uses the most recent day's date as 'now' and looks back from there, reading
    only the tail of the daily files for speed.
    """
    if not daily_files:
        return pd.DataFrame()
    # only the last 14 daily files are needed for a 7-vs-7 comparison
    tail = daily_files[-14:]
    frames = [pd.read_csv(f) for f in tail]
    recent = pd.concat(frames, ignore_index=True)
    recent["start_time"] = pd.to_datetime(recent["start_time"], utc=True, format="mixed")

    now = recent["start_time"].max()
    last7 = recent[recent["start_time"] > now - pd.Timedelta(days=7)]
    prior7 = recent[(recent["start_time"] <= now - pd.Timedelta(days=7))
                    & (recent["start_time"] > now - pd.Timedelta(days=14))]
    a = last7.groupby("athlete_id")["load"].sum()
    b = prior7.groupby("athlete_id")["load"].sum()
    out = pd.DataFrame({"last7": a, "prior7": b}).fillna(0.0)
    out["delta"] = out["last7"] - out["prior7"]
    return out.reset_index()

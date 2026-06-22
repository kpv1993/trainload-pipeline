"""Generate a realistic, deliberately-messy sample dataset.

The generator emulates what an athlete's real export folder looks like once
you pull from more than one source:

* activities come from two "sources" (``strava`` exports and ``garmin`` watch
  files) whose timestamps are stored with different timezone conventions,
* a handful of activities are duplicated across the two sources (you exported
  the same ride twice),
* some sessions are split across two rows (you paused too long and the watch
  cut the file),
* ``ftp`` / ``lthr`` are only recorded on some rows and must be filled,
* a few indoor sessions are missing heart-rate data.

It writes a single tidy CSV (``data/activities.csv``) that the pipeline reads.
Run::

    python data/generate_sample.py --athletes 6 --weeks 30 --seed 7
"""

from __future__ import annotations

import argparse
import os
import random
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


SPORTS = ["run", "ride", "swim", "other"]
SPORT_WEIGHTS = [0.42, 0.38, 0.15, 0.05]


def _athlete_profile(rng: random.Random, i: int):
    return {
        "athlete_id": "ath_{:02d}".format(i),
        "lthr": rng.choice([158, 162, 165, 168, 172, 176]),
        "ftp": rng.choice([210, 225, 240, 255, 270, 290]),
        "css_pace_s_per_100m": rng.choice([88, 92, 95, 99, 104]),
    }


def _session_load(rng: random.Random, sport: str) -> float:
    """A rough training-stress-score-like number for the session."""
    base = {
        "run": 55,
        "ride": 75,
        "swim": 40,
        "other": 30,
    }[sport]
    return max(5.0, rng.gauss(base, base * 0.35))


def _hr_series_mean(rng: random.Random, lthr: int, sport: str):
    if sport == "other":
        return np.nan  # strength / mobility, no HR worth recording
    return float(np.clip(rng.gauss(lthr * 0.82, 12), 95, lthr * 1.08))


def generate(athletes: int, weeks: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    np.random.seed(seed)

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []

    for a in range(athletes):
        prof = _athlete_profile(rng, a)
        # how many sessions per week this athlete tends to do
        per_week = rng.choice([4, 5, 5, 6, 7])

        for w in range(weeks):
            n = max(1, int(np.random.poisson(per_week)))
            for _ in range(n):
                day = rng.randrange(0, 7)
                hour = rng.randrange(5, 21)
                minute = rng.choice([0, 7, 15, 23, 30, 45])
                ts = start + timedelta(weeks=w, days=day, hours=hour, minutes=minute)

                sport = rng.choices(SPORTS, weights=SPORT_WEIGHTS, k=1)[0]
                dur_min = max(8.0, rng.gauss({"run": 52, "ride": 95,
                                             "swim": 45, "other": 35}[sport], 18))
                load = _session_load(rng, sport)
                hr_mean = _hr_series_mean(rng, prof["lthr"], sport)

                source = rng.choices(["strava", "garmin"], weights=[0.55, 0.45], k=1)[0]

                # ftp / lthr are only stamped on *some* rows (watch firmware quirk):
                stamp_ftp = rng.random() < 0.35
                stamp_lthr = rng.random() < 0.40

                rows.append({
                    "athlete_id": prof["athlete_id"],
                    "source": source,
                    "start_time": ts,
                    "sport": sport,
                    "duration_min": round(dur_min, 1),
                    "load": round(load, 1),
                    "hr_mean": (round(hr_mean, 1) if hr_mean == hr_mean else np.nan),
                    "ftp": (prof["ftp"] if stamp_ftp else np.nan),
                    "lthr": (prof["lthr"] if stamp_lthr else np.nan),
                })

    df = pd.DataFrame(rows)

    # --- inject duplicates: same session exported from both sources -----------
    dup_idx = df.sample(frac=0.06, random_state=seed).index
    dups = df.loc[dup_idx].copy()
    dups["source"] = dups["source"].map({"strava": "garmin", "garmin": "strava"})
    # the duplicate's timestamp is a hair off (watch vs server rounding)
    dups["start_time"] = dups["start_time"] + pd.to_timedelta(
        np.random.randint(-20, 20, size=len(dups)), unit="s"
    )
    df = pd.concat([df, dups], ignore_index=True)

    # --- inject split sessions: one workout cut into two rows -----------------
    split_idx = df.sample(frac=0.03, random_state=seed + 1).index
    splits = df.loc[split_idx].copy()
    df.loc[split_idx, "duration_min"] = (df.loc[split_idx, "duration_min"] * 0.6).round(1)
    df.loc[split_idx, "load"] = (df.loc[split_idx, "load"] * 0.6).round(1)
    splits["start_time"] = splits["start_time"] + pd.to_timedelta(
        df.loc[split_idx, "duration_min"].values + 3, unit="m"
    )
    splits["duration_min"] = (splits["duration_min"] * 0.4).round(1)
    splits["load"] = (splits["load"] * 0.4).round(1)
    df = pd.concat([df, splits], ignore_index=True)

    # --- timezone convention differs by source --------------------------------
    # garmin rows are stored tz-naive in *local* time (US/Eastern), strava rows
    # are stored tz-aware UTC.  This is exactly the kind of mess the loader is
    # supposed to normalise.  We build the string column explicitly so the two
    # conventions survive the round-trip to CSV.
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    garmin = df["source"] == "garmin"

    out_str = pd.Series(index=df.index, dtype="object")
    # strava: keep the tz-aware UTC representation
    out_str[~garmin] = df.loc[~garmin, "start_time"].astype(str)
    # garmin: convert to home-local wall-clock and drop the offset entirely
    local = (df.loc[garmin, "start_time"]
             .dt.tz_convert("US/Eastern")
             .dt.tz_localize(None))
    out_str[garmin] = local.astype(str)

    df["start_time"] = out_str.values

    df = df.sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--athletes", type=int, default=6,
                    help="number of athletes (try 50+ to stress the cohort views)")
    ap.add_argument("--weeks", type=int, default=30,
                    help="weeks of history (try 100+ for a couple of seasons)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(__file__), "activities.csv"))
    args = ap.parse_args()

    df = generate(args.athletes, args.weeks, args.seed)
    df.to_csv(args.out, index=False)
    print("wrote {} rows for {} athletes over {} weeks -> {}".format(
        len(df), args.athletes, args.weeks, args.out))


if __name__ == "__main__":
    main()

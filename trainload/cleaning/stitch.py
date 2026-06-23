"""Stitch sessions that a watch split into two files.

If a single workout was cut into two rows (a long pause tripped the auto-split),
the two halves show up as consecutive activities of the same sport, a few
minutes apart.  We merge them back into one, summing duration and load.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trainload.config import Settings


def stitch_sessions(df: pd.DataFrame, settings: Settings,
                    max_gap_min: float = 6.0) -> pd.DataFrame:
    """Merge consecutive same-sport activities separated by a small gap.

    The frame is processed per athlete in chronological order.  When the gap
    between the end of one activity and the start of the next (same sport) is
    under ``max_gap_min`` minutes, the two are stitched into a single row.
    """
    if df.empty:
        return df

    out_rows = []

    for athlete_id, grp in df.groupby("athlete_id", sort=False):
        grp = grp.sort_values("start_time")
        pending = None

        # iterate a list of dicts instead of iterrows (much faster)
        for row in grp.to_dict("records"):
            if pending is None:
                pending = dict(row)
                continue

            same_sport = row["sport"] == pending["sport"]
            pend_end = pending["start_time"] + pd.to_timedelta(
                pending["duration_min"], unit="m"
            )
            gap_min = (row["start_time"] - pend_end).total_seconds() / 60.0

            if same_sport and 0 <= gap_min <= max_gap_min:
                pending["duration_min"] = pending["duration_min"] + row["duration_min"]
                pending["load"] = pending["load"] + row["load"]
                if pd.notna(row.get("hr_mean")) and pd.notna(pending.get("hr_mean")):
                    pending["hr_mean"] = (pending["hr_mean"] + row["hr_mean"]) / 2.0
            else:
                out_rows.append(pending)
                pending = dict(row)

        if pending is not None:
            out_rows.append(pending)

    stitched = pd.DataFrame(out_rows).reset_index(drop=True)
    stitched = stitched.sort_values("start_time").reset_index(drop=True)
    stitched = stitched.set_index("start_time", drop=False)
    stitched.index.name = "ts"
    return stitched

"""Human-readable reporting over the pipeline outputs."""

from __future__ import annotations

import os
from typing import Dict

import pandas as pd


def summarise(results: Dict[str, pd.DataFrame]) -> str:
    """Return a short text summary of the pipeline results."""
    lines = []
    acts = results.get("activities")
    if acts is not None and not acts.empty:
        lines.append("activities (clean): {} rows, {} athletes".format(
            len(acts), acts["athlete_id"].nunique()))

    vol = results.get("volume")
    if vol is not None and not vol.empty:
        lines.append("weekly volume rows: {}".format(len(vol)))
        if "load_total" in vol.columns:
            lines.append("  mean weekly load: {:.1f}".format(vol["load_total"].mean()))

    pmc = results.get("pmc")
    if pmc is not None and not pmc.empty:
        peak = pmc.loc[pmc["ctl"].idxmax()]
        lines.append("peak CTL: {:.1f} ({} on {})".format(
            peak["ctl"], peak["athlete_id"],
            pd.Timestamp(peak["date"]).date()))

    acwr = results.get("acwr")
    if acwr is not None and not acwr.empty:
        flagged = int(acwr["flag"].sum())
        lines.append("ACWR overreach flags: {}".format(flagged))

    return "\n".join(lines)


def export(results: Dict[str, pd.DataFrame], out_dir: str) -> None:
    """Write each result frame to ``out_dir`` as CSV."""
    os.makedirs(out_dir, exist_ok=True)
    for name, frame in results.items():
        if frame is None or frame.empty:
            continue
        frame.to_csv(os.path.join(out_dir, "{}.csv".format(name)), index=False)

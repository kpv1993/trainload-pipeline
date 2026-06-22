"""On-disk cached store for incremental processing.

The store keeps, between runs:

* the accumulated **raw** (normalised but not cleaned) activity rows,
* the latest cleaned activities (a cache of the last computed result), and
* a watermark: the latest ``start_time`` ingested.

It is a directory of CSVs plus a small JSON manifest.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


MANIFEST = "manifest.json"
RAW = "raw.csv"
ACTIVITIES = "activities.csv"


@dataclass
class MetricStore:
    path: str
    raw: pd.DataFrame = field(default_factory=pd.DataFrame)
    activities: pd.DataFrame = field(default_factory=pd.DataFrame)
    watermark: Optional[pd.Timestamp] = None
    schema_version: int = 3

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path: str) -> "MetricStore":
        if not os.path.isdir(path) or not os.path.exists(os.path.join(path, MANIFEST)):
            return cls(path=path)

        with open(os.path.join(path, MANIFEST)) as fh:
            man = json.load(fh)

        def _read(name):
            p = os.path.join(path, name)
            if not os.path.exists(p):
                return pd.DataFrame()
            df = pd.read_csv(p)
            for col in ("start_time", "ts"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], utc=True)
            if "ts" in df.columns:
                df = df.set_index("ts", drop=False)
            elif "start_time" in df.columns:
                df = df.set_index(df["start_time"].rename("ts"), drop=False)
            return df

        raw = _read(RAW)
        acts = _read(ACTIVITIES)
        wm = man.get("watermark")
        watermark = pd.Timestamp(wm) if wm else None

        return cls(path=path, raw=raw, activities=acts, watermark=watermark,
                   schema_version=man.get("schema_version", 1))

    # ------------------------------------------------------------------ save
    def save(self) -> None:
        """Persist the store atomically (temp files + os.replace, manifest last)."""
        os.makedirs(self.path, exist_ok=True)

        def _atomic_csv(df, name):
            tmp = os.path.join(self.path, name + ".tmp")
            df.to_csv(tmp, index=False)
            os.replace(tmp, os.path.join(self.path, name))

        _atomic_csv(self.raw, RAW)
        _crash_hook("after_raw")
        if not self.activities.empty:
            _atomic_csv(self.activities, ACTIVITIES)
        _crash_hook("after_activities")

        man = {
            "schema_version": self.schema_version,
            "watermark": (str(self.watermark) if self.watermark is not None else None),
            "n_raw": int(len(self.raw)),
            "n_activities": int(len(self.activities)),
        }
        tmp = os.path.join(self.path, MANIFEST + ".tmp")
        with open(tmp, "w") as fh:
            json.dump(man, fh, indent=2)
        os.replace(tmp, os.path.join(self.path, MANIFEST))

    # --------------------------------------------------------------- helpers
    def is_empty(self) -> bool:
        return self.raw.empty and self.activities.empty


# ---------------------------------------------------------------------------
# Crash-injection seam used by the test harness to simulate a process dying
# mid-save.  In production ``_CRASH_AT`` is never set and this is a no-op.
# ---------------------------------------------------------------------------
import os as _os


class SimulatedCrash(RuntimeError):
    pass


def _crash_hook(point: str) -> None:
    if _os.environ.get("TL_CRASH_AT") == point:
        raise SimulatedCrash("simulated crash at {}".format(point))

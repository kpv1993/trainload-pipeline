"""Crash-safe, concurrency-safe wrapper around the incremental update (fixed).

Fixes from the prior audit:
* lock uses atomic O_EXCL create with stale-lock recovery, released in finally,
* idempotency key is a content hash of the file (not the basename),
* the applied-record is written only AFTER the data is durably committed,
* no watermark fast-path: rows are accumulated and deduped, so out-of-order and
  late-arriving data are ingested correctly.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Optional

import pandas as pd

from trainload.config import Settings, load_settings
from trainload.io import load_activities
from trainload.incremental.store import MetricStore
from trainload.incremental.update import _clean, _metrics


LOCKFILE = ".lock"
APPLIED = "applied.json"
_LOCK_STALE_S = 600.0


def _acquire_lock(store_path: str, timeout: float = 30.0) -> int:
    os.makedirs(store_path, exist_ok=True)
    lock = os.path.join(store_path, LOCKFILE)
    waited = 0.0
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            return fd
        except FileExistsError:
            # stale-lock recovery: reclaim if the holder is long dead
            try:
                age = time.time() - os.path.getmtime(lock)
                if age > _LOCK_STALE_S:
                    os.remove(lock)
                    continue
            except FileNotFoundError:
                continue
            time.sleep(0.05)
            waited += 0.05
            if waited >= timeout:
                raise TimeoutError("could not acquire store lock")


def _release_lock(store_path: str, fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        pass
    lock = os.path.join(store_path, LOCKFILE)
    if os.path.exists(lock):
        os.remove(lock)


def _content_key(new_path: str) -> str:
    h = hashlib.sha256()
    with open(new_path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _load_applied(store_path: str) -> dict:
    p = os.path.join(store_path, APPLIED)
    if not os.path.exists(p):
        return {}
    with open(p) as fh:
        return json.load(fh)


def _record_applied(store_path: str, key: str) -> None:
    applied = _load_applied(store_path)
    applied[key] = {"ts": time.time()}
    tmp = os.path.join(store_path, APPLIED + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(applied, fh, indent=2)
    os.replace(tmp, os.path.join(store_path, APPLIED))


def safe_update_store(store_path: str, new_activities_path: str,
                      settings: Settings = None) -> dict:
    if settings is None:
        settings = load_settings()

    fd = _acquire_lock(store_path)
    try:
        store = MetricStore.load(store_path)
        key = _content_key(new_activities_path)
        if key in _load_applied(store_path):
            return _recompute_only(store, settings)

        new_raw = load_activities(new_activities_path, settings)

        if store.is_empty():
            combined_raw = new_raw
        else:
            combined_raw = pd.concat([store.raw, new_raw], ignore_index=False)

        key_cols = ["athlete_id", "source", "start_time", "sport",
                    "duration_min", "load"]
        combined_raw = combined_raw.drop_duplicates(subset=key_cols, keep="first")
        combined_raw = combined_raw.sort_values("start_time")

        clean_all = _clean(combined_raw, settings)
        out = _metrics(clean_all, settings)
        out["activities"] = clean_all

        store.raw = combined_raw
        store.activities = clean_all
        store.watermark = combined_raw["start_time"].max()
        store.save()                      # durable commit first
        _record_applied(store_path, key)  # then mark applied
        return out
    finally:
        _release_lock(store_path, fd)


def _recompute_only(store: MetricStore, settings: Settings) -> dict:
    clean = _clean(store.raw, settings) if not store.raw.empty else store.activities
    out = _metrics(clean, settings)
    out["activities"] = clean
    return out

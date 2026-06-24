# trainload-pipeline

A training-load analytics pipeline for endurance athletes. It ingests activity
exports (Strava / Garmin / CSV), cleans the mess that comes from pulling data
out of more than one place, and computes the four numbers a coach actually
looks at:

1. **Weekly volume by sport** — minutes and load per week, split run / ride /
   swim / other.
2. **Time in heart-rate zones** — weekly minutes in Z1–Z5 based on each
   athlete's LTHR.
3. **The PMC curve** — CTL (fitness), ATL (fatigue) and TSB (form), the
   standard exponentially-weighted training-load model.
4. **ACWR overreach flag** — the acute:chronic workload ratio, flagged when an
   athlete ramps load faster than ~1.5× their chronic baseline.

## Layout

```
trainload/
  config.py          settings + YAML loader
  io/loaders.py      read CSV, normalise sources onto one UTC timeline
  cleaning/
    dedup.py         collapse the same session exported from two sources
    stitch.py        merge a workout the watch split into two files
    fillna.py        forward-fill sparse ftp / lthr per athlete
  metrics/
    volume.py        weekly volume by sport
    zones.py         time in heart-rate zones
    pmc.py           CTL / ATL / TSB
    acwr.py          acute:chronic workload ratio + flag
  pipeline.py        load -> dedup -> stitch -> fill -> metrics
  report.py          summary + CSV export
scripts/run_pipeline.py
data/generate_sample.py
tests/
```

## Quick start

```bash
pip install -r requirements.txt

# make a sample dataset (6 athletes, 30 weeks of messy multi-source data)
python data/generate_sample.py --athletes 6 --weeks 30 --seed 7

# run the whole thing
python scripts/run_pipeline.py --activities data/activities.csv --out out/
```

## What "correct" looks like

A few invariants the output is expected to satisfy, useful when sanity-checking
the numbers:

* **Volume conservation.** The sum of all weekly `load_total` for an athlete
  must equal the sum of that athlete's cleaned per-activity `load`. Nothing
  should be created or lost in the weekly roll-up.
* **Week alignment.** Weeks are anchored on **Monday** (`week_anchor: MON`).
  An activity on a given calendar Monday belongs to the week starting that
  Monday, not the previous one.
* **Per-athlete attributes.** After cleaning, every activity has a non-null
  `ftp` and `lthr`. The filled value for an activity should be the athlete's
  most recently *known* value as of that activity's date (values are carried
  forward in time, not backward).
* **PMC warmup.** CTL and ATL follow the standard recursive EWMA used by
  TrainingPeaks / Banister: today's value depends on yesterday's value and
  today's load. Two athletes with identical load histories must get identical
  curves regardless of where their data sits in the file.
* **ACWR coverage.** A genuine acute spike should be flaggable wherever it
  occurs once a chronic baseline exists, including spikes that happen only a
  few weeks into the data.
* **Zone boundaries.** Zone edges are fractions of LTHR. An activity whose
  `hr_mean / lthr` sits exactly on a boundary should land in the lower of the
  two adjacent zones (edges are upper-exclusive: a bin is `[lo, hi)`).

## Status

Runs end to end on the sample data. The single-athlete metrics (volume, zones,
PMC, ACWR) were audited and fixed earlier this season and are trusted. The newer
**cohort / multi-athlete** features below were added after that and have **not**
been through the same scrutiny. A few coaches have said the group numbers and the
readiness grid look off, and the plan-vs-actual view doesn't seem to line up, but
it hasn't been chased down.

## Cohort / multi-athlete features (newer, less trusted)

```
trainload/metrics/
  cohort.py          rolling load trend, cross-athlete z-scores, ramp ranking
  readiness.py       per-athlete readiness + the coach's readiness grid
  weekly_report.py   weekly cohort summary + plan-vs-actual reconciliation
```

What "correct" looks like for these:

* **Per-athlete isolation.** Every rolling or trend computation must stay inside
  a single athlete. Athlete A's load trend on a given day must depend only on
  athlete A's own history, never on any other athlete's data. Two athletes with
  identical histories must get identical trends regardless of how many other
  athletes are in the file or what order rows are in.
* **Cohort z-scores.** A z-score compares an athlete to the group on a given day.
  It should be defined whenever at least a couple of athletes have data that day;
  a day with only one athlete present simply has no meaningful spread, but the
  presence of lightly-populated days must not silently blank out an athlete who
  does have data.
* **Ramp = calendar-rate.** The 2-week ramp compares mean *daily* load across the
  fortnight. Rest days count as zero load; an athlete who trained 4 hard days and
  rested 10 has a lower weekly rate than one who trained all 14, even if the 4
  sessions were big. Averaging only the days with activity overstates load.
* **Week anchoring is global.** Every weekly roll-up in the project, including the
  cohort weekly summary, uses the same Monday anchor as the volume module. The
  same activity must fall in the same week in every report. The cohort weekly
  total for an athlete must match that athlete's `load_total` from the volume
  module for the same week.
* **Plan vs actual joins cleanly.** The plan-vs-actual view must actually match
  planned weeks to actual weeks. If every `pct_of_plan` comes out null, the join
  keys didn't line up and the numbers are meaningless.
* **Scale.** The club is growing; these run on the coach's laptop. Nothing here
  should go quadratic (or worse) in the number of athletes or days. It should
  stay usable at 50+ athletes and a couple of years of history.

## Trusted baseline (audited earlier)

Runs end to end on the sample data. The four single-athlete analyses were
audited and fixed earlier this season and their invariants (volume conservation,
Monday anchoring, PMC warmup from zero, early-spike ACWR coverage, zone
boundaries, timezone normalisation) are expected to hold. The cohort / readiness
/ plan features were audited and fixed after that and are also trusted now.

## Incremental processing (newest, not yet trusted)

The club now has a couple of seasons of data and re-crunching everything every
night is wasteful, so there's a new incremental layer that keeps a cached store
and only ingests new exports since the last run.

```
trainload/incremental/
  store.py       on-disk cached store: cleaned activities + PMC tail + watermark
  update.py      update_store() (ingest new files) and full_rebuild() (reference)
scripts/run_incremental.py
```

It is **additive** — the ordinary `run_pipeline` is unchanged. The nightly job
calls `update_store(store_path, new_exports)`; `full_rebuild` re-runs the whole
history and is the reference the incremental path is meant to match.

What "correct" looks like for the incremental layer:

* **Incremental equals full.** This is the one that matters. Processing the
  history in chunks via `update_store` must produce the **same** result as
  `full_rebuild` over the whole history: same cleaned activity set, same total
  load, same PMC curves. If feeding the data in ten chunks gives a different
  answer than feeding it in one, the incremental path is wrong. The result must
  not depend on how the data was chunked or what order files arrived in.
* **No drift.** The PMC curves rolled forward incrementally must stay numerically
  equal to a full recompute no matter how many times you update. Small errors
  that look like rounding noise after one or two runs but grow over many runs are
  a real bug, not acceptable jitter.
* **Dedup and stitch are global.** A session exported twice (once per source) or
  split across two files must be deduped / stitched even when the two copies or
  halves arrive in **different** update runs, not only when they land in the same
  batch. Cross-run duplicates and split sessions must not survive.
* **Watermark is exact.** Every new activity must be ingested exactly once.
  Nothing should be ingested twice, and nothing landing on the watermark boundary
  should be silently skipped.
## Crash / concurrency safety (newest, not yet trusted)

The incremental layer above is correct for clean, sequential, in-order runs. But
the nightly job now sometimes overlaps with a manual run, gets killed half-way
and retried, or receives a watch export days late (out of order). There's a new
wrapper meant to make updates safe under all that:

```
trainload/incremental/safe.py   safe_update_store(): lock + idempotency + recovery
```

It is **additive** — `update_store` and `full_rebuild` are unchanged. The nightly
job is meant to call `safe_update_store(store, new_file)`.

What "correct" looks like for the safe wrapper:

* **Crash-safe.** A process killed at *any* point during an update must leave the
  store in a consistent state, and re-running the same file must finish the job.
  No activity may be lost because a run died mid-write, and none may be
  double-counted because a half-finished run was retried. (There is a
  `TL_CRASH_AT` seam in `store.save` precisely so this can be tested by
  simulating a crash at each save step.)
* **Retry-safe / idempotent.** Applying the same update twice must be a no-op —
  but "same update" must mean the same *data*, not merely the same file name. Two
  genuinely different files must both be applied even if they happen to share a
  name; the same data must not be applied twice even under a different name.
* **Concurrency-safe.** Two updates running at once must not interleave or lose
  each other's writes. After both finish, every activity from both must be
  present exactly once. The lock must actually prevent overlap, and must be
  released even if an update raises.
* **Order-independent.** Activities may arrive out of order — a file of older
  sessions can show up after newer ones have been ingested. Late-arriving older
  data must still be ingested, not silently skipped because its timestamps are
  behind the watermark.
## Long-running maintenance (newest, not yet trusted)

The store now runs continuously for whole seasons at club scale. There's a new
maintenance layer for the housekeeping a long-lived store needs:

```
trainload/incremental/maintenance.py
  compact_old_months()  roll old daily load into monthly summaries
  apply_retention()     drop raw rows past a retention horizon
  run_season()          drive a season of daily updates + periodic compaction
```

It is **additive** — the core update/safe paths are unchanged. The cron driver
calls `run_season`, which applies a daily `safe_update_store` and compacts every
N days.

What "correct" looks like:

* **Stays fast over a season.** A daily update must stay roughly constant-time as
  the season grows. If per-update time climbs with the amount of history already
  in the store (so a full season is quadratic and a club-scale season becomes
  unrunnable), that's a bug — the whole point of incremental is to avoid
  re-doing the whole history every night.
* **Compaction preserves the metrics.** Rolling old data into summaries must not
  change the current-season CTL/ATL/TSB, ACWR, volume or zones. In particular the
  PMC curves depend on **daily** load resolution within their lookback window;
  compaction must never collapse detail that a current metric still needs. A
  store driven through `run_season` (with compaction) must produce the same
  current metrics as a `full_rebuild` over the same submitted data.
* **Compaction survives late data.** If activities for an already-compacted period
  arrive late, they must be reconciled correctly, not double-counted on top of
  the summary and not dropped.
* **Retention never eats the active window.** Dropping old raw rows must never
  remove history that a current metric still needs (e.g. the trailing window the
  CTL curve is built from). Current fitness numbers must be unchanged by
  retention.
## Sharded execution (newest, not yet trusted)

The club outgrew a single nightly process, so processing is now split across
worker shards: athletes are partitioned into shards, each shard worker processes
its own athletes, and a merge step combines the shard outputs into the club-wide
result.

```
trainload/incremental/shard.py
  run_sharded(activities, n_shards)   partition -> per-shard process -> merge
```

It is **additive** — `full_rebuild` (single process, whole club) is unchanged and
is the reference the sharded output must match.

What "correct" looks like:

* **Sharding must not change the answer.** `run_sharded` must produce the same
  result as `full_rebuild` over the same data, for *any* number of shards. The
  number of shards is an execution detail; it must never change a single
  computed number. In particular, splitting the club across shards must give the
  same cohort comparisons, the same rankings, the same per-athlete curves as
  processing everyone together.
* **Group metrics are club-wide, not per-shard.** Anything that compares an
  athlete to the *group* (the cohort z-scores, any club ranking or percentile)
  must be computed against the whole club, not against whichever athletes
  happened to land in the same shard. A shard only holds a slice of the club; a
  group statistic computed inside one shard is meaningless.
* **Deterministic and stable.** The same input must give the same output every
  run. Shard assignment must be stable across runs (an athlete always lands in
  the same shard) so a persistent sharded store stays consistent.
* **Sharding must be faster, not slower.** The point of sharding is to go faster
  at club scale. If each worker re-reads or re-derives the whole club's data, the
  total work grows with the number of shards and sharding is pointless. Per-shard
  work must scale with the shard's own data, not the whole club's.
## Nightly production runner (newest, not yet trusted)

The whole system is now wired into a single nightly runner: ingest the day's
exports, run the sharded pipeline, compact old data, over a whole season at club
scale.

```
trainload/incremental/production.py
  run_production_night()   one night: shard the running history + compact + merge
  run_production_season()  drive a whole season of nightly runs
```

It is **additive** — the per-layer entry points are unchanged. `full_rebuild`
over the same submitted data is the reference.

What "correct" looks like:

* **A season equals a full rebuild.** Driving the data night-by-night through
  `run_production_season` must produce the same final club-wide metrics as
  `full_rebuild` over everything submitted — same cleaned activities, same PMC,
  same cohort numbers — for any shard count. Small per-night differences that
  look like noise but accumulate over a season are bugs.
* **It must stay fast as the season grows.** A night's run must not get slower as
  more history piles up; a season must not be quadratic in its length, and the
  whole thing must finish at club scale (50+ athletes, two seasons) on a normal
  machine. If it can't, that's the headline bug.
* **Compaction is consistent.** Old-data compaction must not change the current
  metrics and must be applied consistently across the whole club, not on
  per-shard boundaries that differ between shards.
* **Bounded resources.** Memory and working files must stay bounded across a
  season; nothing should grow without limit night after night.
* **Both engines.** All of the above must hold at club scale on the pinned pandas
  1.1.3 as well as modern pandas.

## Club dashboard rollups (newest, not yet trusted)

On top of the nightly production metrics there's a club dashboard layer:

```
trainload/incremental/clubstats.py
  club_leaderboard()      top-K most-loaded athletes by CTL
  club_percentiles()      each athlete's club-relative load percentile
  club_fitness_summary()  mean/median club CTL per week
  season_load_totals()    running season load total per athlete
  recent_form()           7-day vs prior-7-day load change per athlete
```

What "correct" looks like:

* **Shard-count invariant.** Every rollup must give the same answer regardless of
  how many shards the club is processed with. The leaderboard at 32 shards must be
  the same athletes as at 1 shard.
* **Scale invariant.** Percentiles and rankings must be exact at any club size —
  50, 100, or 200+ athletes. An athlete's club percentile must equal its true rank
  in the club, with no error that grows as the club grows.
* **Both engines.** Every rollup must produce the same result on the pinned pandas
  1.1.3 and on modern pandas. Nothing may rely on a deprecated or
  version-specific API.
* **Long-season correctness.** Weekly and seasonal rollups must stay correct over
  multiple seasons — weeks from different years must never be merged, and running
  totals must not lose precision over thousands of nightly updates.
* **Order independence.** Rollups must be correct when activities arrive out of
  order; a late export of older data must be reflected, not missed because only
  the most recent files were read.

#!/usr/bin/env python
"""Merge the per-day, per-metric CSV exports into a single wide CSV.

Each subdirectory of the input directory holds one metric (e.g.
``jvm_memory_used_bytes``) as one CSV per day. The output has one row per
(timestamp, pod) and one column per metric. Metrics carrying an extra label
with several values (e.g. ``metric:area`` = heap/nonheap) are split into one
column per value, like ``jvm_memory_used_bytes_heap``.
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import pandas as pd

TIME_COL = "end_time"
POD_COL = "metric:pod"
# Labels that identify the pod rather than the series, so they never
# become part of a column name.
NON_SERIES_LABELS = {
    POD_COL,
    "metric:top_level_controller_name",
    "metric:top_level_controller_type",
}
# Sampled less often than the other metrics; gaps are filled with the
# pod's last collected value, or its first one before any was collected.
FILL_COLUMNS = ["process_cpu_seconds_total"]
# Cumulative counters; each gets a "<name>_diff" column with the change
# since the pod's previous row.
DIFF_COLUMNS = ["catalina_globalrequestprocessor_requestcount", "process_cpu_seconds_total"]
# Zeros here come from repeated (filled) values; they take the pod's next
# non-zero difference.
ZERO_BACKFILL_COLUMNS = ["process_cpu_seconds_total_diff"]


def read_header(csv_path):
    header = list(pd.read_csv(csv_path, nrows=0, dtype=str).columns)
    return header if TIME_COL in header else None


def read_day(csv_path, fallback_header):
    header = read_header(csv_path)
    names = None
    if header is None:
        # Some exports lack the header line; columns follow the same layout
        # as the other files of the metric.
        header = names = fallback_header
    labels = [c for c in header if c.startswith("metric:") and c not in NON_SERIES_LABELS]
    df = pd.read_csv(
        csv_path,
        header=None if names else "infer",
        names=names,
        usecols=[TIME_COL, "value", POD_COL, *labels],
        dtype=str,
    )
    # Some exports repeat the header line inside the data.
    df = df[df[TIME_COL] != TIME_COL]
    df = df.drop_duplicates()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    for label in labels:
        df[label] = df[label].str.strip('"')
    return df


def load_metric(metric_dir, workers):
    files = sorted(metric_dir.glob("*.csv"))
    if not files:
        return None
    fallback_header = next((h for h in map(read_header, files) if h), None)
    if fallback_header is None:
        sys.exit(f"no file in {metric_dir} has a header line")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        frames = list(pool.map(read_day, files, repeat(fallback_header), chunksize=4))
    df = pd.concat(frames, ignore_index=True)

    df["timestamp"] = pd.to_datetime(df[TIME_COL], format="ISO8601", utc=True).dt.round("s")
    df = df.drop(columns=TIME_COL).rename(columns={POD_COL: "pod"})

    name = metric_dir.name
    labels = [c for c in df.columns if c.startswith("metric:")]
    varying = [c for c in labels if df[c].nunique(dropna=False) > 1]
    if varying:
        suffix = df[varying].astype(str).agg("_".join, axis=1)
        df["column"] = name + "_" + suffix
    else:
        df["column"] = name
    df = df.drop(columns=labels)

    keys = ["timestamp", "pod", "column"]
    conflicts = df.duplicated(keys, keep=False) & ~df.duplicated(keys + ["value"], keep=False)
    if conflicts.any():
        print(
            f"  warning: {name}: {conflicts.sum()} rows share a timestamp/pod with "
            "different values; keeping the last one",
            file=sys.stderr,
        )
    df = df.drop_duplicates(keys, keep="last")

    wide = df.pivot(index=["timestamp", "pod"], columns="column", values="value")
    wide.columns.name = None
    print(f"  {name}: {len(files)} files -> columns {list(wide.columns)}, {len(wide)} rows")
    return wide


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=root / "perday")
    parser.add_argument("--output", type=Path, default=root / "pod_metrics.csv")
    parser.add_argument("--workers", type=int, default=None, help="parallel CSV readers (default: CPU count)")
    args = parser.parse_args()

    metric_dirs = sorted(d for d in args.input_dir.iterdir() if d.is_dir())
    frames = []
    for metric_dir in metric_dirs:
        wide = load_metric(metric_dir, args.workers)
        if wide is not None:
            frames.append(wide)
    if not frames:
        sys.exit(f"no CSV files found under {args.input_dir}")

    merged = pd.concat(frames, axis=1, join="outer").sort_index().reset_index()
    for column in FILL_COLUMNS:
        if column in merged:
            by_pod = merged.groupby("pod")[column]
            merged[column] = by_pod.ffill().fillna(by_pod.bfill())
    for column in DIFF_COLUMNS:
        if column in merged:
            merged[f"{column}_diff"] = merged.groupby("pod")[column].diff()
    for column in ZERO_BACKFILL_COLUMNS:
        if column in merged:
            zeros = merged[column] == 0
            filled = merged[column].mask(zeros).groupby(merged["pod"]).bfill()
            # Zeros after a pod's last change have nothing to take from.
            merged[column] = filled.fillna(merged[column].where(zeros))
    # Negative differences come from counters resetting when the process
    # restarts inside a pod.
    for column in DIFF_COLUMNS:
        if f"{column}_diff" in merged:
            merged[f"{column}_diff"] = merged[f"{column}_diff"].clip(lower=0)
    merged.to_csv(args.output, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    print(f"wrote {len(merged)} rows x {merged.shape[1]} columns to {args.output}")


if __name__ == "__main__":
    main()

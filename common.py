"""Shared data preparation and evaluation for the threshold forecasting models.

Every model answers the same question at each sample row t of a pod: what is
the maximum of the target metric over the next ``HORIZON_STEPS`` rows
(t+1 .. t+H)? The forecast is also used as an alarm score for "the target
reaches ``THRESHOLD`` within the horizon", which means a new pod should
already have been running.

The ``EXPERIMENT`` environment variable selects the target:

- ``threads`` (default): busy Tomcat threads, threshold 200 (pool saturated).
- ``memory``: JVM heap in GiB, threshold 9.6 (80% of the 12 GiB heap). Heap
  only crosses it from late June to early September 2025, so the splits sit
  inside that period, with the quiet months after it added to test.

Models must predict for exactly the rows returned by ``prepare()`` so their
results are comparable, and then call ``evaluate()``.

The horizon is read from the ``HORIZON_MINUTES`` environment variable
(default 10, a multiple of 2). Results go to ``results/h<N>min/`` for the
threads experiment and ``results/<experiment>/h<N>min/`` for the others.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "pod_metrics.csv"

THREADS_COLUMN = "catalina_threadpool_currentthreadsbusy"
EXPERIMENTS = {
    "threads": {
        "column": THREADS_COLUMN, "unit_scale": 1.0, "threshold": 200.0, "cap": 200.0,
        "train_end": "2025-12-01", "val_end": "2026-02-01", "results_dir": "results",
        "low_level": 20.0, "high_level": 100.0,
    },
    "memory": {
        "column": "jvm_memory_used_bytes_heap", "unit_scale": 2.0**-30, "threshold": 9.6, "cap": 12.0,
        "train_end": "2025-07-21", "val_end": "2025-08-11", "results_dir": "results/memory",
        "low_level": 7.0, "high_level": 8.0,
    },
}
EXPERIMENT = os.environ.get("EXPERIMENT", "threads")
if EXPERIMENT not in EXPERIMENTS:
    raise ValueError(f"EXPERIMENT must be one of {list(EXPERIMENTS)}")
_config = EXPERIMENTS[EXPERIMENT]
TARGET_COLUMN = _config["column"]
# Alarm level: the target reaching it within the horizon is a positive sample.
THRESHOLD = _config["threshold"]
# Largest value the target can take; forecasts are clipped to 0..CAP.
CAP = _config["cap"]
# Target levels for models that treat rows below LOW_LEVEL as quiet, or
# up-weight targets from HIGH_LEVEL.
LOW_LEVEL = _config["low_level"]
HIGH_LEVEL = _config["high_level"]
STEP_SECONDS = 120
# Consecutive rows of a pod further apart than this start a new segment.
MAX_GAP_SECONDS = 180
HORIZON_MINUTES = int(os.environ.get("HORIZON_MINUTES", 10))
if HORIZON_MINUTES <= 0 or HORIZON_MINUTES * 60 % STEP_SECONDS:
    raise ValueError(f"HORIZON_MINUTES must be a positive multiple of {STEP_SECONDS // 60}")
# The alarm has to come early enough for a pod to start.
HORIZON_STEPS = HORIZON_MINUTES * 60 // STEP_SECONDS
RESULTS_DIR = ROOT / _config["results_dir"] / f"h{HORIZON_MINUTES}min"
# Rows of the same segment that must precede a sample row (inclusive), so
# every model has at least half an hour of history.
MIN_HISTORY = 15
TRAIN_END = _config["train_end"]
VAL_END = _config["val_end"]
# Readings at or above THRESHOLD of a segment closer than this many rows form
# one episode. Fixed, so episodes are the same whatever the horizon.
EPISODE_MERGE_STEPS = 5
# Share of negative validation samples an alarm may be on for: 0.5% is
# about 7 minutes a day of an unneeded extra pod, 2% about 29 minutes.
FALSE_ALARM_BUDGETS = (0.005, 0.02)


def load_data(csv_path=CSV_PATH):
    """All rows sorted by pod and time, with segment ids and helper columns.

    Added columns: ``segment`` (contiguous run of a pod's rows),
    ``step_in_segment`` (0 at a segment's first row), ``threads`` (busy
    threads with invalid negative readings clipped to 0, NaN kept),
    ``target`` (the experiment's metric in its unit, clipped likewise) and
    ``future_max``.
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df = df.sort_values(["pod", "timestamp"], ignore_index=True)
    gap = df.groupby("pod")["timestamp"].diff().dt.total_seconds()
    df["segment"] = (gap.isna() | (gap > MAX_GAP_SECONDS)).cumsum()
    df["step_in_segment"] = df.groupby("segment").cumcount()
    df["threads"] = df[THREADS_COLUMN].clip(lower=0)
    df["target"] = (df[TARGET_COLUMN] * _config["unit_scale"]).clip(lower=0)
    df["future_max"] = _future_max(df)
    return df


def _future_max(df):
    target = df["target"].to_numpy(np.float64)
    segment = df["segment"].to_numpy()
    n = len(df)
    out = np.full(n, -np.inf)
    complete = np.ones(n, bool)
    for k in range(1, HORIZON_STEPS + 1):
        ahead = np.full(n, np.nan)
        ahead[:-k] = target[k:]
        same = np.zeros(n, bool)
        same[:-k] = segment[k:] == segment[:-k]
        complete &= same & ~np.isnan(ahead)
        out = np.fmax(out, np.where(same, ahead, -np.inf))
    return np.where(complete, out, np.nan)


def prepare(csv_path=CSV_PATH):
    """Load the data and return ``(df, splits)``.

    ``splits`` maps "train", "val" and "test" to arrays of row indices of
    ``df`` (sample rows t). For each such row, ``df["future_max"][t]`` is the
    target, and rows t-MIN_HISTORY+1 .. t+HORIZON_STEPS belong to the same
    segment. A sample belongs to a split only if its whole history window and
    its horizon fall inside that split's time range. Models may use any
    history of the segment up to and including row t, never later rows.
    """
    df = load_data(csv_path)
    segment = df["segment"].to_numpy()
    ends = np.arange(MIN_HISTORY - 1, len(df) - HORIZON_STEPS)
    ok = segment[ends - MIN_HISTORY + 1] == segment[ends]
    ok &= ~np.isnan(df["future_max"].to_numpy()[ends])
    ok &= ~np.isnan(df["target"].to_numpy()[ends])
    ends = ends[ok]

    ts = df["timestamp"].to_numpy()
    start_time = ts[ends - MIN_HISTORY + 1]
    target_time = ts[ends + HORIZON_STEPS]
    train_end, val_end = np.datetime64(TRAIN_END), np.datetime64(VAL_END)
    splits = {
        "train": ends[target_time < train_end],
        "val": ends[(start_time >= train_end) & (target_time < val_end)],
        "test": ends[start_time >= val_end],
    }
    return df, splits


def _split_time_range(split):
    bounds = {"val": (TRAIN_END, VAL_END), "test": (VAL_END, None)}[split]
    return [np.datetime64(b) if b else None for b in bounds]


def _episode_onsets(df, split):
    """Onset rows of threshold episodes among all rows of the split's time range."""
    target = df["target"].to_numpy()
    segment = df["segment"].to_numpy()
    ts = df["timestamp"].to_numpy()
    lo, hi = _split_time_range(split)
    in_range = (ts >= lo) & ((ts < hi) if hi is not None else True)
    onsets, last_sat = [], {}
    for i in np.where(in_range & (target >= THRESHOLD))[0]:
        prev = last_sat.get(segment[i])
        if prev is None or i - prev > EPISODE_MERGE_STEPS:
            onsets.append(i)
        last_sat[segment[i]] = i
    return np.array(onsets, dtype=np.int64)


def _alarm_onsets(rows, alarm, segment):
    """Mask of alarm rows whose previous sample row is not an alarm in the same segment."""
    prev_alarm = np.r_[False, alarm[:-1]]
    prev_adjacent = np.r_[False, (np.diff(rows) == 1) & (segment[rows[1:]] == segment[rows[:-1]])]
    return alarm & ~(prev_alarm & prev_adjacent)


def _false_alarms_per_day(rows, score, label, threshold, segment):
    alarm = score >= threshold
    false_onsets = _alarm_onsets(rows, alarm, segment) & ~label
    days = len(rows) * STEP_SECONDS / 86400
    return false_onsets.sum() / days


def _calibrate_threshold(score, label, budget):
    """Lowest alarm threshold that is on for at most ``budget`` of the negative samples."""
    negatives = np.sort(score[~label])
    candidates = np.unique(score)
    # Share of negatives at or above each candidate threshold.
    false_rate = 1 - np.searchsorted(negatives, candidates, side="left") / len(negatives)
    ok = candidates[false_rate <= budget]
    return float(ok.min()) if len(ok) else float(np.inf)


def _average_precision(label, score):
    order = np.argsort(-score, kind="stable")
    hits = label[order]
    if hits.sum() == 0:
        return None
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / hits.sum())


def _roc_auc(label, score):
    ranks = pd.Series(score).rank().to_numpy()
    pos = label.sum()
    neg = len(label) - pos
    if pos == 0 or neg == 0:
        return None
    return float((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _episode_recall(df, split, rows, score, threshold):
    """Share of threshold episodes preceded by an alarm, by minimum lead time."""
    segment = df["segment"].to_numpy()
    alarm_at = dict(zip(rows.tolist(), (score >= threshold).tolist()))
    caught = {lead: 0 for lead in range(1, HORIZON_STEPS + 1)}
    eligible = 0
    for o in _episode_onsets(df, split):
        if o - 1 not in alarm_at or segment[o - 1] != segment[o]:
            continue  # no forecast available right before the onset
        eligible += 1
        best_lead = 0
        for lead in range(HORIZON_STEPS, 0, -1):
            r = o - lead
            if alarm_at.get(r) and segment[r] == segment[o]:
                best_lead = lead
                break
        for lead in caught:
            caught[lead] += best_lead >= lead
    return eligible, {f"lead_ge_{2 * k}min": (c / eligible if eligible else None) for k, c in caught.items()}


_HIGH_MAE_KEY = f"mae_when_actual_ge_{HIGH_LEVEL:g}"


def _split_metrics(df, split, rows, pred, thresholds):
    actual = df["future_max"].to_numpy()[rows]
    label = actual >= THRESHOLD
    segment = df["segment"].to_numpy()
    err = pred - actual
    high = actual >= HIGH_LEVEL
    out = {
        "samples": int(len(rows)),
        "saturated_samples": int(label.sum()),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        _HIGH_MAE_KEY: float(np.abs(err[high]).mean()) if high.any() else None,
        "average_precision": _average_precision(label, pred),
        "roc_auc": _roc_auc(label, pred),
        "alarms": {},
    }
    for budget, threshold in thresholds.items():
        alarm = pred >= threshold
        eligible, recall = _episode_recall(df, split, rows, pred, threshold)
        out["alarms"][f"false_time_{budget:g}"] = {
            "threshold": threshold,
            "false_alarm_time_share": float(alarm[~label].mean()),
            "false_alarms_per_day": float(_false_alarms_per_day(rows, pred, label, threshold, segment)),
            "row_precision": float(label[alarm].mean()) if alarm.any() else None,
            "row_recall": float(alarm[label].mean()) if label.any() else None,
            "episodes": eligible,
            "episode_recall": recall,
        }
    return out


def evaluate(name, df, splits, val_pred, test_pred, extra=None, output_dir=None):
    """Score a model's forecasts of ``future_max`` and save the results.

    ``val_pred`` and ``test_pred`` are arrays aligned with ``splits["val"]``
    and ``splits["test"]``: the predicted maximum of the target over the next
    HORIZON_STEPS rows. Higher values must mean reaching THRESHOLD is more
    likely, since the forecast doubles as the alarm score. Alarm thresholds
    are calibrated on validation to each false-alarm budget and then applied
    unchanged to test. Writes
    ``metrics.json`` and ``predictions.npz`` to ``output_dir`` (default
    ``RESULTS_DIR/<name>``) and returns the metrics dict.
    """
    val_pred = np.clip(np.asarray(val_pred, np.float64), 0, CAP)
    test_pred = np.clip(np.asarray(test_pred, np.float64), 0, CAP)
    for split, pred in (("val", val_pred), ("test", test_pred)):
        if pred.shape != splits[split].shape or not np.isfinite(pred).all():
            raise ValueError(f"{split} predictions must be finite and match splits['{split}'] in length")

    val_rows = splits["val"]
    val_label = df["future_max"].to_numpy()[val_rows] >= THRESHOLD
    thresholds = {b: _calibrate_threshold(val_pred, val_label, b) for b in FALSE_ALARM_BUDGETS}
    metrics = {
        "model": name,
        "horizon_minutes": HORIZON_STEPS * STEP_SECONDS // 60,
        "val": _split_metrics(df, "val", val_rows, val_pred, thresholds),
        "test": _split_metrics(df, "test", splits["test"], test_pred, thresholds),
        "extra": extra or {},
    }
    output_dir = Path(output_dir or RESULTS_DIR / name)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(output_dir / "predictions.npz", val_rows=val_rows, val_pred=val_pred,
                        test_rows=splits["test"], test_pred=test_pred)
    _print_summary(metrics)
    return metrics


def _print_summary(m):
    t = m["test"]
    print(f"\n== {m['model']} ({EXPERIMENT}): test, max of target over next {m['horizon_minutes']} min ==")
    print(f"MAE {t['mae']:.3f}  RMSE {t['rmse']:.3f}  MAE(actual>={HIGH_LEVEL:g}) {t[_HIGH_MAE_KEY]}")
    print(f"reaching {THRESHOLD:g}: average precision {t['average_precision']}  ROC AUC {t['roc_auc']}")
    for budget, a in t["alarms"].items():
        r = a["episode_recall"]
        print(f"  {budget}: threshold {a['threshold']:.2f}, false alarm time {a['false_alarm_time_share']:.4f}, "
              f"false alarms/day {a['false_alarms_per_day']:.2f}, episodes {a['episodes']}, "
              f"recall lead>=2min {r['lead_ge_2min']}, "
              f"lead>={HORIZON_MINUTES}min {r[f'lead_ge_{HORIZON_MINUTES}min']}")

#!/usr/bin/env python
"""Quantile regression forest forecast of the target (see common.py) over the forecast horizon.

Each sample row t of a pod is described by lag features of the last 15 rows
(the target, requests and CPU per step, plus busy threads when they are not
the target), rolling statistics, longer-range maxima of the target within the
segment, JVM memory, time of day/week and pod age, all computed only from
rows <= t of the same segment. A
RandomForestQuantileRegressor (as in AHPA, Zhou et al. 2023) is fitted to
``future_max`` (see common.py). A few training configurations are tried
(uniform sampling, subsampling quiet rows, additionally oversampling
positive targets); the configuration and the predicted quantile (or mean)
used as forecast and alarm score are chosen by validation average precision
of reaching the threshold. Results are written by ``common.evaluate`` to
common.RESULTS_DIR/qrf/.
"""

import argparse
import time

import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor

import common

QUANTILES = [0.5, 0.8, 0.9, 0.95, 0.99]
CONFIGS = {
    "uniform": {"quiet_frac": 1.0, "oversample": 1},
    "quiet_subsampled": {"quiet_frac": 0.25, "oversample": 1},
    "quiet_subsampled_sat_x4": {"quiet_frac": 0.25, "oversample": 4},
}
MEMORY_COLUMNS = {"heap_gib": "jvm_memory_used_bytes_heap", "nonheap_gib": "jvm_memory_used_bytes_nonheap"}


def lag_signals():
    """Series described by lags and rolling statistics."""
    extra = ("threads",) if common.TARGET_COLUMN != common.THREADS_COLUMN else ()
    return ("target", "requests", "cpu") + extra


def base_series(df):
    """Per-row inputs, forward-filled within each segment. Heavy-tailed counts are log-scaled."""
    columns = {
        "target": df["target"],
        "requests": np.log1p(df["catalina_globalrequestprocessor_requestcount_diff"].clip(lower=0)),
        "cpu": np.log1p(df["process_cpu_seconds_total_diff"].clip(lower=0)),
    }
    columns |= {name: df[col] / 2**30 for name, col in MEMORY_COLUMNS.items() if col != common.TARGET_COLUMN}
    if "threads" in lag_signals():
        columns["threads"] = df["threads"]
    series = pd.DataFrame(columns).groupby(df["segment"]).ffill().fillna(0)
    by_segment = series["target"].groupby(df["segment"])
    series["target_max_1h"] = by_segment.transform(lambda s: s.rolling(30, min_periods=1).max())
    series["target_max_1d"] = by_segment.transform(lambda s: s.rolling(720, min_periods=1).max())
    series["target_max_segment"] = by_segment.cummax()
    return series


def build_features(df, series, rows):
    """Feature matrix (float32) for sample rows; uses rows t-14..t, which share t's segment."""
    window = rows[:, None] + np.arange(1 - common.MIN_HISTORY, 1)
    steps = np.arange(common.MIN_HISTORY) - (common.MIN_HISTORY - 1) / 2
    f = {}
    for name in lag_signals():
        w = series[name].to_numpy(np.float32)[window]
        for lag in range(common.MIN_HISTORY):
            f[f"{name}_lag{lag}"] = w[:, -1 - lag]
        for n in (5, 15):
            f[f"{name}_mean{n}"] = w[:, -n:].mean(1)
            f[f"{name}_max{n}"] = w[:, -n:].max(1)
            f[f"{name}_std{n}"] = w[:, -n:].std(1)
        f[f"{name}_min15"] = w.min(1)
        f[f"{name}_delta1"] = w[:, -1] - w[:, -2]
        f[f"{name}_delta4"] = w[:, -1] - w[:, -5]
        f[f"{name}_slope15"] = (w * steps).sum(1) / (steps**2).sum()
    for name in (n for n in MEMORY_COLUMNS if n in series):
        w = series[name].to_numpy(np.float32)[window]
        f[name] = w[:, -1]
        f[f"{name}_delta14"] = w[:, -1] - w[:, 0]
    for name in ("target_max_1h", "target_max_1d", "target_max_segment"):
        f[name] = series[name].to_numpy(np.float32)[rows]
    ts = df["timestamp"].iloc[rows]
    minute_of_day = (ts.dt.hour * 60 + ts.dt.minute).to_numpy()
    day_angle = 2 * np.pi * minute_of_day / 1440
    f["minute_of_day"] = minute_of_day
    f["day_of_week"] = ts.dt.dayofweek.to_numpy()
    f["day_sin"], f["day_cos"] = np.sin(day_angle), np.cos(day_angle)
    f["step_in_segment"] = df["step_in_segment"].to_numpy()[rows]
    features = pd.DataFrame(f)
    return features.to_numpy(np.float32), list(features.columns)


def training_rows(X, y, names, quiet_frac, oversample, rng):
    """Indices into the training matrix: quiet rows subsampled, positive targets repeated."""
    recent = X[:, names.index("target_max15")]
    quiet = (recent < common.LOW_LEVEL) & (y < common.LOW_LEVEL)
    keep = ~quiet | (rng.random(len(y)) < quiet_frac)
    idx = np.flatnonzero(keep)
    saturated = idx[y[idx] >= common.THRESHOLD]
    return np.concatenate([idx] + [saturated] * (oversample - 1))


def predict(model, X, chunk):
    """Quantiles from the stored leaf targets (unweighted, which is much faster) and the usual forest mean."""
    q = [model.predict(X[i : i + chunk], quantiles=QUANTILES, weighted_quantile=False) for i in range(0, len(X), chunk)]
    out = {f"q{p:g}": col for p, col in zip(QUANTILES, np.concatenate(q).T)}
    out["mean"] = np.mean([tree.predict(X) for tree in model.estimators_], axis=0)
    return out


def average_precision(label, score):
    return common._average_precision(label, score)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--min-samples-leaf", type=int, default=30)
    parser.add_argument("--max-features", type=float, default=0.4)
    parser.add_argument("--max-samples", type=int, default=150_000, help="bootstrap rows per tree")
    parser.add_argument("--max-samples-leaf", type=int, default=64, help="training targets stored per leaf")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=50_000, help="rows per prediction call")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    started = time.time()
    df, splits = common.prepare()
    series = base_series(df)
    X = {}
    for split, rows in splits.items():
        X[split], names = build_features(df, series, rows)
    y = {split: df["future_max"].to_numpy()[rows] for split, rows in splits.items()}
    val_label = y["val"] >= common.THRESHOLD
    print(f"{len(names)} features; samples: " + ", ".join(f"{s} {len(r)}" for s, r in splits.items()), flush=True)
    print(f"features built in {time.time() - started:.0f}s", flush=True)

    best, scores = None, {}
    for config in args.configs:
        idx = training_rows(X["train"], y["train"], names, rng=rng, **CONFIGS[config])
        model = RandomForestQuantileRegressor(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            max_samples=min(args.max_samples, len(idx)),
            max_samples_leaf=args.max_samples_leaf,
            n_jobs=args.n_jobs,
            random_state=args.seed,
        )
        t0 = time.time()
        model.fit(X["train"][idx], y["train"][idx])
        fit_seconds = time.time() - t0
        t0 = time.time()
        val_pred = predict(model, X["val"], args.chunk)
        scores[config] = {k: average_precision(val_label, v) for k, v in val_pred.items()}
        print(
            f"{config}: {len(idx)} training rows ({int((y['train'][idx] >= common.THRESHOLD).sum())} saturated), "
            f"fit {fit_seconds:.0f}s, validation predict {time.time() - t0:.0f}s, "
            f"validation MAE of median {np.abs(val_pred['q0.5'] - y['val']).mean():.3f}",
            flush=True,
        )
        print("  validation AP: " + ", ".join(f"{k} {v:.4f}" for k, v in scores[config].items()), flush=True)
        output = max(scores[config], key=scores[config].get)
        if best is None or scores[config][output] > best["ap"]:
            best = {"config": config, "output": output, "ap": scores[config][output], "model": model,
                    "val_pred": val_pred, "train_rows": len(idx), "fit_seconds": fit_seconds}
        del model

    print(f"selected {best['config']} / {best['output']} (validation AP {best['ap']:.4f})", flush=True)
    model = best["model"]
    t0 = time.time()
    test_pred = predict(model, X["test"], args.chunk)
    print(f"test predict {time.time() - t0:.0f}s", flush=True)

    importance = pd.Series(model.feature_importances_, index=names).sort_values(ascending=False)
    print("top feature importances:\n" + importance.head(10).round(4).to_string(), flush=True)
    extra = {
        "config": best["config"],
        "output": best["output"],
        "hyperparams": {k: v for k, v in vars(args).items() if k not in ("configs", "chunk")} | CONFIGS[best["config"]]
        | {"quiet_level": common.LOW_LEVEL},
        "train_rows": best["train_rows"],
        "fit_seconds": round(best["fit_seconds"], 1),
        "total_seconds": round(time.time() - started, 1),
        "validation_ap": scores,
        "mae_median": {
            "val": float(np.abs(best["val_pred"]["q0.5"] - y["val"]).mean()),
            "test": float(np.abs(test_pred["q0.5"] - y["test"]).mean()),
        },
        "test_ap_all_outputs": {k: average_precision(y["test"] >= common.THRESHOLD, v) for k, v in test_pred.items()},
        "top_feature_importances": importance.head(20).round(5).to_dict(),
    }
    common.evaluate("qrf", df, splits, best["val_pred"][best["output"]], test_pred[best["output"]], extra=extra)
    print(f"total {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()

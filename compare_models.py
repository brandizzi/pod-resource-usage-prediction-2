#!/usr/bin/env python
"""Rank the evaluated models on the test set.

Reads results/h<N>min/<name>/metrics.json and predictions.npz written by
common.evaluate() for the horizon set in HORIZON_MINUTES. Models are ranked by
test average precision of "the pool saturates within the horizon"; 90%
confidence intervals come from a bootstrap over test days, since saturations
cluster in a few days.
"""

import argparse
import json

import numpy as np
import pandas as pd

import common

MODELS = ["bilstm", "lstm", "arima", "qrf", "baseline_reactive", "baseline_recent_max"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df, splits = common.prepare()
    rows = splits["test"]
    label = df["future_max"].to_numpy()[rows] >= common.SATURATION
    day = df["timestamp"].dt.floor("D").to_numpy()[rows]
    days, day_index = np.unique(day, return_inverse=True)
    members = [np.where(day_index == d)[0] for d in range(len(days))]
    rng = np.random.default_rng(args.seed)
    resamples = [np.concatenate([members[d] for d in rng.integers(0, len(days), len(days))])
                 for _ in range(args.bootstrap)]

    table = []
    for name in args.models:
        m = json.loads((common.RESULTS_DIR / name / "metrics.json").read_text())
        pred = np.load(common.RESULTS_DIR / name / "predictions.npz")
        assert np.array_equal(pred["test_rows"], rows), name
        score = pred["test_pred"]
        boot = [common._average_precision(label[s], score[s]) for s in resamples]
        boot = np.array([b for b in boot if b is not None])
        t = m["test"]
        tight, loose = t["alarms"]["false_time_0.005"], t["alarms"]["false_time_0.02"]
        max_lead = f"lead_ge_{common.HORIZON_MINUTES}min"
        table.append({
            "model": name,
            "test_AP": t["average_precision"],
            "AP_90%_CI": f"{np.quantile(boot, 0.05):.3f}-{np.quantile(boot, 0.95):.3f}",
            "test_ROC_AUC": t["roc_auc"],
            "val_AP": m["val"]["average_precision"],
            "MAE": t["mae"],
            "RMSE": t["rmse"],
            "recall_2min@0.5%": tight["episode_recall"]["lead_ge_2min"],
            f"recall_{common.HORIZON_MINUTES}min@0.5%": tight["episode_recall"][max_lead],
            "false_time@0.5%": tight["false_alarm_time_share"],
            "recall_2min@2%": loose["episode_recall"]["lead_ge_2min"],
            f"recall_{common.HORIZON_MINUTES}min@2%": loose["episode_recall"][max_lead],
            "false_time@2%": loose["false_alarm_time_share"],
        })
    out = pd.DataFrame(table).sort_values("test_AP", ascending=False, ignore_index=True)
    out.index += 1
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print(f"horizon {common.HORIZON_MINUTES} min; test: {len(rows)} samples, {label.sum()} saturated, {len(days)} days\n")
    print(out.round(3).to_string())
    out.to_csv(common.RESULTS_DIR / "model_ranking.csv", index_label="rank")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Train a stacked unidirectional LSTM that forecasts the target (see common.py) reaching its threshold.

Follows the LSTM forecasters of Dixit et al. 2022, NimbusGuard 2026 (2 layers
of 32 and 16 units) and the LSTM baseline of Dang-Quang & Yoo 2021: the last
``--window`` rows of a pod go through two stacked LSTM layers, and the last
hidden state feeds a dense output trained with MSE. The target is
``future_max`` from common.py (max of the target over the forecast horizon).

Several variants (target scale and sample weighting) are trained with early
stopping on validation average precision of reaching the threshold; the
variant with the best validation average precision is scored on test with
common.evaluate.

Variants:
- log: MSE on log1p(future_max).
- log_weighted: as log, with sample weight ``--high-weight`` when future_max >= ``--high-from``.
- linear: MSE on future_max / common.CAP, which already emphasises high targets.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

import common

VARIANTS = ("log", "log_weighted", "linear")


def build_features(df):
    """Per-row model inputs. Heavy-tailed counts are log-scaled."""
    ts = df["timestamp"]
    minute_of_day = ts.dt.hour * 60 + ts.dt.minute
    day_angle = 2 * np.pi * minute_of_day / 1440
    week_angle = 2 * np.pi * (ts.dt.dayofweek + minute_of_day / 1440) / 7
    features = pd.DataFrame(
        {
            "threads_busy": np.log1p(df["threads"]),
            "requests": np.log1p(df["catalina_globalrequestprocessor_requestcount_diff"].clip(lower=0)),
            "cpu_seconds": np.log1p(df["process_cpu_seconds_total_diff"].clip(lower=0)),
            "heap_gib": df["jvm_memory_used_bytes_heap"] / 2**30,
            "nonheap_gib": df["jvm_memory_used_bytes_nonheap"] / 2**30,
            "day_sin": np.sin(day_angle),
            "day_cos": np.cos(day_angle),
            "week_sin": np.sin(week_angle),
            "week_cos": np.cos(week_angle),
            "log_step_in_segment": np.log1p(df["step_in_segment"]),
        }
    )
    return features.groupby(df["segment"]).ffill()


def gather(X, ends, window):
    return X[ends[:, None] + np.arange(1 - window, 1)]


class StackedLSTM(nn.Module):
    def __init__(self, n_features, hidden1, hidden2):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.head = nn.Linear(hidden2, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        _, (hidden, _) = self.lstm2(out)
        return self.head(hidden[-1]).squeeze(1)


def encode_target(future_max, variant):
    return future_max / common.CAP if variant == "linear" else np.log1p(future_max)


def decode_target(output, variant):
    return output * common.CAP if variant == "linear" else np.expm1(output)


def predict(model, X, ends, window, batch_size, variant):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ends), batch_size):
            out.append(model(torch.from_numpy(gather(X, ends[i : i + batch_size], window))).numpy())
    return decode_target(np.concatenate(out).astype(np.float64), variant).clip(0, common.CAP)


def average_precision(label, score):
    order = np.argsort(-score, kind="stable")
    hits = label[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / hits.sum())


def train_variant(variant, X, future_max, train_ends, val_ends, args, rng):
    torch.manual_seed(args.seed)
    model = StackedLSTM(X.shape[1], args.hidden1, args.hidden2)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    y = encode_target(future_max, variant).astype(np.float32)
    weight = np.ones(len(future_max), np.float32)
    if variant == "log_weighted":
        weight[future_max >= args.high_from] = args.high_weight
    val_label = future_max[val_ends] >= common.THRESHOLD

    best = {"ap": -np.inf, "epoch": 0, "state": None, "val_pred": None}
    stale = 0
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        order = rng.permutation(train_ends)
        total_loss = 0.0
        for i in range(0, len(order), args.batch_size):
            batch = order[i : i + args.batch_size]
            xb = torch.from_numpy(gather(X, batch, args.window))
            wb = torch.from_numpy(weight[batch])
            optimizer.zero_grad()
            loss = (wb * (model(xb) - torch.from_numpy(y[batch])) ** 2).sum() / wb.sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch)

        val_pred = predict(model, X, val_ends, args.window, args.batch_size * 8, variant)
        ap = average_precision(val_label, val_pred)
        mae = float(np.abs(val_pred - future_max[val_ends]).mean())
        print(f"[{variant}] epoch {epoch:2d}  train loss {total_loss / len(order):.5f}  "
              f"val MAE {mae:.3f}  val AP {ap:.4f}  ({time.time() - started:.0f}s)", flush=True)
        if ap > best["ap"]:
            best = {"ap": ap, "epoch": epoch, "state": {k: v.clone() for k, v in model.state_dict().items()},
                    "val_pred": val_pred}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"[{variant}] no validation AP improvement for {args.patience} epochs; stopping", flush=True)
                break
    model.load_state_dict(best["state"])
    return model, best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", default=",".join(VARIANTS), help=f"comma-separated subset of {VARIANTS}")
    parser.add_argument("--window", type=int, default=common.MIN_HISTORY, help="input rows per sample (at most 15)")
    parser.add_argument("--hidden1", type=int, default=32)
    parser.add_argument("--hidden2", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5, help="epochs without validation AP improvement before stopping")
    parser.add_argument("--high-from", type=float, default=common.HIGH_LEVEL,
                        help="future_max from which log_weighted up-weights samples")
    parser.add_argument("--high-weight", type=float, default=20.0)
    parser.add_argument("--low-keep", type=float, default=1.0,
                        help="share of training samples with future_max < --high-from kept each run")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    variants = args.variants.split(",")
    if not set(variants) <= set(VARIANTS):
        parser.error(f"--variants must be a subset of {VARIANTS}")
    if not 1 <= args.window <= common.MIN_HISTORY:
        parser.error(f"--window must be between 1 and {common.MIN_HISTORY}")

    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    started = time.time()

    df, splits = common.prepare()
    features = build_features(df)
    train_rows = (df["timestamp"] < np.datetime64(common.TRAIN_END)).to_numpy()
    mean = features[train_rows].mean()
    std = features[train_rows].std().replace(0, 1)
    X = ((features - mean) / std).fillna(0).to_numpy(np.float32)
    future_max = df["future_max"].fillna(0).to_numpy(np.float64)
    del features

    train_ends = splits["train"]
    if args.low_keep < 1:
        low = future_max[train_ends] < args.high_from
        train_ends = train_ends[~low | (rng.random(len(train_ends)) < args.low_keep)]
    val_ends = splits["val"]
    print(f"samples: train {len(train_ends)}, validation {len(val_ends)}, test {len(splits['test'])}; "
          f"positive train {(future_max[train_ends] >= common.THRESHOLD).sum()}, "
          f"val {(future_max[val_ends] >= common.THRESHOLD).sum()}", flush=True)

    runs = {}
    for variant in variants:
        variant_started = time.time()
        model, best = train_variant(variant, X, future_max, train_ends, val_ends, args, rng)
        runs[variant] = {"model": model, **best, "train_seconds": time.time() - variant_started}
        print(f"[{variant}] best val AP {best['ap']:.4f} at epoch {best['epoch']}", flush=True)

    chosen = max(runs, key=lambda v: runs[v]["ap"])
    run = runs[chosen]
    print(f"chosen variant: {chosen}", flush=True)
    test_pred = predict(run["model"], X, splits["test"], args.window, args.batch_size * 8, chosen)
    extra = vars(args) | {
        "chosen_variant": chosen,
        "best_epoch": run["epoch"],
        "val_average_precision_by_variant": {v: r["ap"] for v, r in runs.items()},
        "best_epoch_by_variant": {v: r["epoch"] for v, r in runs.items()},
        "train_seconds_by_variant": {v: round(r["train_seconds"], 1) for v, r in runs.items()},
        "total_seconds": round(time.time() - started, 1),
        "features": list(mean.index),
        "architecture": f"LSTM({args.hidden1}) -> LSTM({args.hidden2}) -> last hidden -> Linear(1), MSE",
    }
    common.evaluate("lstm", df, splits, run["val_pred"], test_pred, extra=extra)
    torch.save({"model_state": run["model"].state_dict(), "config": extra,
                "feature_mean": mean.to_dict(), "feature_std": std.to_dict()},
               common.RESULTS_DIR / "lstm" / "model.pt")


if __name__ == "__main__":
    main()

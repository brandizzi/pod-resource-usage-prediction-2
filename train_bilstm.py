#!/usr/bin/env python
"""Bi-LSTM forecast of the maximum busy Tomcat threads over the next 10 minutes.

Follows Dang-Quang & Yoo (2021), "Deep Learning-Based Autoscaling Using
Bidirectional LSTM for Kubernetes": a window of the last 10 rows, one
bidirectional LSTM layer with 30 units per direction whose final states are
concatenated and fed to a dense output, MSE loss, inputs scaled to [0, 1]
with train statistics, up to 50 epochs with early stopping (patience 5) on
the validation loss.

Several variants are trained and the one with the best validation average
precision of saturation is evaluated on test with ``common.evaluate``:

- paper: univariate busy threads, target future_max / 200.
- multi: multivariate inputs (busy threads, requests, CPU, heap, nonheap,
  time of day/week, log1p(step_in_segment)) and a log1p target.
- multi_weighted: as multi, with samples whose target is at least
  ``--high-threshold`` weighted ``--high-weight`` times in the loss.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import common

VARIANTS = {
    "paper": {"multivariate": False, "log_target": False, "weighted": False},
    "multi": {"multivariate": True, "log_target": True, "weighted": False},
    "multi_weighted": {"multivariate": True, "log_target": True, "weighted": True},
}


def build_features(df, multivariate):
    """Per-row model inputs. Heavy-tailed counts are log-scaled in the multivariate variant."""
    if not multivariate:
        return pd.DataFrame({"threads_busy": df["threads"]}).groupby(df["segment"]).ffill()
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
            "log_step": np.log1p(df["step_in_segment"]),
        }
    )
    return features.groupby(df["segment"]).ffill()


def gather(X, ends, window):
    return X[ends[:, None] + np.arange(1 - window, 1)]


class BiLSTM(nn.Module):
    def __init__(self, n_features, hidden):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.head(torch.cat([hidden[-2], hidden[-1]], dim=1)).squeeze(1)


def predict(model, X, ends, window, batch_size):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ends), batch_size):
            out.append(model(torch.from_numpy(gather(X, ends[i : i + batch_size], window))).numpy())
    return np.concatenate(out)


def train_variant(name, cfg, df, splits, args, rng):
    started = time.time()
    torch.manual_seed(args.seed)
    features = build_features(df, cfg["multivariate"])
    train_rows = (df["timestamp"] < np.datetime64(common.TRAIN_END)).to_numpy()
    lo, hi = features[train_rows].min(), features[train_rows].max()
    X = ((features - lo) / (hi - lo).replace(0, 1)).fillna(0).to_numpy(np.float32)

    future_max = df["future_max"].to_numpy()
    if cfg["log_target"]:
        scale = np.log1p(common.SATURATION)
        y = (np.log1p(np.nan_to_num(future_max)) / scale).astype(np.float32)
        to_threads = lambda p: np.expm1(np.clip(p, 0, None) * scale)
    else:
        y = (np.nan_to_num(future_max) / common.SATURATION).astype(np.float32)
        to_threads = lambda p: p * common.SATURATION
    weight = np.where(cfg["weighted"] & (future_max >= args.high_threshold), args.high_weight, 1.0).astype(np.float32)

    train, val = splits["train"], splits["val"]
    val_label = future_max[val] >= common.SATURATION
    model = BiLSTM(X.shape[1], args.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    checkpoint = args.output_dir / f"model_{name}.pt"
    best_loss, best_ap, best_epoch, stale = np.inf, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        model.train()
        order = rng.permutation(train)
        total = 0.0
        for i in range(0, len(order), args.batch_size):
            batch = order[i : i + args.batch_size]
            xb = torch.from_numpy(gather(X, batch, args.window))
            yb, wb = torch.from_numpy(y[batch]), torch.from_numpy(weight[batch])
            optimizer.zero_grad()
            loss = (wb * (model(xb) - yb) ** 2).sum() / wb.sum()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        val_out = predict(model, X, val, args.window, args.batch_size * 8)
        val_loss = float(((val_out - y[val]) ** 2).mean())
        val_ap = common._average_precision(val_label, to_threads(val_out))
        print(f"[{name}] epoch {epoch:2d}  train loss {total / len(order):.6f}  val loss {val_loss:.6f}  "
              f"val AP {val_ap:.4f}  ({time.time() - epoch_started:.0f}s)", flush=True)
        if val_loss < best_loss:
            best_loss, best_ap, best_epoch, stale = val_loss, val_ap, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"[{name}] no validation improvement for {args.patience} epochs; stopping", flush=True)
                break
    model.load_state_dict(torch.load(checkpoint))
    print(f"[{name}] best epoch {best_epoch}, val loss {best_loss:.6f}, val AP {best_ap:.4f}", flush=True)
    return {
        "model": model, "X": X, "to_threads": to_threads, "val_ap": best_ap, "val_loss": best_loss,
        "best_epoch": best_epoch, "epochs_run": epoch, "train_seconds": time.time() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=common.RESULTS_DIR / "bilstm")
    parser.add_argument("--variants", default=",".join(VARIANTS), help=f"comma-separated subset of {list(VARIANTS)}")
    parser.add_argument("--window", type=int, default=10, help=f"input rows per sample (at most {common.MIN_HISTORY})")
    parser.add_argument("--hidden", type=int, default=30, help="LSTM units per direction")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--high-threshold", type=float, default=100, help="target from which samples are up-weighted")
    parser.add_argument("--high-weight", type=float, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.window > common.MIN_HISTORY:
        parser.error(f"--window must be at most {common.MIN_HISTORY}")
    variants = args.variants.split(",")
    if unknown := set(variants) - set(VARIANTS):
        parser.error(f"unknown variants {sorted(unknown)}")

    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    df, splits = common.prepare()
    print({k: len(v) for k, v in splits.items()}, flush=True)

    results = {name: train_variant(name, VARIANTS[name], df, splits, args, rng) for name in variants}
    best = max(results, key=lambda n: results[n]["val_ap"] or 0)
    r = results[best]
    print(f"chosen variant: {best} (val AP {r['val_ap']:.4f})", flush=True)
    val_pred = r["to_threads"](predict(r["model"], r["X"], splits["val"], args.window, args.batch_size * 8))
    test_pred = r["to_threads"](predict(r["model"], r["X"], splits["test"], args.window, args.batch_size * 8))
    extra = {
        "variant": best, **VARIANTS[best],
        "window": args.window, "hidden_per_direction": args.hidden, "layers": 1, "loss": "MSE",
        "input_scaling": "min-max [0,1] with train stats", "optimizer": "Adam", "lr": args.lr,
        "batch_size": args.batch_size, "max_epochs": args.epochs, "patience": args.patience,
        "early_stopping": "validation MSE", "high_threshold": args.high_threshold, "high_weight": args.high_weight,
        "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"], "train_seconds": round(r["train_seconds"], 1),
        "total_seconds": round(time.time() - started, 1),
        "variants": {n: {"val_average_precision": v["val_ap"], "val_loss": v["val_loss"], "best_epoch": v["best_epoch"],
                         "epochs_run": v["epochs_run"], "train_seconds": round(v["train_seconds"], 1)}
                     for n, v in results.items()},
    }
    common.evaluate("bilstm", df, splits, val_pred, test_pred, extra=extra, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

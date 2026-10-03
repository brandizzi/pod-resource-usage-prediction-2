#!/usr/bin/env python
"""Train a BiGRU that forecasts Tomcat busy threads a few minutes ahead.

Reads the wide CSV produced by build_metrics_csv.py. Each sample is a window
of the last ``--window`` rows of one pod, and the target is the pod's busy
thread count ``--horizon-minutes`` after the window's last row. Windows never
cross a gap in the pod's data. The data is split by time into train,
validation and test sets, and the model is compared on the test set against
naive baselines.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

TARGET = "catalina_threadpool_currentthreadsbusy"
STEP_SECONDS = 120
# Consecutive rows further apart than this start a new segment.
MAX_GAP_SECONDS = 180


def load_data(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values(["pod", "timestamp"], ignore_index=True)
    gap = df.groupby("pod")["timestamp"].diff().dt.total_seconds()
    df["segment"] = (gap.isna() | (gap > MAX_GAP_SECONDS)).cumsum()
    return df


def build_features(df):
    """Per-row model inputs. Heavy-tailed counts are log-scaled."""
    ts = df["timestamp"]
    minute_of_day = ts.dt.hour * 60 + ts.dt.minute
    day_angle = 2 * np.pi * minute_of_day / 1440
    week_angle = 2 * np.pi * (ts.dt.dayofweek + minute_of_day / 1440) / 7
    features = pd.DataFrame(
        {
            # A few readings are -2, which is not a valid thread count.
            "threads_busy": np.log1p(df[TARGET].clip(lower=0)),
            "requests": np.log1p(df["catalina_globalrequestprocessor_requestcount_diff"]),
            "cpu_seconds": np.log1p(df["process_cpu_seconds_total_diff"]),
            "heap_gib": df["jvm_memory_used_bytes_heap"] / 2**30,
            "nonheap_gib": df["jvm_memory_used_bytes_nonheap"] / 2**30,
            "day_sin": np.sin(day_angle),
            "day_cos": np.cos(day_angle),
            "week_sin": np.sin(week_angle),
            "week_cos": np.cos(week_angle),
        }
    )
    return features.groupby(df["segment"]).ffill()


def window_ends(df, window, horizon):
    """Row indices that can end a window with a known target `horizon` rows later."""
    segment = df["segment"].to_numpy()
    target = df[TARGET].to_numpy()
    ends = np.arange(window - 1, len(df) - horizon)
    ok = (segment[ends - window + 1] == segment[ends]) & (segment[ends + horizon] == segment[ends])
    ends = ends[ok]
    return ends[~np.isnan(target[ends + horizon])]


def split_ends(df, ends, window, horizon, train_end, val_end):
    """Assign windows to splits so that no window spans a split boundary."""
    start_time = df["timestamp"].to_numpy()[ends - window + 1]
    target_time = df["timestamp"].to_numpy()[ends + horizon]
    train = ends[target_time < train_end]
    val = ends[(start_time >= train_end) & (target_time < val_end)]
    test = ends[start_time >= val_end]
    return train, val, test


def gather(X, ends, window):
    return X[ends[:, None] + np.arange(1 - window, 1)]


class BiGRU(nn.Module):
    def __init__(self, n_features, hidden, layers, dropout):
        super().__init__()
        self.gru = nn.GRU(
            n_features,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        _, hidden = self.gru(x)
        last_layer = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.head(last_layer).squeeze(1)


def predict(model, X, ends, window, batch_size):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ends), batch_size):
            xb = torch.from_numpy(gather(X, ends[i : i + batch_size], window))
            out.append(model(xb).numpy())
    return np.expm1(np.concatenate(out)).clip(min=0)


def metrics(actual, predicted):
    err = predicted - actual
    busy = actual >= 10
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "r2": float(1 - (err**2).sum() / ((actual - actual.mean()) ** 2).sum()),
        "mae_when_actual_ge_10": float(np.abs(err[busy]).mean()) if busy.any() else None,
    }


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=root / "pod_metrics.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "bigru")
    parser.add_argument("--horizon-minutes", type=int, default=6)
    parser.add_argument("--window", type=int, default=30, help="input rows per sample (30 rows = 1 hour)")
    parser.add_argument("--train-end", default="2026-01-01")
    parser.add_argument("--val-end", default="2026-02-01")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3, help="epochs without validation improvement before stopping")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.horizon_minutes * 60 % STEP_SECONDS:
        parser.error(f"--horizon-minutes must be a multiple of {STEP_SECONDS // 60}")
    horizon = args.horizon_minutes * 60 // STEP_SECONDS
    train_end = np.datetime64(pd.Timestamp(args.train_end, tz="UTC").tz_localize(None))
    val_end = np.datetime64(pd.Timestamp(args.val_end, tz="UTC").tz_localize(None))

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(os.cpu_count())

    df = load_data(args.input)
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    features = build_features(df)
    ends = window_ends(df, args.window, horizon)
    train_ends, val_ends, test_ends = split_ends(df, ends, args.window, horizon, train_end, val_end)
    print(f"samples: train {len(train_ends)}, validation {len(val_ends)}, test {len(test_ends)}")

    train_rows = (df["timestamp"] < train_end).to_numpy()
    mean = features[train_rows].mean()
    std = features[train_rows].std().replace(0, 1)
    X = ((features - mean) / std).fillna(0).to_numpy(np.float32)
    actual = df[TARGET].clip(lower=0).to_numpy(np.float32)
    y = np.log1p(actual)

    model = BiGRU(X.shape[1], args.hidden, args.layers, args.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.HuberLoss()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "model.pt"
    best_val_mae, stale_epochs = np.inf, 0
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        order = rng.permutation(train_ends)
        total_loss = 0.0
        for i in range(0, len(order), args.batch_size):
            batch = order[i : i + args.batch_size]
            xb = torch.from_numpy(gather(X, batch, args.window))
            yb = torch.from_numpy(y[batch + horizon])
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch)

        val_pred = predict(model, X, val_ends, args.window, args.batch_size * 4)
        val_mae = float(np.abs(val_pred - actual[val_ends + horizon]).mean())
        print(
            f"epoch {epoch:2d}  train loss {total_loss / len(order):.4f}  "
            f"validation MAE {val_mae:.4f}  ({time.time() - started:.0f}s)"
        )
        if val_mae < best_val_mae:
            best_val_mae, stale_epochs = val_mae, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": vars(args) | {"input": str(args.input), "output_dir": str(args.output_dir)},
                    "feature_names": list(features.columns),
                    "feature_mean": mean.to_dict(),
                    "feature_std": std.to_dict(),
                },
                checkpoint,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"no validation improvement for {args.patience} epochs; stopping")
                break

    model.load_state_dict(torch.load(checkpoint)["model_state"])
    test_actual = actual[test_ends + horizon]
    observed = np.expm1(features["threads_busy"].fillna(0).to_numpy(np.float32))
    predictions = {
        "BiGRU": predict(model, X, test_ends, args.window, args.batch_size * 4),
        "Persistence (last value)": observed[test_ends],
        "Moving average (last 30 min)": gather(observed, test_ends, 15).mean(axis=1),
    }
    results = {name: metrics(test_actual, pred) for name, pred in predictions.items()}

    print(f"\nTest set ({len(test_ends)} samples, from {args.val_end}), forecasting {args.horizon_minutes} min ahead:")
    print(f"{'model':30s} {'MAE':>7s} {'RMSE':>7s} {'R²':>7s} {'MAE when ≥10':>13s}")
    for name, m in results.items():
        print(f"{name:30s} {m['mae']:7.3f} {m['rmse']:7.3f} {m['r2']:7.3f} {m['mae_when_actual_ge_10']:13.3f}")

    (args.output_dir / "metrics.json").write_text(
        json.dumps({"best_validation_mae": best_val_mae, "test": results}, indent=2)
    )
    pd.DataFrame(
        {
            "timestamp": df["timestamp"].to_numpy()[test_ends + horizon],
            "pod": df["pod"].to_numpy()[test_ends],
            "actual": test_actual,
            "predicted": predictions["BiGRU"],
            "persistence": predictions["Persistence (last value)"],
        }
    ).to_csv(args.output_dir / "test_predictions.csv", index=False)
    print(f"\nsaved model, metrics and test predictions to {args.output_dir}")


if __name__ == "__main__":
    main()

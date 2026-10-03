#!/usr/bin/env python
"""Naive reference forecasts that the models have to beat.

- reactive: the current busy threads, i.e. what a threshold autoscaler sees.
- recent_max: the maximum busy threads over the last 30 minutes.
"""

import numpy as np

import common


def main():
    df, splits = common.prepare()
    threads = df["threads"].to_numpy()
    for split, rows in splits.items():
        print(f"{split}: {len(rows)} samples, {int((df['future_max'].to_numpy()[rows] >= common.SATURATION).sum())} saturated")

    def recent_max(rows):
        window = rows[:, None] + np.arange(1 - common.MIN_HISTORY, 1)
        return np.nanmax(threads[window], axis=1)

    common.evaluate("baseline_reactive", df, splits, threads[splits["val"]], threads[splits["test"]])
    common.evaluate("baseline_recent_max", df, splits, recent_max(splits["val"]), recent_max(splits["test"]))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Naive reference forecasts that the models have to beat.

- reactive: the current value of the target, i.e. what a threshold autoscaler sees.
- recent_max: the maximum of the target over the last 30 minutes.
"""

import numpy as np

import common


def main():
    df, splits = common.prepare()
    target = df["target"].to_numpy()
    for split, rows in splits.items():
        print(f"{split}: {len(rows)} samples, {int((df['future_max'].to_numpy()[rows] >= common.THRESHOLD).sum())} positive")

    def recent_max(rows):
        window = rows[:, None] + np.arange(1 - common.MIN_HISTORY, 1)
        return np.nanmax(target[window], axis=1)

    common.evaluate("baseline_reactive", df, splits, target[splits["val"]], target[splits["test"]])
    common.evaluate("baseline_recent_max", df, splits, recent_max(splits["val"]), recent_max(splits["test"]))


if __name__ == "__main__":
    main()

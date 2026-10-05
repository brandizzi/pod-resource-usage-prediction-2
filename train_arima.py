#!/usr/bin/env python
"""Univariate ARIMA forecast of the maximum busy Tomcat threads over the forecast horizon.

One global set of ARIMA(p,d,q) parameters is fitted by maximum likelihood
(statsmodels SARIMAX, Kalman filter) on a random sample of long training-period
segments, concatenated with runs of missing values between them so that they
barely influence each other. The order is chosen by AIC over a small grid,
separately on the raw and the log1p scale of busy threads.

The AIC-best order of each scale and ARIMA(4,1,0) (Dang-Quang & Yoo 2021) are
then compared on validation. For a sample row t, each segment is Kalman
filtered once with the fixed parameters; the filtered state at t (which only
uses rows <= t of that segment) gives the 1..H step forecast means and
variances, and the score is max_h(mean_h + z * sd_h), transformed back to
threads. The candidate model and z are chosen by validation average precision
of saturation, and the test set is predicted only at the end.
"""

import argparse
import os
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

import common

SCALES = {
    "raw": (lambda x: x, lambda x: x),
    "log1p": (np.log1p, np.expm1),
}
PAPER_ORDER = (4, 1, 0)


def fit_sample(df, min_length, rows, gap, rng):
    """Busy threads of random long training-period segments, separated by `gap` NaNs."""
    train = df[df["timestamp"] < np.datetime64(common.TRAIN_END)]
    groups = {s: g["threads"].to_numpy() for s, g in train.groupby("segment") if len(g) >= min_length}
    parts, total = [], 0
    for s in rng.permutation(list(groups)):
        parts += [groups[s], np.full(gap, np.nan)]
        total += len(groups[s])
        if total >= rows:
            break
    return np.concatenate(parts[:-1]), len(parts) // 2, total


def make_model(endog, order):
    return SARIMAX(endog, order=order, trend="c" if order[1] == 0 else "n")


def fit_order(y, order):
    """AIC and named parameters, without keeping the memory-heavy results object."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        res = make_model(y, order).fit(disp=False, maxiter=200, low_memory=True)
    return {"aic": float(res.aic), "params": dict(zip(res.model.param_names, map(float, res.params)))}


def forecast_paths(df, rows, scale, order, params, horizon):
    """Mean and sd (model scale) of the 1..horizon step forecasts made at each row."""
    forward, _ = SCALES[scale]
    y = forward(df["threads"].to_numpy(np.float64))
    segment = df["segment"].to_numpy()
    mean = np.empty((len(rows), horizon))
    sd = np.empty((len(rows), horizon))
    seg_of_rows = segment[rows]
    bounds = np.r_[0, np.flatnonzero(np.diff(seg_of_rows)) + 1, len(rows)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        start = np.searchsorted(segment, seg_of_rows[a])
        res = make_model(y[start : rows[b - 1] + 1], order).filter(params)
        ssm = res.model.ssm
        Z, T = ssm["design"][:, :, 0] if ssm["design"].ndim == 3 else ssm["design"], ssm["transition"]
        T = T[:, :, 0] if T.ndim == 3 else T
        R = ssm["selection"][:, :, 0] if ssm["selection"].ndim == 3 else ssm["selection"]
        Q = ssm["state_cov"][:, :, 0] if ssm["state_cov"].ndim == 3 else ssm["state_cov"]
        c = ssm["state_intercept"]
        c = c[:, -1] if c.ndim == 2 else c
        d = np.ravel(ssm["obs_intercept"])[-1]
        noise = np.ravel(ssm["obs_cov"])[-1]
        idx = rows[a:b] - start
        state = res.filter_results.filtered_state[:, idx].T
        cov = np.moveaxis(res.filter_results.filtered_state_cov[:, :, idx], 2, 0)
        RQR = R @ Q @ R.T
        for h in range(horizon):
            state = state @ T.T + c
            cov = T @ cov @ T.T + RQR
            mean[a:b, h] = state @ Z[0] + d
            sd[a:b, h] = np.sqrt(np.maximum(np.einsum("i,nij,j->n", Z[0], cov, Z[0]) + noise, 0))
    return mean, sd


def score(mean, sd, scale, z):
    return SCALES[scale][1]((mean + z * sd).max(axis=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="arima", help="results go to results/<name>")
    parser.add_argument("--fit-rows", type=int, default=150_000, help="training rows used to fit the parameters")
    parser.add_argument("--min-segment", type=int, default=200, help="minimum length of a fitting segment")
    parser.add_argument("--gap", type=int, default=100, help="missing values between concatenated segments")
    parser.add_argument("--max-p", type=int, default=4)
    parser.add_argument("--max-d", type=int, default=1)
    parser.add_argument("--max-q", type=int, default=2)
    parser.add_argument("--z-grid", default="0,0.5,1,1.5,2,2.5,3", help="upper-bound multipliers tried on validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    z_grid = [float(z) for z in args.z_grid.split(",")]
    started = time.time()

    df, splits = common.prepare()
    actual = df["future_max"].to_numpy()
    val_rows, test_rows = splits["val"], splits["test"]
    val_label = actual[val_rows] >= common.SATURATION
    print({k: len(v) for k, v in splits.items()}, flush=True)

    raw, n_segments, n_obs = fit_sample(df, args.min_segment, args.fit_rows, args.gap, np.random.default_rng(args.seed))
    print(f"fitting sample: {n_segments} segments, {n_obs} rows", flush=True)

    fits = {}
    for scale, (forward, _) in SCALES.items():
        y = forward(raw)
        for p in range(args.max_p + 1):
            for d in range(args.max_d + 1):
                for q in range(args.max_q + 1):
                    t0 = time.time()
                    try:
                        res = fit_order(y, (p, d, q))
                    except Exception as e:
                        print(f"{scale} ARIMA{(p, d, q)} failed: {e}", flush=True)
                        continue
                    fits[scale, (p, d, q)] = res
                    print(f"{scale:6s} ARIMA{(p, d, q)}  AIC {res['aic']:14.1f}  ({time.time() - t0:.1f}s)", flush=True)

    candidates = []
    for scale in SCALES:
        best = min((o for s, o in fits if s == scale), key=lambda o: fits[scale, o]["aic"])
        print(f"{scale}: AIC-best order {best}", flush=True)
        candidates += [(scale, best), (scale, PAPER_ORDER)]
    candidates = list(dict.fromkeys(c for c in candidates if c in fits))

    selection, paths = [], {}
    for scale, order in candidates:
        params = np.array(list(fits[scale, order]["params"].values()))
        t0 = time.time()
        mean, sd = forecast_paths(df, val_rows, scale, order, params, common.HORIZON_STEPS)
        paths[scale, order] = mean, sd
        for z in z_grid:
            pred = np.clip(score(mean, sd, scale, z), 0, common.SATURATION)
            ap = common._average_precision(val_label, pred)
            mae = float(np.abs(pred - actual[val_rows]).mean())
            selection.append({"scale": scale, "order": list(order), "z": z, "val_average_precision": ap, "val_mae": mae})
            print(f"val {scale:6s} ARIMA{order} z={z:<4g} AP {ap:.4f}  MAE {mae:.3f}", flush=True)
        print(f"  ({time.time() - t0:.1f}s)", flush=True)

    best = max(selection, key=lambda r: r["val_average_precision"])
    scale, order, z = best["scale"], tuple(best["order"]), best["z"]
    fit = fits[scale, order]
    print(f"chosen: {scale} ARIMA{order}, z={z}, params {fit['params']}", flush=True)

    params = np.array(list(fit["params"].values()))
    val_pred = score(*paths[scale, order], scale, z)
    test_pred = score(*forecast_paths(df, test_rows, scale, order, params, common.HORIZON_STEPS), scale, z)
    extra = {
        "scale": scale,
        "order": list(order),
        "trend": "c" if order[1] == 0 else "n",
        "z": z,
        "params": fit["params"],
        "aic": fit["aic"],
        "fit_sample": {"segments": n_segments, "rows": n_obs, "min_segment": args.min_segment, "gap": args.gap},
        "aic_grid": {f"{s} {o}": r["aic"] for (s, o), r in fits.items()},
        "validation_selection": selection,
        "run_seconds": round(time.time() - started, 1),
    }
    common.evaluate(args.name, df, splits, val_pred, test_pred, extra=extra)
    print(f"total run time {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()

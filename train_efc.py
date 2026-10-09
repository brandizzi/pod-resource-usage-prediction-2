#!/usr/bin/env python
"""Energy-based Flow Classifier (EFC) for saturation of the Tomcat thread pool.

Follows Souza et al., "A Novel Open Set Energy-based Flow Classifier for
Network Intrusion Detection" (arXiv 2109.11224), and the single-class EFC it
extends. Each sample row t of a pod is described by the lag and rolling
features of train_qrf.py, discretized into at most ``--bins`` quantile bins
per feature (fitted on training rows). For each class, a Potts model is
inferred from single and pairwise bin frequencies with pseudocount weight
``--pseudocount``: couplings e_ij = -C^-1 (C the covariance of one-hot
encoded bins, last bin of each feature as reference) and local fields h_i.
The energy of a sample under a class model is
H = -sum_{i<j} e_ij(a_i, a_j) - sum_i h_i(a_i); lower means more typical.

The classes are "the pool saturates within the horizon" (future_max >= 200)
and "it does not". Variants, chosen by validation average precision:

- single_class: the original EFC, a model of non-saturating rows only;
  the alarm score is its energy (how anomalous the row is).
- two_class_capped: the multi-class EFC as in the paper, each class
  undersampled to ``--max-per-class`` training rows; the score is
  H_normal - H_saturating, a log-likelihood ratio up to a constant.
- two_class_full: as two_class_capped, with all non-saturating rows.

The score is mapped monotonically to 0..200 for common.evaluate, so MAE and
RMSE are not meaningful for this model. The paper's own decision rule (lowest
energy class, "suspicious" above that class's 95th percentile training
energy) is also reported in the metrics' ``extra``.
"""

import argparse
import time

import numpy as np
from scipy.special import expit

import common
from train_qrf import base_series, build_features

VARIANTS = ["single_class", "two_class_capped", "two_class_full"]


def fit_bins(X, bins):
    """Inner quantile edges per feature, from training rows."""
    qs = np.linspace(0, 1, bins + 1)[1:-1]
    return [np.unique(np.quantile(col, qs)) for col in X.T]


def discretize(X, edges):
    """Bin index per feature; a feature with k edges has k + 1 bins."""
    return np.stack([np.searchsorted(e, col, side="right") for e, col in zip(edges, X.T)], axis=1)


class PottsModel:
    """Couplings and local fields of one class, on one-hot bins excluding each feature's last bin."""

    def __init__(self, A, n_bins, pseudocount, chunk=20_000):
        n, m = A.shape
        offsets = np.r_[0, np.cumsum(n_bins - 1)]
        self.n_bins, self.offsets = n_bins, offsets
        dim = offsets[-1]
        freq, pair = np.zeros(dim), np.zeros((dim, dim))
        for s in range(0, n, chunk):
            o = self._onehot(A[s : s + chunk])
            freq += o.sum(0)
            pair += o.T @ o
        freq, pair = freq / n, pair / n
        q = np.repeat(n_bins, n_bins - 1).astype(np.float64)
        f_i = (1 - pseudocount) * freq + pseudocount / q
        f_ij = (1 - pseudocount) * pair + pseudocount / np.outer(q, q)
        feature = np.repeat(np.arange(m), n_bins - 1)
        same = feature[:, None] == feature[None, :]
        f_ij[same] = 0
        f_ij[np.diag_indices(dim)] = f_i
        cov = f_ij - np.outer(f_i, f_i)
        couplings = -np.linalg.inv(cov)
        couplings[same] = 0
        # Reference (last) bin frequency of each feature after pseudocounts.
        ref = np.array([(1 - pseudocount) * (A[:, i] == n_bins[i] - 1).mean() + pseudocount / n_bins[i]
                        for i in range(m)])
        self.fields = (np.log(f_i / ref[feature]) - couplings @ f_i).astype(np.float32)
        self.couplings = couplings.astype(np.float32)

    def _onehot(self, A):
        n, m = A.shape
        out = np.zeros((n, self.offsets[-1]), np.float32)
        for i in range(m):
            keep = A[:, i] < self.n_bins[i] - 1
            out[np.flatnonzero(keep), self.offsets[i] + A[keep, i]] = 1
        return out

    def energy(self, A, chunk=20_000):
        out = []
        for s in range(0, len(A), chunk):
            o = self._onehot(A[s : s + chunk])
            out.append(-0.5 * np.einsum("nd,nd->n", o @ self.couplings, o) - o @ self.fields)
        return np.concatenate(out).astype(np.float64)


def to_prediction(score, reference):
    """Monotone map of scores into 0..200, scaled by the training score spread."""
    center, spread = np.median(reference), reference.std() or 1.0
    return common.SATURATION * expit((score - center) / spread)


def paper_rule(energies, cutoffs, label):
    """Shares of the paper's decisions (lowest-energy class, or suspicious above its cutoff) per true class."""
    best = np.argmin(energies, axis=1)
    lowest = energies[np.arange(len(best)), best]
    decision = np.where(lowest <= np.asarray(cutoffs)[best], best, 2)
    names = ["normal", "saturating", "suspicious"]
    return {
        true: {names[d]: float((decision[label == (true == "saturating")] == d).mean()) for d in range(3)}
        for true in ("normal", "saturating")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    parser.add_argument("--bins", type=int, default=30, help="maximum discretization bins per feature")
    parser.add_argument("--pseudocount", type=float, default=0.5)
    parser.add_argument("--max-per-class", type=int, default=5000, help="training rows per class in two_class_capped")
    parser.add_argument("--cutoff-percentile", type=float, default=95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    started = time.time()
    df, splits = common.prepare()
    series = base_series(df)
    X, label = {}, {}
    for split, rows in splits.items():
        X[split], names = build_features(df, series, rows)
        label[split] = df["future_max"].to_numpy()[rows] >= common.SATURATION
    edges = fit_bins(X["train"], args.bins)
    A = {split: discretize(x, edges) for split, x in X.items()}
    n_bins = np.array([len(e) + 1 for e in edges])
    used = n_bins > 1
    A = {split: a[:, used] for split, a in A.items()}
    n_bins = n_bins[used]
    print(f"{used.sum()} of {len(names)} features with more than one bin, {(n_bins - 1).sum()} one-hot columns; "
          f"saturated train {label['train'].sum()}, val {label['val'].sum()}, test {label['test'].sum()} "
          f"({time.time() - started:.0f}s)", flush=True)

    normal_rows = np.flatnonzero(~label["train"])
    saturated_rows = np.flatnonzero(label["train"])
    capped = lambda r: np.sort(rng.choice(r, min(len(r), args.max_per_class), replace=False))
    # Training rows on which the score spread and the paper's cutoffs are measured.
    reference_rows = np.sort(rng.choice(len(label["train"]), min(len(label["train"]), 50_000), replace=False))

    def energies(models, a):
        return np.stack([m.energy(a) for m in models.values()], axis=1)

    def score(e):
        return e[:, 0] - e[:, 1] if e.shape[1] > 1 else e[:, 0]

    results = {}
    for variant in args.variants:
        t0 = time.time()
        if variant == "single_class":
            classes = {"normal": normal_rows}
        elif variant == "two_class_capped":
            classes = {"normal": capped(normal_rows), "saturating": capped(saturated_rows)}
        else:
            classes = {"normal": normal_rows, "saturating": saturated_rows}
        models = {c: PottsModel(A["train"][r], n_bins, args.pseudocount) for c, r in classes.items()}
        val_score = score(energies(models, A["val"]))
        ap = common._average_precision(label["val"], val_score)
        results[variant] = {"models": models, "classes": classes, "val_ap": ap, "val_score": val_score,
                            "seconds": time.time() - t0}
        print(f"{variant}: training rows { {c: len(r) for c, r in classes.items()} }, validation AP {ap:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    best = max(results, key=lambda v: results[v]["val_ap"] or 0)
    r = results[best]
    test_energy = energies(r["models"], A["test"])
    test_score = score(test_energy)
    train_score = score(energies(r["models"], A["train"][reference_rows]))
    print(f"selected {best}", flush=True)

    extra = {
        "variant": best,
        "hyperparams": {"bins": args.bins, "pseudocount": args.pseudocount, "max_per_class": args.max_per_class,
                        "cutoff_percentile": args.cutoff_percentile, "seed": args.seed},
        "features": int(used.sum()),
        "onehot_columns": int((n_bins - 1).sum()),
        "training_rows": {c: len(rows) for c, rows in r["classes"].items()},
        "validation_ap": {v: x["val_ap"] for v, x in results.items()},
        "fit_seconds": {v: round(x["seconds"], 1) for v, x in results.items()},
        "total_seconds": round(time.time() - started, 1),
    }
    if best != "single_class":
        cutoffs = []
        for c, model in r["models"].items():
            rows = r["classes"][c]
            sample = rows if len(rows) <= 50_000 else rng.choice(rows, 50_000, replace=False)
            cutoffs.append(float(np.percentile(model.energy(A["train"][sample]), args.cutoff_percentile)))
        extra["paper_rule_cutoffs"] = cutoffs
        extra["paper_rule_test"] = paper_rule(test_energy, cutoffs, label["test"])
        print(f"paper decision rule on test: {extra['paper_rule_test']}", flush=True)

    common.evaluate("efc", df, splits, to_prediction(r["val_score"], train_score),
                    to_prediction(test_score, train_score), extra=extra)
    print(f"total {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Query-level paired statistics that avoid seed-level pseudoreplication.

The 30 stratified seed splits overlap heavily, so per-seed metrics are not
independent and an across-seed t-test overstates significance. Instead we
aggregate each metric to one value per unique query (its mean over the seeds
in which that query appears in the test set), then compare two variants with a
paired test over the shared set of queries:

  * a paired sign-flip randomization test for the two-sided p-value, and
  * a paired bootstrap over queries for the 95% CI of the mean difference.

Usage:
  python query_level_stats.py <perquery_dir> <variantA> <variantB> [metric]
  metric in {top1, rr}; default top1.
"""
import sys
import os

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260724)


def per_query_means(path, metric):
    df = pd.read_csv(path)
    return df.groupby("rid")[metric].mean()


def paired_randomization_p(diff, n_iter=100000):
    obs = float(np.mean(diff))
    signs = RNG.choice([-1.0, 1.0], size=(n_iter, diff.size))
    null = np.mean(signs * np.abs(diff), axis=1)
    return obs, float(np.mean(np.abs(null) >= abs(obs) - 1e-12))


def paired_bootstrap_ci(diff, n_iter=100000, alpha=0.05):
    idx = RNG.integers(0, diff.size, size=(n_iter, diff.size))
    means = np.mean(diff[idx], axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def main():
    d, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
    metric = sys.argv[4] if len(sys.argv) > 4 else "top1"
    sa = per_query_means(os.path.join(d, f"perquery_{a}.csv"), metric)
    sb = per_query_means(os.path.join(d, f"perquery_{b}.csv"), metric)
    shared = sorted(set(sa.index) & set(sb.index))
    va, vb = sa.loc[shared].to_numpy(), sb.loc[shared].to_numpy()
    diff = vb - va
    obs, p = paired_randomization_p(diff)
    lo, hi = paired_bootstrap_ci(diff)
    print(f"{b} vs {a} [{metric}] over {len(shared)} unique queries:")
    print(f"  mean(A)={va.mean():.4f}  mean(B)={vb.mean():.4f}")
    print(f"  paired delta = {obs:+.4f}  95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  sign-flip randomization p = {p:.4f}")


if __name__ == "__main__":
    main()

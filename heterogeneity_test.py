#!/usr/bin/env python3
"""Direct test that the reformulation effect differs between corpus types.

WHY THIS EXISTS
---------------
The manuscript claimed that the reformulation benefit "does not replicate"
on external corpora, supported by a significant effect on the engineered
corpus and non-significant effects on the three external ones.

That is not a test. A significant result in one group and a non-significant
result in another does not establish that the two differ; the comparison of
significance is not the significance of the comparison. The engineered corpus
also has the smallest n, so the difference in p-values is partly a difference
in power.

This script tests the contrast directly: is the mean paired Top-1 difference
on the engineered corpus different from the mean on the pooled external
corpora? Two procedures, both at the requirement level:

  * permutation - shuffle the engineered/external label across requirements
    and recompute the difference of means. This builds a null in which corpus
    type carries no information.
  * bootstrap   - resample within each group and report the interval for the
    difference of means.

SCOPE, stated because it is easy to overread. This compares one engineered
corpus against three external corpora that are themselves correlated
replications from a single institution. A significant contrast establishes
that these corpora differ, not that engineered benchmarks in general inflate
effects. It cannot separate "we authored it" from any other property that
distinguishes our corpus from NIST's.

This analysis is EXPLORATORY. It was written after seeing the per-corpus
effects, in response to review, and is not covered by the hybrid
preregistration.

Usage:
    python heterogeneity_test.py
"""
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

import confirmatory_stats as cs

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "results_v3", "confirmatory")

CORPORA = {
    "nist": ("csf_benchmark", "csf_"),
    "hipaa": ("hipaa_benchmark", ""),
    "pf": ("pf_benchmark", ""),
    "diagnostic": ("diagnostic_benchmark", "diag_"),
}
ENGINEERED = ("diagnostic",)
EXTERNAL = ("nist", "hipaa", "pf")

BASELINE, TREATMENT = "multi", "reform"
N_PERM = 100000
SEED = 20260801


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def git_state():
    def run(*a):
        try:
            return subprocess.run(a, cwd=HERE, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:
            return None
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain"))}


def per_requirement_diffs(key):
    """One paired Top-1 difference per requirement, matching run_confirmatory."""
    d = os.path.join(HERE, "results_v3", key)
    def load(method):
        df = pd.read_csv(os.path.join(d, f"perquery_{method}.csv"))
        return df.groupby("rid")["top1"].mean()
    a, b = load(TREATMENT), load(BASELINE)
    common = a.index.intersection(b.index)
    return (a.loc[common] - b.loc[common]).values


def main():
    print("=" * 72)
    print("EXPLORATORY. Written after seeing the per-corpus effects, in")
    print("response to review. Not covered by any preregistration.")
    print("=" * 72)
    print()

    groups = {k: per_requirement_diffs(k) for k in CORPORA}
    eng = np.concatenate([groups[k] for k in ENGINEERED])
    ext = np.concatenate([groups[k] for k in EXTERNAL])
    labels = np.array(["engineered"] * len(eng) + ["external"] * len(ext))
    alld = np.concatenate([eng, ext])

    obs = float(eng.mean() - ext.mean())
    print(f"engineered  n={len(eng):3d}  mean {eng.mean():+.4f}")
    print(f"external    n={len(ext):3d}  mean {ext.mean():+.4f}")
    print(f"difference           {obs:+.4f}\n")

    # --- permutation: shuffle the group label -------------------------------
    rng = np.random.default_rng(SEED)
    n_eng = len(eng)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        p = rng.permutation(alld)
        null[i] = p[:n_eng].mean() - p[n_eng:].mean()
    p_perm = (1.0 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (N_PERM + 1.0)

    # --- bootstrap: resample within group -----------------------------------
    brng = np.random.default_rng(SEED)
    draws = np.empty(cs.BOOTSTRAP_N)
    for i in range(cs.BOOTSTRAP_N):
        a = eng[brng.integers(0, len(eng), len(eng))]
        b = ext[brng.integers(0, len(ext), len(ext))]
        draws[i] = a.mean() - b.mean()
    lo, hi = np.quantile(draws, [0.025, 0.975])

    print(f"permutation p (two-sided, {N_PERM} shuffles) = {p_perm:.5f}")
    print(f"bootstrap 95% CI for the difference = [{lo:+.4f}, {hi:+.4f}]")
    print()

    # Per-corpus contrast against the pooled external mean, descriptive.
    print("each corpus against the pooled external mean (descriptive):")
    for k in CORPORA:
        print(f"  {k:11s} n={len(groups[k]):3d}  mean {groups[k].mean():+.4f}")

    record = {
        "status": "exploratory",
        "status_reason": "written in response to review, after the per-corpus "
                         "effects were known; not preregistered",
        "question": "does the reformulation effect differ between the "
                    "engineered corpus and the pooled external corpora",
        "scope_limit": "one engineered corpus against three correlated NIST "
                       "replications; establishes that these corpora differ, "
                       "not that engineered benchmarks inflate effects in "
                       "general, and cannot separate authorship from other "
                       "properties of the NIST ecosystem",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "argv": sys.argv,
        "git": git_state(),
        "engineered": {"corpora": list(ENGINEERED), "n": len(eng),
                       "mean": float(eng.mean())},
        "external": {"corpora": list(EXTERNAL), "n": len(ext),
                     "mean": float(ext.mean())},
        "difference": obs,
        "permutation": {"p_value": float(p_perm), "n_permutations": N_PERM,
                        "seed": SEED, "method": "label permutation"},
        "bootstrap": {"ci_low": float(lo), "ci_high": float(hi),
                      "n_boot": cs.BOOTSTRAP_N, "seed": SEED,
                      "coverage": "95% two-sided", "method": "within-group"},
        "per_corpus_means": {k: float(v.mean()) for k, v in groups.items()},
        "spec_hashes": {"heterogeneity_test.py": sha256(os.path.abspath(__file__)),
                        "confirmatory_stats.py": sha256(
                            os.path.join(HERE, "confirmatory_stats.py"))},
        "environment": {"python": sys.version.split()[0],
                        "platform": platform.platform(),
                        "numpy": np.__version__},
    }
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "heterogeneity_engineered_vs_external.json")
    with open(out, "w", newline="\n") as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"\nrecord written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

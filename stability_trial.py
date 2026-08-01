#!/usr/bin/env python3
"""Execution-stability trial: how often does the same input change answer?

The study's remaining claim for a deterministic pipeline is that it returns
the same answer every run. That claim only has force if something else
visibly does not, so this measures it rather than asserting it.

Sampling parameters cannot be set on current frontier models, so there is no
temperature to pin to zero. Repeat-run variation is therefore measured, not
assumed away.

The response cache MUST be bypassed here. A cached reply would return
identical output by construction and manufacture a zero flip rate, which is
the exact failure this trial exists to detect.

Primary statistic: the any-flip rate, the share of requirements whose top-1
is not identical across all N runs, with a Wilson interval. Reported as a
descriptive repeated-run probe, not a powered test.

Usage:
    python stability_trial.py --corpus hipaa_benchmark --runs 10 --sample 50
    python stability_trial.py --analyze hipaa_benchmark
"""
import argparse
import glob
import math
import os
import subprocess
import sys

import pandas as pd


def wilson(k, n, z=1.959963984540054):
    """Wilson score interval for a proportion; sane at k=0, unlike normal."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def run_trial(corpus, runs, sample, effort):
    """Execute N uncached passes, each tagged with its own run id."""
    for run_id in range(runs):
        print(f"\n--- run {run_id + 1}/{runs} (cache bypassed) ---")
        cmd = [sys.executable, "llm_backend.py", "--corpus", corpus,
               "--effort", effort, "--bypass-cache", "--run-id", str(run_id)]
        if sample:
            cmd += ["--limit", str(sample)]
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"run {run_id} failed with code {r.returncode}")
            return r.returncode
    return 0


def analyze(corpus):
    """Flip rates across the completed runs."""
    paths = sorted(glob.glob(os.path.join(corpus, "llm_rerank_run*.csv")))
    if len(paths) < 2:
        print(f"need at least 2 runs in {corpus}, found {len(paths)}")
        return 1

    frames = [pd.read_csv(p) for p in paths]
    n_runs = len(frames)
    merged = pd.concat(frames)
    print(f"{corpus}: {n_runs} runs, {len(merged)} total observations")

    top1 = merged.pivot_table(index="rid", columns="run_id",
                              values="top1_control", aggfunc="first")
    dec = merged.pivot_table(index="rid", columns="run_id",
                             values="abstain", aggfunc="first")

    def any_flip(frame):
        # A row flips if it shows more than one distinct value across runs.
        return frame.apply(lambda r: r.dropna().nunique() > 1, axis=1)

    t_flip = any_flip(top1)
    d_flip = any_flip(dec)
    n = len(t_flip)

    for label, flips in (("top-1 selection", t_flip),
                         ("accept/abstain decision", d_flip)):
        k = int(flips.sum())
        lo, hi = wilson(k, n)
        print(f"  {label:<26} any-flip {k}/{n} = {k / n:.3f}  "
              f"95% Wilson [{lo:.3f}, {hi:.3f}]")

    # Secondary: mean pairwise disagreement across run pairs.
    pairs = [(i, j) for i in range(n_runs) for j in range(i + 1, n_runs)]
    if pairs and set(top1.columns) >= {p for pr in pairs for p in pr}:
        rates = [(top1[i] != top1[j]).mean() for i, j in pairs]
        print(f"  mean pairwise disagreement  {sum(rates) / len(rates):.3f}")

    print("\n  A deterministic backend is EXPECTED to be zero here under the")
    print("  locked environment. That expectation is verified, not assumed.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--sample", type=int, default=50,
                    help="requirements per run; keeps the trial affordable")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--analyze", metavar="CORPUS")
    args = ap.parse_args()

    if args.analyze:
        return analyze(args.analyze)
    if not args.corpus:
        ap.error("give --corpus to run, or --analyze to summarize")
    rc = run_trial(args.corpus, args.runs, args.sample, args.effort)
    return analyze(args.corpus) if rc == 0 else rc


if __name__ == "__main__":
    sys.exit(main())

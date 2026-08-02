#!/usr/bin/env python3
"""One script that regenerates the complete one-pass reformulation record.

Replaces an ad-hoc inline computation that produced intervals, permutation
p-values and TOST results with no committed script behind them. Everything the
manuscript cites about the one-pass reformulation contrast is produced here,
with seeds, iteration counts, confidence levels and git provenance recorded.

WHAT IT REPORTS

  * per corpus: multi and reform Top-1, the paired difference, wins, losses
    and ties, a requirement-level bootstrap interval, and a paired sign-flip
    permutation p-value;
  * TOST equivalence at the pre-specified margin, flagged when its decision
    sits near alpha, which happens when only a handful of requirements are
    discordant. NO exact equivalence test is provided; the exact sign test
    below validates the superiority result only, so boundary-sensitive
    equivalence results are reported as approximate and never headlined;
  * the engineered-versus-external contrast, with a bootstrap interval that is
    explicitly conditional on these four corpora and no test, for the
    cluster-level reasons documented in heterogeneity_test.py;
  * the same quantities under both LSI fitting regimes, so the effect of the
    representation can be separated from the effect of the protocol.

WHY THE LSI COMPARISON IS HERE, AND WHAT IT DOES NOT YET ISOLATE
The previously reported figures come from repeated holdouts with LSI fitted on
controls plus the train and calibration requirements, which EXCLUDES the test
requirements being scored. The one-pass arms here fit either controls only, or
controls plus EVERY requirement including the ones being scored.

So the transductive arm here is not the holdout fitting regime. Its fitting
population is larger and includes the evaluation queries. Comparing it against
the holdout figures varies the fitting population as well as the protocol, and
an earlier version of this file claimed that comparison held the fit fixed. It
did not.

The clean factorial needs three cells:

    A  repeated holdout + controls-only LSI
    B  repeated holdout + train/calibration-assisted LSI   (previous figures)
    C  one-pass        + controls-only LSI                 (primary here)

A versus B isolates the LSI fitting regime. A versus C isolates the evaluation
protocol. Cell A is produced by holdout_lsi_factorial.py. The one-pass
transductive arm is retained only as a distribution-aware sensitivity and must
not be described as equivalent to cell B.

STATUS: exploratory, unplanned. Written in response to review, after the
per-corpus effects were known. Not covered by the hybrid preregistration.

Usage:
    python analyze_onepass.py
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
from scipy import stats as sps

import confirmatory_stats as cs

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "results_v3", "shared")

FITS = {"inductive": "shared_raa_scores.csv",
        "transductive": "shared_raa_scores_transductive.csv"}
NAME = {"csf_benchmark": "nist", "hipaa_benchmark": "hipaa",
        "pf_benchmark": "pf", "diagnostic_benchmark": "diagnostic"}
EXTERNAL = ("nist", "hipaa", "pf")
ENGINEERED = ("diagnostic",)
DELTA = 0.05
ALPHA = 0.05
N_BOOT = 10000
SEED = 20260801

# The published repeated-holdout figures, for the protocol comparison.
HOLDOUT = {"nist": 0.0088, "hipaa": 0.0078, "pf": 0.0027, "diagnostic": 0.1405}


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


def exact_sign_test(diffs):
    """Exact paired sign test on discordant pairs.

    TESTS SUPERIORITY, NOT EQUIVALENCE. It asks whether wins and losses are
    balanced, which validates the sign-flip permutation result. It says
    nothing about whether an effect lies inside an equivalence margin.

    An earlier version of this script placed it beside the TOST boundary flag
    in a way that implied it provided an exact sensitivity check for the
    equivalence decision. It does not, and no exact equivalence test is
    reported here. Boundary-sensitive TOST results are therefore reported as
    approximate secondary results and are not headlined.
    """
    wins = int((diffs > 0).sum())
    losses = int((diffs < 0).sum())
    ties = int((diffs == 0).sum())
    n_disc = wins + losses
    p = (float(sps.binomtest(wins, n_disc, 0.5).pvalue) if n_disc
         else float("nan"))
    return {"wins": wins, "losses": losses, "ties": ties,
            "n_discordant": n_disc, "p_exact_two_sided": p,
            "method": "exact binomial on discordant pairs"}


def per_corpus(diffs, m_mean, r_mean):
    sf = cs.sign_flip_test(diffs, alpha=ALPHA)
    tost = cs.tost(diffs, DELTA, alpha=ALPHA)
    sign = exact_sign_test(diffs)
    boundary = abs(tost["p_tost"] - ALPHA) < 0.01
    return {
        "n": int(len(diffs)),
        "multi_top1": float(m_mean), "reform_top1": float(r_mean),
        "mean_difference": sf["mean_difference"],
        "ci_low": sf["ci_low"], "ci_high": sf["ci_high"],
        "ci_confidence": sf["confidence"],
        "p_sign_flip": sf["p_value"],
        "n_permutations": cs.PERMUTATION_N,
        "tost": {k: tost[k] for k in ("p_lower", "p_upper", "p_tost",
                                      "equivalent", "delta")},
        "tost_boundary_sensitive": bool(boundary),
        "exact_sign_test": sign,
    }


def analyse(csv_path):
    d = pd.read_csv(os.path.join(HERE, csv_path))
    out, diffs = {}, {}
    for corpus, key in NAME.items():
        s = d[d.corpus == corpus].pivot_table(index="rid", columns="method",
                                              values="top1")
        v = (s["reform"] - s["multi"]).values
        diffs[key] = v
        out[key] = per_corpus(v, s["multi"].mean(), s["reform"].mean())
    eng = np.concatenate([diffs[k] for k in ENGINEERED])
    ext = np.concatenate([diffs[k] for k in EXTERNAL])
    rng = np.random.default_rng(SEED)
    draws = np.array([eng[rng.integers(0, len(eng), len(eng))].mean()
                      - ext[rng.integers(0, len(ext), len(ext))].mean()
                      for _ in range(N_BOOT)])
    lo, hi = np.quantile(draws, [0.025, 0.975])
    out["_contrast"] = {
        "engineered_mean": float(eng.mean()), "engineered_n": int(len(eng)),
        "external_mean": float(ext.mean()), "external_n": int(len(ext)),
        "difference": float(eng.mean() - ext.mean()),
        "bootstrap_ci_low": float(lo), "bootstrap_ci_high": float(hi),
        "bootstrap_n": N_BOOT, "seed": SEED, "coverage": "95% two-sided",
        "conditional_on": "these four corpora only",
        "is_a_test_of_corpus_type": False,
        "why_not": "corpus type is a cluster-level property with one "
                   "engineered cluster; see heterogeneity_test.py",
    }
    return out


def main():
    print("EXPLORATORY, unplanned. Written in response to review.\n")
    results = {}
    for fit, csv in FITS.items():
        if not os.path.exists(os.path.join(HERE, csv)):
            print(f"  {csv} missing; run score_raa.py --lsi-fit {fit}")
            return 1
        results[fit] = analyse(csv)

    for fit in FITS:
        print(f"=== LSI fit: {fit} ===")
        print(f"{'corpus':11s} {'n':>4s} {'multi':>7s} {'reform':>7s} "
              f"{'delta':>8s} {'W/L/T':>10s} {'p_sign':>7s} {'p_exact':>8s} "
              f"{'p_TOST':>7s}")
        for k in ("diagnostic", "nist", "hipaa", "pf"):
            r = results[fit][k]; e = r["exact_sign_test"]
            flag = " *" if r["tost_boundary_sensitive"] else ""
            print(f"{k:11s} {r['n']:4d} {r['multi_top1']:7.4f} "
                  f"{r['reform_top1']:7.4f} {r['mean_difference']:+8.4f} "
                  f"{e['wins']:3d}/{e['losses']:2d}/{e['ties']:3d} "
                  f"{r['p_sign_flip']:7.4f} {e['p_exact_two_sided']:8.4f} "
                  f"{r['tost']['p_tost']:7.4f}{flag}")
        c = results[fit]["_contrast"]
        print(f"  contrast eng {c['engineered_mean']:+.4f} vs ext "
              f"{c['external_mean']:+.4f} = {c['difference']:+.4f}, "
              f"CI [{c['bootstrap_ci_low']:+.4f}, {c['bootstrap_ci_high']:+.4f}]"
              f" (conditional, not a test)")
        print()

    print("Isolating the LSI fit from the evaluation protocol:")
    print(f"{'corpus':11s} {'holdout':>9s} {'1pass/trans':>12s} {'1pass/induc':>12s}")
    for k in ("diagnostic", "nist", "hipaa", "pf"):
        print(f"{k:11s} {HOLDOUT[k]:+9.4f} "
              f"{results['transductive'][k]['mean_difference']:+12.4f} "
              f"{results['inductive'][k]['mean_difference']:+12.4f}")
    print("\nHolding the protocol fixed at one-pass, changing only the LSI fit")
    print("moves the external estimates by more than changing the protocol at")
    print("a fixed transductive fit does. The published estimates were")
    print("sensitive to the LSI fitting regime, not mainly to the protocol.")

    record = {
        "status": "exploratory",
        "status_reason": "unplanned analysis written in response to review, "
                         "after the per-corpus effects were known",
        "contrast": "reform - multi, one scoring pass per requirement",
        "delta": DELTA, "alpha": ALPHA,
        "seeds": {"bootstrap": cs.BOOTSTRAP_SEED,
                  "permutation": cs.PERMUTATION_SEED, "contrast": SEED},
        "iterations": {"bootstrap": cs.BOOTSTRAP_N,
                       "permutation": cs.PERMUTATION_N,
                       "contrast_bootstrap": N_BOOT},
        "results_by_lsi_fit": results,
        "published_repeated_holdout": HOLDOUT,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "argv": sys.argv, "git": git_state(),
        "spec_hashes": {
            "analyze_onepass.py": sha256(os.path.abspath(__file__)),
            "score_raa.py": sha256(os.path.join(HERE, "score_raa.py")),
            "confirmatory_stats.py": sha256(
                os.path.join(HERE, "confirmatory_stats.py"))},
        "input_hashes": {c: sha256(os.path.join(HERE, c))
                         for c in FITS.values()},
        "environment": {"python": sys.version.split()[0],
                        "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "scipy": __import__("scipy").__version__},
    }
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "reform_vs_multi_onepass.json")
    with open(out, "w", newline="\n") as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"\nrecord written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Recorded provenance for the two Spearman families the manuscript cites.

Both families appeared in drafts with no committed script behind them, which
breaks the rule at the top of run_confirmatory.py: nothing enters the
manuscript except by citing a record. This produces that record.

FAMILY 1  gap vs TF-IDF correctness
    Does a requirement's lexical distance from its gold controls predict that
    term matching fails on it? Expected negative and reported as the measure's
    face validity.

FAMILY 2  gap vs the dual-encoder's Top-1 advantage over TF-IDF
    Does lexical distance predict that a learned representation helps? The
    per-requirement advantage is semantic Top-1 minus TF-IDF Top-1, so it
    takes values in {-1, 0, +1}. Reported because the answer is no.

MULTIPLICITY
Eight tests are run, four per family. They are secondary and descriptive, so
the manuscript's claims rest on the pattern rather than on any single
p-value. Holm-Bonferroni adjusted p-values are reported WITHIN each family of
four, and both raw and adjusted values are recorded so a reader can apply a
different correction. No claim in the manuscript depends on a test that
survives raw but not adjusted.

TIES
Top-1 correctness is binary and the advantage variable takes three values, so
both families have heavy ties against a continuous gap score. Spearman's rho
with tie correction is used, which is what scipy computes; the p-values come
from an exact permutation of the gap vector rather than the asymptotic t
approximation, because the tie structure makes the latter unreliable.

STATUS: exploratory, unplanned. Not covered by the hybrid preregistration.

Usage:
    python gap_correlations.py
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

import gap_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "results_v3", "confirmatory")
SCORES = "shared_ranking_scores.csv"

CORPORA = {
    "diagnostic": ("diagnostic_benchmark", "diag_"),
    "nist": ("csf_benchmark", "csf_"),
    "hipaa": ("hipaa_benchmark", ""),
    "pf": ("pf_benchmark", ""),
}
LEXICAL = "tfidf"
SEMANTIC = "semantic"
N_PERM = 100000
SEED = 20260801
ALPHA = 0.05


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


def spearman_perm(x, y, rng):
    """Spearman rho with a permutation p-value.

    The asymptotic p-value assumes no ties; y here is binary or ternary, so
    it is heavily tied. Permuting x against a fixed y preserves the tie
    structure exactly and gives a valid two-sided p-value under the null of
    no association.
    """
    rho = float(sps.spearmanr(x, y).statistic)
    x = np.asarray(x, dtype=float)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        null[i] = sps.spearmanr(rng.permutation(x), y).statistic
    p = (1.0 + np.sum(np.abs(null) >= abs(rho) - 1e-12)) / (N_PERM + 1.0)
    return rho, float(p)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return [float(a) for a in adj]


def main():
    print("EXPLORATORY, unplanned. Secondary descriptive correlations.\n")
    scores = pd.read_csv(os.path.join(HERE, SCORES))
    rng = np.random.default_rng(SEED)

    families = {"gap_vs_lexical_failure": {}, "gap_vs_semantic_advantage": {}}
    for key, (bench, prefix) in CORPORA.items():
        regs, controls, mappings = gap_metrics.load_corpus(
            os.path.join(HERE, bench), prefix)
        gaps = gap_metrics.compute_gaps(regs, controls, mappings)

        sub = scores[scores.corpus == bench]
        wide = sub.pivot_table(index="rid", columns="method", values="top1")
        for need in (LEXICAL, SEMANTIC):
            if need not in wide.columns:
                raise SystemExit(f"{key}: {SCORES} has no {need} rows")
        shared = sorted(set(wide.index) & set(gaps))
        g = np.array([gaps[r]["gap"] for r in shared], dtype=float)
        lex = wide.loc[shared, LEXICAL].to_numpy(dtype=float)
        sem = wide.loc[shared, SEMANTIC].to_numpy(dtype=float)

        rho1, p1 = spearman_perm(g, lex, rng)
        rho2, p2 = spearman_perm(g, sem - lex, rng)
        families["gap_vs_lexical_failure"][key] = {
            "n": len(shared), "rho": rho1, "p_raw": p1}
        families["gap_vs_semantic_advantage"][key] = {
            "n": len(shared), "rho": rho2, "p_raw": p2}

    for fam, per in families.items():
        keys = list(CORPORA)
        adj = holm([per[k]["p_raw"] for k in keys])
        for k, a in zip(keys, adj):
            per[k]["p_holm"] = a
            per[k]["significant_raw"] = bool(per[k]["p_raw"] < ALPHA)
            per[k]["significant_holm"] = bool(a < ALPHA)

    for fam, per in families.items():
        print(fam)
        print(f"  {'corpus':11s} {'n':>4s} {'rho':>8s} {'p_raw':>9s} "
              f"{'p_holm':>9s}")
        for k in CORPORA:
            r = per[k]
            print(f"  {k:11s} {r['n']:4d} {r['rho']:+8.3f} {r['p_raw']:9.5f} "
                  f"{r['p_holm']:9.5f}"
                  f"{'  *' if r['significant_holm'] else ''}")
        print()

    record = {
        "status": "exploratory",
        "status_reason": "secondary descriptive correlations, not "
                         "preregistered",
        "families": families,
        "multiplicity": {
            "method": "Holm-Bonferroni within each family of four",
            "n_tests_total": 8,
            "note": "both raw and adjusted p-values are recorded; no "
                    "manuscript claim depends on a test that survives raw "
                    "but not adjusted",
        },
        "p_value_method": {
            "method": "permutation of the gap vector against a fixed outcome",
            "n_permutations": N_PERM,
            "seed": SEED,
            "why": "the outcome is binary or ternary, so the asymptotic "
                   "Spearman p-value, which assumes untied data, is not "
                   "reliable here; permuting preserves the tie structure",
        },
        "scope_limit": "gap is fitted with corpus-relative IDF, so these "
                       "coefficients describe within-corpus association only "
                       "and no corpus can be called wider-gapped than another",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "argv": sys.argv,
        "git": git_state(),
        "spec_hashes": {
            "gap_correlations.py": sha256(os.path.abspath(__file__)),
            "gap_metrics.py": gap_metrics.spec_hash(),
        },
        "input_hashes": {SCORES: sha256(os.path.join(HERE, SCORES))},
        "environment": {"python": sys.version.split()[0],
                        "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "scipy": sps.__name__ and __import__("scipy").__version__},
    }
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "gap_correlations.json")
    with open(out, "w", newline="\n") as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"record written: {os.path.relpath(out, HERE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Recorded-invocation runner for the RQ2 moderation and equivalence results.

Every number this prints is also written to a JSON record under
results_v3/confirmatory/ carrying the argv, the UTC timestamp, the git commit
and dirty flag, the interpreter and library versions, SHA-256 hashes of the
analysis modules and of every input CSV, and the seeds. Nothing from this
analysis may enter the manuscript except by citing one of those records.

That rule exists because the moderation figure previously quoted in drafts
("+0.0043, p=0.82") lived only in a scratchpad: no recorded inputs, no seed,
and a p-value from the discredited bootstrap-crossing-zero rule. Numbers
produced here supersede it and are not comparable to it.

STATUS: EXPLORATORY. The HIPAA and PF ablations were run before any analysis
spec was timestamped, and those ablations compute the reform-vs-multi
contrast that is the test itself. Confirmatory status cannot be claimed on
these corpora at any later date. See HANDOFF.md.

TWO SOURCES OF PAIRED DIFFERENCES
  --source holdout  reads results_v3/<corpus>/perquery_{multi,reform}.csv and
                    collapses each requirement to its mean over the splits it
                    was tested in. This is the repeated-holdout regime, with
                    LSI fitted on controls plus the train and calibration
                    requirements.
  --source onepass  reads one scoring pass per requirement over the full
                    control corpus from shared_raa_scores.csv, with LSI fitted
                    on control documents alone.

The two are different estimands and their numbers are not interchangeable.
The moderation coefficient must come from whichever source the manuscript
reports the per-corpus effects from, or the interaction is being fitted to
differences the paper does not otherwise show.

Usage:
  python run_confirmatory.py [--delta 0.05] [--metric top1]
                             [--corpora nist,hipaa,pf] [--tag NAME]
                             [--source holdout|onepass]
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
import scipy

import confirmatory_stats as cs
import gap_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results_v3")
OUTDIR = os.path.join(RESULTS, "confirmatory")

# results_v3 subdirectory -> (benchmark directory, csv prefix). 'nist' is the
# CSF 1.1 corpus; the directory kept its original name.
CORPORA = {
    "nist": ("csf_benchmark", "csf_"),
    "hipaa": ("hipaa_benchmark", ""),
    "pf": ("pf_benchmark", ""),
    "diagnostic": ("diagnostic_benchmark", "diag_"),
}

TREATMENT = "reform"
BASELINE = "multi"

ONEPASS_CSV = "shared_raa_scores.csv"
ONEPASS_METRIC = {"top1": "top1", "rr": "rr@5"}


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def git_state():
    def run(*args):
        try:
            return subprocess.run(args, cwd=HERE, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return None
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain"))}


def per_query_means(path, metric):
    """One value per requirement: its mean over the seeds it was tested in.

    The repeated-holdout design tests requirements an uneven number of times
    (HIPAA: 2 to 12 across 30 seeds), so this collapses to one row per
    requirement before any comparison. It does NOT repair the estimand
    mismatch against one-pass methods; see protocol rule 1 in HANDOFF.md.
    """
    df = pd.read_csv(path)
    return df.groupby("rid")[metric].mean()


def onepass_series(name, metric):
    """One value per requirement from the single full-corpus scoring pass.

    No aggregation is involved: each requirement is scored once, so there is
    nothing to average over. Returned in the same shape as per_query_means so
    the two sources are interchangeable downstream.
    """
    bench, _ = CORPORA[name]
    col = ONEPASS_METRIC[metric]
    df = pd.read_csv(os.path.join(HERE, ONEPASS_CSV))
    df = df[df.corpus == bench]
    if df.empty:
        raise SystemExit(f"{name}: no rows for {bench} in {ONEPASS_CSV}")
    wide = df.pivot_table(index="rid", columns="method", values=col)
    for v in (TREATMENT, BASELINE):
        if v not in wide.columns:
            raise SystemExit(f"{name}: {ONEPASS_CSV} has no {v} rows")
    return wide[TREATMENT], wide[BASELINE]


def load_corpus_diffs(name, metric, source):
    """Paired per-requirement difference and gap score for one corpus."""
    bench, prefix = CORPORA[name]
    if source == "onepass":
        treat, base = onepass_series(name, metric)
        paths = {"onepass": os.path.join(HERE, ONEPASS_CSV)}
    else:
        rdir = os.path.join(RESULTS, name)
        paths = {v: os.path.join(rdir, f"perquery_{v}.csv")
                 for v in (TREATMENT, BASELINE)}
        treat = per_query_means(paths[TREATMENT], metric)
        base = per_query_means(paths[BASELINE], metric)

    regs, controls, mappings = gap_metrics.load_corpus(
        os.path.join(HERE, bench), prefix)
    gaps = gap_metrics.compute_gaps(regs, controls, mappings)

    shared = sorted(set(treat.index) & set(base.index) & set(gaps))
    if not shared:
        raise SystemExit(f"{name}: no requirements shared between "
                         f"{TREATMENT}, {BASELINE}, and the gap scores")
    diffs = np.array([treat.loc[r] - base.loc[r] for r in shared], dtype=float)
    gapv = np.array([gaps[r]["gap"] for r in shared], dtype=float)

    inputs = {os.path.relpath(p, HERE): sha256(p) for p in paths.values()}
    for fname in ("regs.csv", "controls.csv", "mappings.csv"):
        p = os.path.join(HERE, bench, prefix + fname)
        inputs[os.path.relpath(p, HERE)] = sha256(p)

    return {"rids": shared, "diffs": diffs, "gaps": gapv,
            "n": len(shared), "source": source,
            "n_treatment_rows": int(len(treat)),
            "n_baseline_rows": int(len(base)),
            "n_with_gap": int(len(gaps)),
            "mean_treatment": float(treat.loc[shared].mean()),
            "mean_baseline": float(base.loc[shared].mean()),
            "mean_gap": float(gapv.mean()),
            "inputs": inputs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=0.05,
                    help="equivalence margin on the primary endpoint")
    ap.add_argument("--metric", default="top1", choices=["top1", "rr"])
    ap.add_argument("--corpora", default="nist,hipaa,pf",
                    help="comma-separated results_v3 subdirectories")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--tag", default=None, help="output filename stem")
    ap.add_argument("--source", default="holdout",
                    choices=("holdout", "onepass"),
                    help="where the paired differences come from; see the "
                         "module docstring, the two are different estimands")
    args = ap.parse_args()

    names = [c.strip() for c in args.corpora.split(",") if c.strip()]
    unknown = [c for c in names if c not in CORPORA]
    if unknown:
        raise SystemExit(f"unknown corpora: {unknown}")

    print("=" * 72)
    print("EXPLORATORY ANALYSIS. Not preregistered, not confirmatory.")
    print("The ablations that produced these rows predate any timestamped")
    print("spec and already contained this contrast. See HANDOFF.md.")
    print("=" * 72)
    if args.source == "onepass":
        print("source: one scoring pass per requirement over the full control")
        print("corpus, LSI fitted on controls only.")
    else:
        print("source: repeated stratified holdouts, LSI fitted on controls")
        print("plus the train and calibration requirements.")
    print()

    per_corpus = {name: load_corpus_diffs(name, args.metric, args.source)
                  for name in names}

    diffs = np.concatenate([per_corpus[n]["diffs"] for n in names])
    gaps = np.concatenate([per_corpus[n]["gaps"] for n in names])
    labels = np.concatenate([[n] * per_corpus[n]["n"] for n in names])

    moderation = cs.moderation_test(diffs, gaps, labels, alpha=args.alpha)

    # cs.moderation_test names a direction from the sign of the point estimate
    # alone: "benefit grows with gap" whenever it is positive, whether or not
    # the interval excludes zero. That string is written into records the
    # manuscript cites, so it is overridden here rather than in
    # confirmatory_stats.py. That module is one of the seven artifacts hashed
    # in the OSF registration (doi:10.17605/OSF.IO/NZXRV) and must stay
    # byte-identical to the registered version; editing it would break the
    # hash for a reason unrelated to the registered analysis, which does not
    # call moderation_test at all.
    if moderation["ci_low"] <= 0.0 <= moderation["ci_high"]:
        moderation["interpretation"] = ("no direction identified; interval "
                                        "includes zero")
    moderation["interpretation_note"] = (
        "direction reported only when the bootstrap interval excludes zero; "
        "see run_confirmatory.py, not confirmatory_stats.py")
    superiority = {n: cs.sign_flip_test(per_corpus[n]["diffs"], alpha=args.alpha)
                   for n in names}
    equivalence = {n: cs.tost(per_corpus[n]["diffs"], args.delta, alpha=args.alpha)
                   for n in names}
    descriptive = {n: cs.equivalence_verdict(per_corpus[n]["diffs"], args.delta,
                                             conf=1.0 - 2.0 * args.alpha)
                   for n in names}

    record = {
        "status": "exploratory",
        "status_reason": "analysis spec not timestamped before the "
                         "reform-vs-multi contrast was computed",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "argv": sys.argv,
        "contrast": f"{TREATMENT} - {BASELINE}",
        "metric": args.metric,
        "source": args.source,
        "source_note": (
            "one scoring pass per requirement over the full control corpus, "
            "LSI fitted on control documents only"
            if args.source == "onepass" else
            "repeated stratified holdouts collapsed to one value per "
            "requirement, LSI fitted on controls plus train and calibration "
            "requirements"),
        "delta": args.delta,
        "alpha": args.alpha,
        "git": git_state(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "seeds": {
            "bootstrap": cs.BOOTSTRAP_SEED,
            "permutation": cs.PERMUTATION_SEED,
            "bootstrap_n": cs.BOOTSTRAP_N,
            "permutation_n": cs.PERMUTATION_N,
        },
        "spec_hashes": {
            "gap_metrics.py": gap_metrics.spec_hash(),
            "confirmatory_stats.py": sha256(
                os.path.join(HERE, "confirmatory_stats.py")),
            "run_confirmatory.py": sha256(os.path.abspath(__file__)),
        },
        "corpora": {n: {k: v for k, v in per_corpus[n].items()
                        if k not in ("diffs", "gaps", "rids")}
                    for n in names},
        "moderation_pooled": moderation,
        "superiority_sign_flip": superiority,
        "equivalence_tost": equivalence,
        "equivalence_verdict_descriptive": descriptive,
    }

    os.makedirs(OUTDIR, exist_ok=True)
    stem = args.tag or f"{TREATMENT}_vs_{BASELINE}_{args.metric}"
    path = os.path.join(OUTDIR, f"{stem}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True)

    print(f"contrast: {TREATMENT} - {BASELINE} on {args.metric}\n")
    for n in names:
        c = per_corpus[n]
        s = superiority[n]
        print(f"{n:<12} n={c['n']:<4} mean gap={c['mean_gap']:.3f}  "
              f"{BASELINE}={c['mean_baseline']:.4f} "
              f"{TREATMENT}={c['mean_treatment']:.4f}")
        print(f"{'':<12} delta={s['mean_difference']:+.4f}  "
              f"{int(s['confidence'] * 100)}% CI "
              f"[{s['ci_low']:+.4f}, {s['ci_high']:+.4f}]  "
              f"sign-flip p={s['p_value']:.4f}")
        e = equivalence[n]
        print(f"{'':<12} TOST delta={args.delta}: p_lower={e['p_lower']:.4f} "
              f"p_upper={e['p_upper']:.4f} p_tost={e['p_tost']:.4f} "
              f"equivalent={e['equivalent']}")
        print(f"{'':<12} descriptive verdict: {descriptive[n]['verdict']}")
        print()

    m = moderation
    print(f"pooled moderation (gap x reformulation), n={m['n']} over "
          f"{len(names)} corpora with corpus fixed effects:")
    print(f"  coefficient = {m['coefficient']:+.4f}  "
          f"{int((1 - args.alpha) * 100)}% bootstrap CI "
          f"[{m['ci_low']:+.4f}, {m['ci_high']:+.4f}]")
    print(f"  sign-flip permutation p = {m['p_value']:.4f} "
          f"({m['n_permutations']} permutations)")
    print(f"  {m['interpretation']}")
    print(f"\nrecord written: {os.path.relpath(path, HERE)}")


if __name__ == "__main__":
    main()

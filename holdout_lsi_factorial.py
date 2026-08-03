#!/usr/bin/env python3
"""The missing factorial cell: repeated holdout with controls-only LSI.

Two things changed at once between the previously reported reformulation
figures and the one-pass figures: the evaluation protocol, and the population
LSI is fitted on. Neither comparison available so far separates them.

    A  repeated holdout + controls-only LSI      <- produced here
    B  repeated holdout + train/cal-assisted LSI <- previously reported
    C  one-pass        + controls-only LSI       <- score_raa.py default

    A vs B isolates the LSI fitting regime, protocol held fixed.
    A vs C isolates the evaluation protocol, LSI fit held fixed.

The one-pass transductive arm in score_raa.py is NOT cell B. It fits on every
requirement including the ones being scored, whereas B fits on train and
calibration requirements only, excluding the scored ones. It is a
distribution-aware sensitivity, not the holdout regime.

This script reproduces the holdout protocol exactly as raa_agent.run_variant
does for the ranking arms: the same stratified split function, the same ratios,
the same seeds, the same multi and reform configurations. The only deliberate
difference is that LSI is fitted on control documents alone.

Ranking only. Thresholds are disabled, so no coverage, selective accuracy or
gap detection is produced; those still belong to the calibrated holdout runs.

STATUS: exploratory, unplanned. Written in response to review.

Usage:
    USE_TF=0 python holdout_lsi_factorial.py
    USE_TF=0 python holdout_lsi_factorial.py --runs 5 --corpus pf
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys

os.environ.setdefault("USE_TF", "0")

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "results_v3", "shared")

CORPORA = {
    "nist": ("csf_benchmark", "csf_"),
    "hipaa": ("hipaa_benchmark", ""),
    "pf": ("pf_benchmark", ""),
    "diagnostic": ("diagnostic_benchmark", "diag_"),
}
# Matching the published ablation protocol: raa_agent.parse_args defaults
# --holdout 0.20 and --cal 0.15, not 0.30/0.20. Getting these wrong made the
# reimplementation fail to reproduce HIPAA and PF, since the test-set size per
# split changes and with it which requirements are scored how often. Verified
# against the original per-seed test-set sizes: HIPAA gives 13 per split under
# 0.20 across its five framework strata, which is what the released
# perquery CSVs contain.
HOLDOUT_RATIO = 0.20
CAL_RATIO = 0.15
N_RUNS = 30
BASE_SEED = 42

VARIANTS = {
    "multi": dict(enable_multi=True, enable_reform=False,
                  enable_decompose=False, enable_crossref=False,
                  enable_verify=False),
    "reform": dict(enable_multi=True, enable_reform=True,
                   enable_decompose=False, enable_crossref=False,
                   enable_verify=False),
}


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


def run_corpus(key, directory, prefix, n_runs, lsi_fit):
    from raa_agent import (AgentTools, ComplianceAgent, Control, LSIIndex,
                           Regulation, build_bm25_scorer, build_tfidf_scorer,
                           stratified_split)

    d = os.path.join(HERE, directory)
    regs_df = pd.read_csv(f"{d}/{prefix}regs.csv")
    ctls_df = pd.read_csv(f"{d}/{prefix}controls.csv")
    maps_df = pd.read_csv(f"{d}/{prefix}mappings.csv")

    gold = {}
    for r, c in zip(maps_df["regulation_id"], maps_df["control_id"]):
        gold.setdefault(int(r), set()).add(int(c))

    ctrl_texts = list(ctls_df["text"])
    ctrl_ids = [int(c) for c in ctls_df["id"]]
    fam = (list(ctls_df["family"]) if "family" in ctls_df.columns
           else ["NA"] * len(ctrl_ids))
    controls = [Control(control_id=i, text=t, regulation_id=-1, quality=1.0,
                        match_type="NA", family=str(f))
                for i, (t, f) in enumerate(zip(ctrl_texts, fam))]
    fw = (list(regs_df["framework"]) if "framework" in regs_df.columns
          else ["NA"] * len(regs_df))
    regs = {int(r): Regulation(regulation_id=int(r), text=t, framework=str(f))
            for r, t, f in zip(regs_df["id"], regs_df["text"], fw)}
    reg_list = [regs[r] for r in sorted(regs)]
    gt_lists = {r: sorted(gold.get(r, [])) for r in regs}

    tfidf = build_tfidf_scorer(ctrl_texts)
    bm25 = build_bm25_scorer(ctrl_texts)

    # Per-requirement accumulator, mirroring per_query_means in
    # run_confirmatory: a requirement contributes the mean over the splits it
    # was tested in, so requirements tested more often are not weighted more.
    acc = {v: {} for v in VARIANTS}

    for run in range(n_runs):
        seed = BASE_SEED + run
        train_ids, cal_ids, test_ids = stratified_split(
            reg_list, HOLDOUT_RATIO, CAL_RATIO, seed)
        if lsi_fit == "controls_only":
            lsi = LSIIndex(ctrl_texts, [], n_components=100,
                           include_regs_in_fit=False).score
        else:  # train/cal-assisted, the published regime
            fit_texts = [regs[r].text for r in (train_ids + cal_ids)]
            lsi = LSIIndex(ctrl_texts, fit_texts, n_components=100,
                           include_regs_in_fit=True).score
        scorers = {"tfidf": tfidf, "bm25": bm25, "lsi": lsi}
        tools = AgentTools(scorers, controls, reg_list, gt_lists)

        for name, cfg in VARIANTS.items():
            agent = ComplianceAgent(tools=tools, conf_thr=0.0, gap_thr=0.0,
                                    rel_gap_retry_threshold=0.10, **cfg)
            for rid in test_ids:
                if rid not in gold:
                    continue
                trace = agent.solve(regs[rid], primary_backend="bm25")
                top = ctrl_ids[trace.decision.ranked[0].control_id]
                acc[name].setdefault(rid, []).append(
                    1.0 if top in gold[rid] else 0.0)
        if (run + 1) % 5 == 0:
            print(f"    {key}: {run + 1}/{n_runs} splits", flush=True)

    shared = set(acc["multi"]) & set(acc["reform"])
    m = np.array([np.mean(acc["multi"][r]) for r in sorted(shared)])
    f = np.array([np.mean(acc["reform"][r]) for r in sorted(shared)])
    times = [len(acc["multi"][r]) for r in sorted(shared)]
    return {"n": len(shared), "multi_top1": float(m.mean()),
            "reform_top1": float(f.mean()),
            "mean_difference": float((f - m).mean()),
            "times_tested_min": int(min(times)),
            "times_tested_max": int(max(times))}, (f - m)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=sorted(CORPORA))
    ap.add_argument("--runs", type=int, default=N_RUNS)
    ap.add_argument("--lsi-fit", choices=("controls_only", "train_cal"),
                    default="controls_only")
    args = ap.parse_args()

    import confirmatory_stats as cs

    print("EXPLORATORY, unplanned. Repeated holdout, ranking only.")
    print(f"LSI fit: {args.lsi_fit};  splits: {args.runs};  "
          f"holdout {HOLDOUT_RATIO}, cal {CAL_RATIO}, base seed {BASE_SEED}\n")

    keys = [args.corpus] if args.corpus else list(CORPORA)
    out, diffs = {}, {}
    for key in keys:
        directory, prefix = CORPORA[key]
        if not os.path.isdir(os.path.join(HERE, directory)):
            print(f"  {key}: missing, skipped")
            continue
        r, dv = run_corpus(key, directory, prefix, args.runs, args.lsi_fit)
        sf = cs.sign_flip_test(dv, alpha=0.05)
        r.update({"ci_low": sf["ci_low"], "ci_high": sf["ci_high"],
                  "p_sign_flip": sf["p_value"],
                  "wins": int((dv > 0).sum()), "losses": int((dv < 0).sum()),
                  "ties": int((dv == 0).sum())})
        out[key] = r
        diffs[key] = dv
        print(f"  {key:11s} n={r['n']:3d} multi={r['multi_top1']:.4f} "
              f"reform={r['reform_top1']:.4f} delta={r['mean_difference']:+.4f} "
              f"p={r['p_sign_flip']:.4f} "
              f"(tested {r['times_tested_min']}-{r['times_tested_max']}x)")

    record = {
        "status": "exploratory",
        "status_reason": "unplanned; written in response to review",
        "cell": f"repeated holdout + {args.lsi_fit} LSI",
        "ranking_only": True,
        "protocol": {"holdout_ratio": HOLDOUT_RATIO, "cal_ratio": CAL_RATIO,
                     "n_splits": args.runs, "base_seed": BASE_SEED,
                     "aggregation": "per-requirement mean over the splits it "
                                    "was tested in"},
        "lsi_fit": args.lsi_fit,
        "results": out,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "argv": sys.argv, "git": git_state(),
        "spec_hashes": {
            "holdout_lsi_factorial.py": sha256(os.path.abspath(__file__)),
            "raa_agent.py": sha256(os.path.join(HERE, "raa_agent.py"))},
        "environment": {"python": sys.version.split()[0],
                        "platform": platform.platform(),
                        "numpy": np.__version__},
    }
    os.makedirs(OUTDIR, exist_ok=True)
    stem = f"holdout_{args.lsi_fit}.json"
    with open(os.path.join(OUTDIR, stem), "w", newline="\n") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"\nrecord written: results_v3/shared/{stem}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Score the full RAA pipeline once per requirement, on the shared population.

WHY THIS EXISTS
---------------
score_all.py scores every frozen retrieval method once per requirement over the
full control corpus, giving one estimand for all of them. The agent was never
scored that way. Its numbers came from repeated stratified holdouts, so any
table putting RAA beside those methods was mixing populations, and the paper
could not honestly write "RAA" in a cross-corpus comparison.

HANDOFF.md states the rule this fixes: rrf_lexical is NOT RAA. RAA adds
conditional reformulation and decomposition on top of multi-backend fusion,
and those can reorder. Until the full pipeline runs on the shared protocol,
no arm in the shared table may carry the name.

WHY THIS IS VALID
-----------------
The agent's RANKING needs no training. Reformulation fires on the top-2
relative margin of the current scores, decomposition on the same signal, and
neither consults a label. Corroboration and verification are decision-only
after the round-two fixes, so they cannot reorder. What the holdout splits
were ever needed for is the accept/abstain THRESHOLD, which affects
Decision.status and not Decision.ranked.

This script therefore runs the pipeline once per requirement over the whole
corpus with the thresholds set to accept everything, and records only ranking
metrics. It produces NO decision metrics. Coverage, selective accuracy and
gap detection still require calibration splits and are not reported here.

LSI FIT
-------
run_variant fits LSI on controls plus the train and calibration requirement
texts. With no splits there is no train set, so this script fits LSI on
control documents only, matching the inductive variant used elsewhere in the
shared protocol. This is a deliberate deviation from the published agent
configuration and is recorded in the output.

Usage:
    USE_TF=0 python score_raa.py
    USE_TF=0 python score_raa.py --corpus pf
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

CORPORA = {
    "nist": ("csf_benchmark", "csf_"),
    "hipaa": ("hipaa_benchmark", ""),
    "pf": ("pf_benchmark", ""),
    "diagnostic": ("diagnostic_benchmark", "diag_"),
}
TOPK = 5


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


def metrics(order_ids, gold_set, k=TOPK):
    """Identical to score_all.metrics, so rows are directly comparable."""
    top1 = 1.0 if order_ids[0] in gold_set else 0.0
    rr = 0.0
    for rank, cid in enumerate(order_ids[:k], start=1):
        if cid in gold_set:
            rr = 1.0 / rank
            break
    hits = sum(1 for cid in order_ids[:k] if cid in gold_set)
    return {"top1": top1, "predicted_top1": order_ids[0], "rr@5": rr,
            "recall@5": hits / len(gold_set) if gold_set else 0.0,
            "precision@5": hits / k,
            "first_gold_rank": next(
                (i + 1 for i, cid in enumerate(order_ids) if cid in gold_set),
                None)}


def score_corpus(key, directory, prefix, variants):
    from raa_agent import (AgentTools, ComplianceAgent, Control, LSIIndex,
                           Regulation, build_bm25_scorer, build_tfidf_scorer)

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

    print(f"  {key}: {len(ctrl_ids)} controls, building scorers...", flush=True)
    scorers = {
        "tfidf": build_tfidf_scorer(ctrl_texts),
        "bm25": build_bm25_scorer(ctrl_texts),
        # controls only: no train split exists in a one-pass protocol
        "lsi": LSIIndex(ctrl_texts, [], n_components=100,
                        include_regs_in_fit=False).score,
    }

    reg_list = [regs[r] for r in sorted(regs)]
    gt_lists = {r: sorted(gold.get(r, [])) for r in regs}
    tools = AgentTools(scorers, controls, reg_list, gt_lists)

    rows = []
    for name, cfg in variants.items():
        agent = ComplianceAgent(tools=tools, conf_thr=0.0, gap_thr=0.0,
                                rel_gap_retry_threshold=0.10, **cfg)
        fired = 0
        for rid in sorted(regs):
            if rid not in gold:
                continue
            trace = agent.solve(regs[rid], primary_backend="bm25")
            order = [ctrl_ids[c.control_id] for c in trace.decision.ranked]
            m = metrics(order, gold[rid])
            m.update({"corpus": directory, "method": name, "rid": rid,
                      "n_gold": len(gold[rid])})
            rows.append(m)
            if "reformulate" in trace.tools_used:
                fired += 1
        n = sum(1 for r in regs if r in gold)
        print(f"    {name:16s} n={n:3d}  Top-1={np.mean([r['top1'] for r in rows if r['method']==name]):.4f}"
              f"  reform fired {fired}/{n}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=sorted(CORPORA))
    ap.add_argument("--out", default="shared_raa_scores.csv")
    args = ap.parse_args()

    # Two things are needed and they are not the same contrast.
    #
    # multi vs reform isolates reformulation exactly as the published ablation
    # does: identical configuration except the reformulation step. An earlier
    # version of this script compared full RAA against a variant with both
    # reformulation AND decomposition disabled, which confounds the two.
    #
    # raa_full is the whole system, for arms that need to carry the name.
    variants = {
        "multi": dict(enable_multi=True, enable_reform=False,
                      enable_decompose=False, enable_crossref=False,
                      enable_verify=False),
        "reform": dict(enable_multi=True, enable_reform=True,
                       enable_decompose=False, enable_crossref=False,
                       enable_verify=False),
        "raa_full": dict(enable_multi=True, enable_reform=True,
                         enable_decompose=True, enable_crossref=True,
                         enable_verify=True),
    }

    print("Full RAA on the shared population. Ranking metrics only;")
    print("decision metrics need calibration splits and are not produced.\n")

    keys = [args.corpus] if args.corpus else list(CORPORA)
    frames = []
    for key in keys:
        directory, prefix = CORPORA[key]
        if not os.path.isdir(os.path.join(HERE, directory)):
            print(f"  {key}: {directory} missing, skipped")
            continue
        frames.append(score_corpus(key, directory, prefix, variants))

    if not frames:
        return 1
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(os.path.join(HERE, args.out), index=False)
    print(f"\nwrote {args.out}  ({len(out)} rows)")

    piv = out.pivot_table(index="corpus", columns="method", values="top1",
                          aggfunc="mean")
    piv.index = [i.replace("_benchmark", "") for i in piv.index]
    print("\nTop-1, one value per requirement over the full corpus:")
    print(piv.round(4).to_string())

    record = {
        "status": "exploratory",
        "purpose": "full RAA scored once per requirement on the shared "
                   "population, so it can be compared with score_all.py rows",
        "ranking_only": True,
        "no_decision_metrics_reason": "coverage, selective accuracy and gap "
                                      "detection require calibration splits",
        "lsi_fit": "control documents only; run_variant fits on controls plus "
                   "train and calibration requirements, which do not exist in "
                   "a one-pass protocol",
        "primary_backend": "bm25",
        "rel_gap_retry_threshold": 0.10,
        "thresholds": "conf_thr=gap_thr=0.0; ranking is independent of them",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "argv": sys.argv, "git": git_state(),
        "spec_hashes": {"score_raa.py": sha256(os.path.abspath(__file__)),
                        "raa_agent.py": sha256(os.path.join(HERE, "raa_agent.py"))},
        "environment": {"python": sys.version.split()[0],
                        "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__},
        "top1": {c: {m: float(v) for m, v in row.items()}
                 for c, row in piv.iterrows()},
    }
    rd = os.path.join(HERE, "results_v3", "shared")
    os.makedirs(rd, exist_ok=True)
    with open(os.path.join(rd, "raa_shared_population.json"), "w",
              newline="\n") as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"record written: results_v3/shared/raa_shared_population.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

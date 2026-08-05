#!/usr/bin/env python3
"""Sweep the reformulation trigger constant under the PRIMARY protocol.

WHY THIS EXISTS
The sensitivity figures previously cited in the manuscript came from
results_v3/sensitivity/sensitivity.csv, which no committed script produces.
It was assembled by hand from raa_agent.py runs, and its diagnostic +Reform
Top-1 (0.658) sits above the ablation table's (0.644) with nothing on record
to account for the difference. Citing a number nobody can regenerate, in a
paper whose argument rests on provenance, is the wrong way round.

This replaces it. The sweep runs under the protocol the paper's ranking
claims use: one scoring pass per requirement over the full control corpus,
LSI fitted on control documents alone. That also makes the output directly
comparable to Table~\\ref{tab:multi} rather than to a different regime.

WHAT IT ANSWERS
tau_rel is the relative top-2 margin below which the reformulation tool is
invoked. It was set to 0.10 by hand. If the reported reformulation effect
moved appreciably with it, the effect would be an artifact of a tuned
constant. The sweep reports, for each setting and corpus, the reform-minus-
multi Top-1 difference and the four instrumentation counts, so a reader can
see both whether the effect moves and whether the gate is doing any
selection at all.

STATUS: exploratory, unplanned. Not covered by the hybrid preregistration.

Usage:
    USE_TF=0 python sensitivity_sweep.py
    USE_TF=0 python sensitivity_sweep.py --corpus diagnostic
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
    "diagnostic": ("diagnostic_benchmark", "diag_"),
    "nist": ("csf_benchmark", "csf_"),
    "hipaa": ("hipaa_benchmark", ""),
    "pf": ("pf_benchmark", ""),
}
TAUS = (0.05, 0.10, 0.15, 0.20)

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


def run_corpus(key, directory, prefix):
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

    # Inductive: controls only, matching the primary protocol.
    scorers = {"tfidf": build_tfidf_scorer(ctrl_texts),
               "bm25": build_bm25_scorer(ctrl_texts),
               "lsi": LSIIndex(ctrl_texts, [], n_components=100,
                               include_regs_in_fit=False).score}
    reg_list = [regs[r] for r in sorted(regs)]
    tools = AgentTools(scorers, controls, reg_list,
                       {r: sorted(gold.get(r, [])) for r in regs})

    rows = []
    for tau in TAUS:
        top1, gate, expanded = {}, 0, 0
        for name, cfg in VARIANTS.items():
            agent = ComplianceAgent(tools=tools, conf_thr=0.0, gap_thr=0.0,
                                    rel_gap_retry_threshold=tau, **cfg)
            hits, order_by_rid = [], {}
            for rid in sorted(regs):
                if rid not in gold:
                    continue
                trace = agent.solve(regs[rid], primary_backend="bm25")
                top = ctrl_ids[trace.decision.ranked[0].control_id]
                order_by_rid[rid] = top
                hits.append(1.0 if top in gold[rid] else 0.0)
                if name == "reform" and "reformulate" in trace.tools_used:
                    gate += 1
                    if tools.reformulate(regs[rid].text) != regs[rid].text:
                        expanded += 1
            top1[name] = (float(np.mean(hits)), order_by_rid)

        m_mean, m_top = top1["multi"]
        r_mean, r_top = top1["reform"]
        shared = sorted(set(m_top) & set(r_top))
        changed = sum(1 for r in shared if m_top[r] != r_top[r])
        rows.append({
            "tau_rel": tau, "n": len(shared),
            "multi_top1": round(m_mean, 4), "reform_top1": round(r_mean, 4),
            "difference": round(r_mean - m_mean, 4),
            "gate_fired": gate, "expanded": expanded,
            "top1_identity_changed": changed,
        })
        print(f"    tau={tau:.2f}  multi={m_mean:.4f} reform={r_mean:.4f} "
              f"delta={r_mean - m_mean:+.4f}  gate {gate}/{len(shared)} "
              f"expanded {expanded}  top1 moved {changed}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=sorted(CORPORA))
    args = ap.parse_args()

    print("EXPLORATORY. Primary protocol: one pass per requirement over the")
    print("full control corpus, LSI fitted on controls only.\n")

    keys = [args.corpus] if args.corpus else list(CORPORA)
    results = {}
    for key in keys:
        directory, prefix = CORPORA[key]
        if not os.path.isdir(os.path.join(HERE, directory)):
            print(f"  {key}: missing, skipped")
            continue
        print(f"  {key}:")
        results[key] = run_corpus(key, directory, prefix)

    spans = {k: round(max(r["difference"] for r in v)
                      - min(r["difference"] for r in v), 4)
             for k, v in results.items()}
    print("\nrange of the reform effect across tau settings:")
    for k, v in spans.items():
        print(f"  {k:11s} {v:+.4f}")

    record = {
        "status": "exploratory",
        "status_reason": "unplanned sensitivity sweep; not preregistered",
        "protocol": "one scoring pass per requirement over the full control "
                    "corpus, LSI fitted on control documents only",
        "supersedes": "results_v3/sensitivity/sensitivity.csv, which no "
                      "committed script produces and whose absolute level "
                      "could not be reconciled with the ablation table",
        "constant": "rel_gap_retry_threshold (tau_rel), the relative top-2 "
                    "margin below which reformulation is invoked",
        "taus": list(TAUS),
        "results": results,
        "effect_range_across_taus": spans,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "argv": sys.argv, "git": git_state(),
        "spec_hashes": {
            "sensitivity_sweep.py": sha256(os.path.abspath(__file__)),
            "raa_agent.py": sha256(os.path.join(HERE, "raa_agent.py"))},
        "environment": {"python": sys.version.split()[0],
                        "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__},
    }
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "sensitivity_tau.json")
    with open(out, "w", newline="\n") as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"\nrecord written: {os.path.relpath(out, HERE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

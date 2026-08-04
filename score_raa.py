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

LSI FIT, AND WHY IT IS A SEPARATE KNOB
--------------------------------------
run_variant fits LSI on controls plus the train and calibration requirement
texts, so the latent space depends on which requirements landed in the split.
A one-pass protocol has no train set, so the natural choice is controls only.

That means moving from the published holdout numbers to these ones changes
TWO things at once: the evaluation protocol and the LSI fitting regime.
Attributing the difference to protocol alone would be confounded. --lsi-fit
exposes the second knob so it can be varied while the protocol is held fixed:

    inductive     controls only (default; matches the shared protocol)
    transductive  controls plus every requirement text (matches score_all's
                  "lsi", and is the closest available analogue of the
                  published split-dependent fit)

Running both under one protocol isolates how much of the movement is
representation fitting rather than evaluation design.

INSTRUMENTATION
---------------
Four distinct things are counted, because they are not the same and an
earlier version of this script conflated the first with the rest:

    gate_fired        the ambiguity gate invoked the reformulation tool
    expanded          the tool actually produced a query different from the
                      original, so an expansion existed to retrieve with
    ranking_changed   the post-reformulation ranking differs from the
                      pre-reformulation one at any position
    top1_changed      the predicted top-1 control differs

Reporting only the first supports "the gate activated on nearly every
query". It does not support "reformulation was always on".

Usage:
    USE_TF=0 python score_raa.py
    USE_TF=0 python score_raa.py --lsi-fit transductive --out shared_raa_scores_transductive.csv
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


def score_corpus(key, directory, prefix, variants, lsi_fit):
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
        "lsi": (LSIIndex(ctrl_texts, [], n_components=100,
                         include_regs_in_fit=False).score
                if lsi_fit == "inductive" else
                LSIIndex(ctrl_texts, list(regs_df["text"]), n_components=100,
                         include_regs_in_fit=True).score),
    }

    reg_list = [regs[r] for r in sorted(regs)]
    gt_lists = {r: sorted(gold.get(r, [])) for r in regs}
    tools = AgentTools(scorers, controls, reg_list, gt_lists)

    rows, counters, stats = [], {}, {k: {} for k in variants}
    for name, cfg in variants.items():
        agent = ComplianceAgent(tools=tools, conf_thr=0.0, gap_thr=0.0,
                                rel_gap_retry_threshold=0.10, **cfg)
        # top1_changed and ranking_changed are contrasts between two arms, so
        # they cannot be counted inside a single arm's loop. They are computed
        # below. An earlier version carried a per-arm top1_changed here that
        # was initialised to zero and never incremented.
        counts = {"gate_fired": 0, "expanded": 0}
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
                counts["gate_fired"] += 1
                # Did the tool actually change the query text?
                expanded = tools.reformulate(regs[rid].text)
                if expanded != regs[rid].text:
                    counts["expanded"] += 1
            stats[name].setdefault(rid, {}).update(
                {"top1": m["predicted_top1"], "order": tuple(order)})
        n = sum(1 for r in regs if r in gold)
        counters[name] = dict(counts, n=n)
        mean = np.mean([r["top1"] for r in rows if r["method"] == name])
        print(f"    {name:16s} n={n:3d}  Top-1={mean:.4f}"
              f"  gate {counts['gate_fired']:3d}/{n}"
              f"  expanded {counts['expanded']:3d}/{n}", flush=True)

    # How many requirements did reformulation actually move? Gate invocation
    # and expansion both overcount this: an expansion can leave the ranking,
    # or at least the top-1, unchanged. ranking_changed is the widest of the
    # four counts and top1_changed the narrowest, and neither is the number
    # that drives Top-1: that is the count of requirements whose top-1
    # CORRECTNESS flips, which is smaller again and lives in the W/L/T
    # columns of the statistical records.
    def contrast(a, b):
        shared = sorted(set(stats[a]) & set(stats[b]))
        return {
            "n": len(shared),
            "top1_changed": sum(1 for r in shared
                                if stats[a][r]["top1"] != stats[b][r]["top1"]),
            "ranking_changed": sum(1 for r in shared
                                   if stats[a][r]["order"] != stats[b][r]["order"]),
            "differing_rids": [r for r in shared
                               if stats[a][r]["order"] != stats[b][r]["order"]],
        }

    if "multi" in stats and "reform" in stats:
        c = contrast("multi", "reform")
        counters["reform_vs_multi"] = {k: v for k, v in c.items()
                                       if k != "differing_rids"}
        # Kept under the old key so existing records and readers do not break.
        counters["top1_changed_reform_vs_multi"] = {
            "changed": c["top1_changed"], "n": c["n"]}
        print(f"    {'top-1 moved':16s} {c['top1_changed']:3d}/{c['n']}"
              f"   ranking moved {c['ranking_changed']:3d}/{c['n']}", flush=True)

    # The manuscript states that corroboration and verification are
    # decision-only and never reorder candidates. Checking that needs the
    # decomposition step isolated first, because it is a ranking tool and
    # sits between the two arms.
    #
    #   reform -> decomp     turns on decomposition alone
    #   decomp -> raa_full   turns on corroboration and verification alone
    #
    # Only the second contrast bears on the decision-only claim.
    if "reform" in stats and "decomp" in stats:
        c = contrast("reform", "decomp")
        counters["decomp_vs_reform"] = c
        print(f"    {'decomposition':16s} ranking moved "
              f"{c['ranking_changed']:3d}/{c['n']}"
              f"  rids={c['differing_rids']}", flush=True)
    if "decomp" in stats and "raa_full" in stats:
        c = contrast("decomp", "raa_full")
        counters["raa_full_vs_decomp"] = c
        print(f"    {'decision-only':16s} ranking moved "
              f"{c['ranking_changed']:3d}/{c['n']}"
              f"  rids={c['differing_rids']}", flush=True)
    return pd.DataFrame(rows), counters


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=sorted(CORPORA))
    # --out defaults per fitting regime, deliberately. It used to default to
    # shared_raa_scores.csv whatever --lsi-fit said, so running the
    # transductive arm without also passing --out silently overwrote the
    # primary inductive scores with transductive ones. Nothing downstream
    # notices: the file has the same columns and the same row count, and the
    # only symptom is that analyze_onepass.py reports identical numbers for
    # two arms that should differ.
    ap.add_argument("--out", default=None,
                    help="output CSV; defaults to shared_raa_scores.csv for "
                         "the inductive fit and "
                         "shared_raa_scores_transductive.csv for the "
                         "transductive one")
    ap.add_argument("--lsi-fit", choices=("inductive", "transductive"),
                    default="inductive",
                    help="inductive = controls only (shared protocol); "
                         "transductive = controls plus all requirement texts")
    args = ap.parse_args()
    if args.out is None:
        args.out = ("shared_raa_scores.csv" if args.lsi_fit == "inductive"
                    else "shared_raa_scores_transductive.csv")

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
        # +Decomp exists so that raa_full vs reform can be decomposed. Without
        # it, that contrast switches on decomposition, corroboration and
        # verification together, and decomposition is a RANKING tool. An
        # earlier version used the two-arm contrast to claim it had checked
        # whether the decision-only tools reorder. It had not: any reordering
        # it found is attributable to decomposition first.
        "decomp": dict(enable_multi=True, enable_reform=True,
                       enable_decompose=True, enable_crossref=False,
                       enable_verify=False),
        "raa_full": dict(enable_multi=True, enable_reform=True,
                         enable_decompose=True, enable_crossref=True,
                         enable_verify=True),
    }

    print("Full RAA on the shared population. Ranking metrics only;")
    print("decision metrics need calibration splits and are not produced.\n")

    # Sample the working-tree state BEFORE writing any output. Called after
    # the CSV is written, git_state() sees this script's own product and
    # reports dirty:true, which reads as "produced from uncommitted code"
    # when the code was in fact clean. The flag has to describe the tree that
    # produced the numbers, not the tree after they landed.
    git_at_start = git_state()

    keys = [args.corpus] if args.corpus else list(CORPORA)
    frames, all_counters = [], {}
    for key in keys:
        directory, prefix = CORPORA[key]
        if not os.path.isdir(os.path.join(HERE, directory)):
            print(f"  {key}: {directory} missing, skipped")
            continue
        f, c = score_corpus(key, directory, prefix, variants, args.lsi_fit)
        frames.append(f)
        all_counters[key] = c

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
        "lsi_fit": args.lsi_fit,
        "lsi_fit_note": "inductive = controls only. run_variant fits on "
                        "controls plus train and calibration requirements, so "
                        "comparing these numbers with the published holdout "
                        "figures varies protocol AND representation fitting "
                        "together; run both --lsi-fit settings to separate them",
        "reformulation_counters": all_counters,
        "primary_backend": "bm25",
        "rel_gap_retry_threshold": 0.10,
        "thresholds": "conf_thr=gap_thr=0.0; ranking is independent of them",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "argv": sys.argv, "git": git_at_start,
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
    stem = f"raa_shared_population_{args.lsi_fit}.json"
    with open(os.path.join(rd, stem), "w", newline="\n") as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"record written: results_v3/shared/{stem}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

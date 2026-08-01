#!/usr/bin/env python3
"""Export the shared first-stage candidate set, once, for every reranker.

Why this exists: the published comparison reranked the dual-encoder top-20
with a cross-encoder while the agent and lexical methods retrieved over the
full corpus. That is a valid end-to-end comparison of pipelines, but it
cannot attribute performance to the reranker independently of its candidate
generator. Exporting one candidate set that every reranker consumes fixes the
attribution question without discarding the end-to-end comparison.

The generator is the dual-encoder all-MiniLM-L6-v2 at k=20, matching the
existing cross-encoder protocol exactly so old and new numbers stay
comparable.

Also reports the candidate-set Recall@20 ceiling, which bounds what any
reranker can achieve and must be stated before any conditional result.

Usage:
    USE_TF=0 python first_stage.py                    # all corpora
    USE_TF=0 python first_stage.py hipaa_benchmark:   # one corpus
"""
import os
import sys

import numpy as np
import pandas as pd

MODEL = "all-MiniLM-L6-v2"
K = 20

CORPORA = [
    ("diagnostic_benchmark", "diag_"),
    ("csf_benchmark", "csf_"),
    ("hipaa_benchmark", ""),
    ("pf_benchmark", ""),
]


def load(directory, prefix):
    d = directory.rstrip("/\\")
    regs = pd.read_csv(f"{d}/{prefix}regs.csv")
    ctls = pd.read_csv(f"{d}/{prefix}controls.csv")
    maps = pd.read_csv(f"{d}/{prefix}mappings.csv")
    return regs, ctls, maps


def export(directory, prefix, model):
    regs, ctls, maps = load(directory, prefix)
    gold = {}
    for r, c in zip(maps["regulation_id"], maps["control_id"]):
        gold.setdefault(r, set()).add(c)

    ctrl_emb = model.encode(list(ctls["text"]), convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)
    req_emb = model.encode(list(regs["text"]), convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False)
    sims = req_emb @ ctrl_emb.T

    rows = []
    hit_any = 0
    gold_total = gold_found = 0
    scored = 0
    for i, rid in enumerate(regs["id"]):
        if rid not in gold:
            continue
        scored += 1
        order = np.argsort(-sims[i])[:K]
        for rank, j in enumerate(order, start=1):
            cid = int(ctls["id"].iloc[j])
            rows.append((rid, rank, cid, float(sims[i][j]),
                         int(cid in gold[rid])))
        found = len(gold[rid] & {int(ctls["id"].iloc[j]) for j in order})
        gold_total += len(gold[rid])
        gold_found += found
        hit_any += int(found > 0)

    out = pd.DataFrame(rows, columns=["rid", "rank", "control_id", "score",
                                      "is_gold"])
    path = f"{directory.rstrip('/')}/candidates_k{K}.csv"
    out.to_csv(path, index=False)

    req_recall = hit_any / scored if scored else 0.0
    link_recall = gold_found / gold_total if gold_total else 0.0
    print(f"  {directory:<22} n={scored:<4} "
          f"Recall@{K} (any gold) = {req_recall:.3f}   "
          f"link-level Recall@{K} = {link_recall:.3f}")
    return {"corpus": directory, "n": scored, "k": K,
            "requirement_recall_at_k": round(req_recall, 4),
            "link_recall_at_k": round(link_recall, 4),
            "unreachable_requirements": scored - hit_any}


def main():
    os.environ.setdefault("USE_TF", "0")
    from sentence_transformers import SentenceTransformer

    targets = []
    for arg in sys.argv[1:]:
        d, _, p = arg.partition(":")
        targets.append((d, p))
    targets = targets or CORPORA

    print(f"First-stage candidate export: {MODEL}, k={K}\n")
    print("CANDIDATE-SET CEILING (bounds every reranker):")
    model = SentenceTransformer(MODEL)

    summary = []
    for directory, prefix in targets:
        if not os.path.isdir(directory):
            print(f"  {directory}: not found, skipped")
            continue
        summary.append(export(directory, prefix, model))

    pd.DataFrame(summary).to_csv("first_stage_ceiling.csv", index=False)
    print("\nwrote first_stage_ceiling.csv")
    print("\nAny query whose gold control is absent from these candidates is a")
    print("candidate-generation miss, not evidence about open-world detection.")


if __name__ == "__main__":
    main()

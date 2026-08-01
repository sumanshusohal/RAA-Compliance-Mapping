#!/usr/bin/env python3
"""Score every method on the identical evaluation population.

WHY THIS EXISTS
---------------
The published protocol used repeated stratified holdouts, so a requirement
appears in the test split an uneven number of times (on HIPAA, between 2 and
12 across 30 seeds). Pooling those rows weights some requirements six times
more than others. The local LLM, by contrast, was evaluated exactly once per
requirement. Comparing the two directly mixes estimands, and any resulting
"method A beats method B" claim is an artifact of that mismatch.

The fix rests on an observation the holdout design obscures: for frozen
retrieval methods, RANKING requires no training. TF-IDF, BM25, LSI, the
dual-encoder, the cross-encoder, and an LLM reranker all produce the same
ranking for a requirement regardless of any split. So every requirement can
be scored exactly once, over the full control corpus, giving one value per
requirement per method on a common population.

Splits are still needed for DECISION metrics, because the accept/abstain
threshold is calibrated. That is handled separately in fold_protocol.py.

Output: one row per (corpus, method, requirement) with rank-based metrics,
which is the shared table every later comparison should be computed from.

Usage:
    USE_TF=0 python score_all.py
    USE_TF=0 python score_all.py --corpus hipaa_benchmark
"""
import argparse
import os
import sys

os.environ.setdefault("USE_TF", "0")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from raa_agent import (  # noqa: E402
    LSIIndex, build_bm25_scorer, build_crossencoder_reranker,
    build_semantic_scorer, build_tfidf_scorer,
)

CORPORA = [
    ("diagnostic_benchmark", "diag_"),
    ("csf_benchmark", "csf_"),
    ("hipaa_benchmark", ""),
    ("pf_benchmark", ""),
]

TOPK = 5
RRF_K = 60


def load(directory, prefix):
    d = directory.rstrip("/\\")
    regs = pd.read_csv(f"{d}/{prefix}regs.csv")
    ctls = pd.read_csv(f"{d}/{prefix}controls.csv")
    maps = pd.read_csv(f"{d}/{prefix}mappings.csv")
    gold = {}
    for r, c in zip(maps["regulation_id"], maps["control_id"]):
        gold.setdefault(r, set()).add(c)
    return regs, ctls, gold


def metrics(order, gold_set, k=TOPK):
    """Rank metrics for one requirement from a ranked control-id list.

    Records the predicted top-1 control id, not just its correctness, so
    complementarity between methods (which requirements each uniquely
    solves) is reproducible from this file alone.
    """
    top1 = 1.0 if order[0] in gold_set else 0.0
    rr = 0.0
    for rank, cid in enumerate(order[:k], start=1):
        if cid in gold_set:
            rr = 1.0 / rank
            break
    hits = sum(1 for cid in order[:k] if cid in gold_set)
    return {
        "top1": top1,
        "predicted_top1": order[0],
        "rr@5": rr,
        "recall@5": hits / len(gold_set) if gold_set else 0.0,
        "precision@5": hits / k,
        "first_gold_rank": next(
            (i + 1 for i, cid in enumerate(order) if cid in gold_set), None),
    }


def rrf_fuse(score_arrays, ctrl_ids):
    """Reciprocal rank fusion, matching the agent's multi-backend step."""
    fused = np.zeros(len(ctrl_ids), dtype=float)
    for scores in score_arrays:
        order = np.argsort(-scores)
        ranks = np.empty(len(scores), dtype=float)
        ranks[order] = np.arange(1, len(scores) + 1)
        fused += 1.0 / (RRF_K + ranks)
    return fused


def score_corpus(directory, prefix):
    regs, ctls, gold = load(directory, prefix)
    ctrl_texts = list(ctls["text"])
    ctrl_ids = list(ctls["id"])
    reg_texts = list(regs["text"])

    print(f"  building scorers ({len(ctrl_texts)} controls)...", flush=True)
    # LSI is fitted TWICE on purpose.
    #
    # raa_agent's LSIIndex defaults to include_regs_in_fit=True, so the latent
    # space is fitted on controls PLUS the evaluation queries. That is
    # label-free but transductive: it uses the whole unlabeled query
    # collection at fit time, and it is what the published results did.
    # 'lsi_inductive' refits on control documents only, so every requirement
    # is genuinely unseen. Both are reported so the transductive effect is a
    # measured sensitivity rather than a silent property.
    lsi_transductive = LSIIndex(ctrl_texts, reg_texts, n_components=100,
                                include_regs_in_fit=True).score
    lsi_inductive = LSIIndex(ctrl_texts, [], n_components=100,
                             include_regs_in_fit=False).score
    scorers = {
        "tfidf": build_tfidf_scorer(ctrl_texts),
        "bm25": build_bm25_scorer(ctrl_texts),
        "lsi": lsi_transductive,
        "lsi_inductive": lsi_inductive,
        "semantic": build_semantic_scorer(ctrl_texts),
        "reranker": build_crossencoder_reranker(ctrl_texts),
    }

    rows = []
    for rid, text in zip(regs["id"], regs["text"]):
        if rid not in gold:
            continue
        raw = {}
        for name, fn in scorers.items():
            s = np.asarray(fn(text), dtype=float)
            raw[name] = s
            order = [ctrl_ids[i] for i in np.argsort(-s)]
            m = metrics(order, gold[rid])
            m.update({"corpus": directory, "method": name, "rid": rid,
                      "n_gold": len(gold[rid])})
            rows.append(m)

        # Multi-backend fusion over the three lexical scorers. This is the
        # agent's ranking-shaping 'multi' step, NOT full RAA: RAA adds
        # conditional reformulation and decomposition on top, which can
        # reorder. Label it accordingly and never call it RAA.
        for label, lsi_key in (("rrf_lexical", "lsi"),
                               ("rrf_lexical_inductive", "lsi_inductive")):
            fused = rrf_fuse([raw["tfidf"], raw["bm25"], raw[lsi_key]],
                             ctrl_ids)
            order = [ctrl_ids[i] for i in np.argsort(-fused)]
            m = metrics(order, gold[rid])
            m.update({"corpus": directory, "method": label, "rid": rid,
                      "n_gold": len(gold[rid])})
            rows.append(m)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--out", default="shared_ranking_scores.csv")
    args = ap.parse_args()

    targets = ([(args.corpus, args.prefix)] if args.corpus else CORPORA)
    frames = []
    for directory, prefix in targets:
        if not os.path.isdir(directory):
            print(f"  {directory}: missing, skipped")
            continue
        print(f"=== {directory} ===", flush=True)
        frames.append(score_corpus(directory, prefix))

    if not frames:
        return 1
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(out)} rows)")

    piv = out.pivot_table(index="corpus", columns="method", values="top1",
                          aggfunc="mean")
    piv.index = [i.replace("_benchmark", "") for i in piv.index]

    # The two protocols must be read separately. End-to-end methods rank the
    # whole control corpus; conditional rerankers see only the dual-encoder
    # top-20 and are bounded by that candidate ceiling. Comparing across the
    # two sections is a pipeline-attribution error.
    e2e = ["tfidf", "bm25", "lsi", "lsi_inductive", "rrf_lexical",
           "rrf_lexical_inductive", "semantic"]
    cond = ["reranker"]

    print("\nEND-TO-END (ranks the full control corpus, no ceiling):")
    print(piv[[c for c in e2e if c in piv.columns]].round(3).to_string())
    print("\nCONDITIONAL (reranks dual-encoder top-20, bounded by its "
          "Recall@20 ceiling):")
    print(piv[[c for c in cond if c in piv.columns]].round(3).to_string())
    print("\nThe local LLM belongs in the CONDITIONAL section: it also")
    print("reranks the top-20 and cannot exceed the same ceiling.")

    if {"lsi", "lsi_inductive"} <= set(piv.columns):
        print("\nTransductive-LSI sensitivity (published fit minus "
              "controls-only fit):")
        delta = (piv["lsi"] - piv["lsi_inductive"]).round(3)
        print(delta.to_string())
    print("\nEvery cell is a macro mean over the SAME requirements.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

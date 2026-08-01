#!/usr/bin/env python3
"""Run the open-weight reranker across every corpus, loading the model once.

Free: no API key, no account, no per-request cost. This is the reproducible
half of the LLM comparison, and it is the half a reviewer can actually rerun.

Usage:
    USE_TF=0 python run_local_all.py
    USE_TF=0 python run_local_all.py --model Qwen/Qwen2.5-7B-Instruct
"""
import argparse
import os
import sys
import time

os.environ.setdefault("USE_TF", "0")

import pandas as pd  # noqa: E402

from llm_backend import build_prompt, load_candidates, parse_response  # noqa: E402
from local_backend import _Msg, generate, load_model  # noqa: E402

CORPORA = [
    ("diagnostic_benchmark", "diag_"),
    ("csf_benchmark", "csf_"),
    ("hipaa_benchmark", ""),
    ("pf_benchmark", ""),
]


def run_corpus(corpus, prefix, tok, model, model_name, run_id, limit=None):
    reg_text, per_req, _ = load_candidates(corpus, prefix)
    rids = sorted(per_req)
    if limit:
        rids = rids[:limit]

    rows = []
    t0 = time.perf_counter()
    for i, rid in enumerate(rids, start=1):
        prompt = build_prompt(reg_text[rid], per_req[rid])
        started = time.perf_counter()
        body, n_in, n_out, stop = generate(tok, model, prompt)
        latency = time.perf_counter() - started

        availability, payload = parse_response(
            _Msg(body, stop, model_name), len(per_req[rid]))
        top = None
        if availability == "valid" and payload["ranking"]:
            top = per_req[rid][payload["ranking"][0] - 1][0]

        rows.append({
            "rid": rid, "availability": availability, "stop_reason": stop,
            "top1_control": top,
            "abstain": payload["abstain"] if payload else None,
            "confidence": payload["confidence"] if payload else None,
            "input_tokens": n_in, "output_tokens": n_out,
            "latency_s": round(latency, 4), "cached": False,
            "model": model_name, "effort": "n/a-local", "run_id": run_id,
        })
        if i % 20 == 0 or i == len(rids):
            el = time.perf_counter() - t0
            rate = el / i
            print(f"    {corpus} {i}/{len(rids)}  {el:.0f}s elapsed, "
                  f"{rate:.1f}s/req, ~{rate * (len(rids) - i):.0f}s left",
                  flush=True)

    out = pd.DataFrame(rows)
    path = f"{corpus.rstrip('/')}/local_rerank_run{run_id}.csv"
    out.to_csv(path, index=False)
    return out, path


def score(out, corpus, prefix):
    """Top-1 under both accountings: among valid, and counting invalid as
    failure. The gap between them is the availability effect."""
    maps = pd.read_csv(f"{corpus.rstrip('/')}/{prefix}mappings.csv")
    gold = {}
    for a, b in zip(maps["regulation_id"], maps["control_id"]):
        gold.setdefault(a, set()).add(b)

    valid = out[out["availability"] == "valid"]
    hits = sum(1 for _, r in valid.iterrows()
               if pd.notna(r["top1_control"])
               and r["top1_control"] in gold.get(r["rid"], set()))
    n_all, n_valid = len(out), len(valid)
    return {
        "corpus": corpus,
        "n": n_all,
        "valid": n_valid,
        "availability_rate": round(n_valid / n_all, 4) if n_all else 0.0,
        "top1_among_valid": round(hits / n_valid, 4) if n_valid else 0.0,
        "top1_invalid_as_failure": round(hits / n_all, 4) if n_all else 0.0,
        "mean_latency_s": round(out["latency_s"].mean(), 3),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--run-id", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    print(f"loading {args.model} (4bit={not args.no_4bit})", flush=True)
    tok, model = load_model(args.model, not args.no_4bit)
    print(f"loaded on {next(model.parameters()).device}\n", flush=True)

    summary = []
    for corpus, prefix in CORPORA:
        if not os.path.isdir(corpus):
            print(f"  {corpus}: missing, skipped", flush=True)
            continue
        print(f"  === {corpus} ===", flush=True)
        out, path = run_corpus(corpus, prefix, tok, model, args.model,
                               args.run_id, args.limit)
        s = score(out, corpus, prefix)
        summary.append(s)
        print(f"    wrote {path}")
        print(f"    availability {s['availability_rate']:.3f}  "
              f"Top-1 valid {s['top1_among_valid']:.3f}  "
              f"Top-1 strict {s['top1_invalid_as_failure']:.3f}\n", flush=True)

    df = pd.DataFrame(summary)
    df.to_csv("local_rerank_summary.csv", index=False)
    print("=" * 68)
    print(df.to_string(index=False))
    print("=" * 68)
    print("\nThe gap between top1_among_valid and top1_invalid_as_failure is")
    print("the availability effect. Reporting only the former overstates the")
    print("backend by exactly that margin.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

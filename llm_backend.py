#!/usr/bin/env python3
"""LLM reranking backends for the conditional comparison.

The LLM is a COMPARATOR here, not part of RAA. RAA stays deterministic and
LLM-free; this module exists to answer two questions the study cannot answer
without it: how a 2026-standard reranker performs on the same candidate sets,
and what run-to-run stability actually costs.

Design decisions that are load-bearing:

  * No sampling parameters. temperature/top_p/top_k are not available on
    current frontier models, so there is no determinism knob to pin. That is
    precisely why stability is measured rather than assumed.
  * Candidate-constrained. The model ranks a supplied candidate list and may
    abstain. Every returned id is validated against that list; an id outside
    it is an INVALID response, not a wrong answer.
  * No fallback in the primary treatment. A refusal is a real operational
    property of this model on security-control text, so it is counted and
    reported rather than silently re-served by a second model. Server-side
    fallback is also rejected by the Batches API.
  * stop_reason is recorded on every call. Truncation and refusal are
    availability failures, not retrieval failures, and must not be folded
    into misses.

Credentials come from the environment (ANTHROPIC_API_KEY). Never pass a key
as an argument, a flag, or a literal.

Usage:
    python llm_backend.py --corpus hipaa_benchmark --dry-run
    python llm_backend.py --corpus hipaa_benchmark --limit 10
"""
import argparse
import hashlib
import json
import os
import sys
import time

import pandas as pd

MODEL = "claude-opus-5"
MAX_TOKENS = 8192
DEFAULT_EFFORT = "high"      # frozen after the dev-split sweep
CACHE_DIR = "llm_cache"
PARSER_VERSION = "1.0.0"

# Spend guards. Exceeded means abort, not warn.
MAX_REQUESTS = 5000
MAX_TOTAL_TOKENS = 20_000_000

SYSTEM_PROMPT = (
    "You are assisting a compliance mapping study. Given one regulatory "
    "requirement and a numbered list of candidate security controls, rank the "
    "candidates by how well each one implements the requirement.\n\n"
    "Rules:\n"
    "- Choose only from the supplied candidate numbers. Never invent an id.\n"
    "- Rank at most 5 candidates, best first.\n"
    "- If no candidate plausibly implements the requirement, return an empty "
    "ranking and set abstain to true.\n"
    "- Judge on whether the control would satisfy the requirement, not on "
    "shared wording."
)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "ranking": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Candidate numbers, best first, at most 5.",
        },
        "abstain": {
            "type": "boolean",
            "description": "True when no candidate implements the requirement.",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence in the top choice, 0.0 to 1.0.",
        },
    },
    "required": ["ranking", "abstain", "confidence"],
    "additionalProperties": False,
}


def build_prompt(requirement, candidates):
    """Render one query. Candidate order is part of the cache key."""
    lines = [f"REQUIREMENT:\n{requirement}\n", "CANDIDATE CONTROLS:"]
    for n, (_, text) in enumerate(candidates, start=1):
        lines.append(f"{n}. {text}")
    lines.append("\nRank the candidates that implement this requirement.")
    return "\n".join(lines)


def cache_key(prompt, effort):
    """Hash the COMPLETE request-shaping surface.

    Anything omitted here would silently serve a stale response after a
    prompt, schema, or parameter change.
    """
    payload = json.dumps({
        "model": MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "schema": RESPONSE_SCHEMA,
        "max_tokens": MAX_TOKENS,
        "effort": effort,
        "thinking": "default-on",
        "parser_version": PARSER_VERSION,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Budget:
    """Hard caps. Exceeding one aborts the run."""

    def __init__(self, max_requests=MAX_REQUESTS, max_tokens=MAX_TOTAL_TOKENS):
        self.max_requests = max_requests
        self.max_tokens = max_tokens
        self.requests = 0
        self.tokens = 0

    def charge(self, tokens):
        self.requests += 1
        self.tokens += tokens
        if self.requests > self.max_requests:
            raise RuntimeError(
                f"request cap exceeded ({self.requests} > {self.max_requests})")
        if self.tokens > self.max_tokens:
            raise RuntimeError(
                f"token cap exceeded ({self.tokens} > {self.max_tokens})")


def _cache_path(key):
    return os.path.join(CACHE_DIR, f"{key}.json")


def load_cached(key):
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_cached(key, record):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_cache_path(key), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)


def parse_response(message, n_candidates):
    """Turn an API message into an availability status plus a parsed result.

    Returns (availability, payload). availability is 'valid' or one of the
    invalid subtypes; payload is None when invalid.
    """
    stop = getattr(message, "stop_reason", None)
    if stop == "refusal":
        return "refusal", None
    if stop == "max_tokens":
        return "truncation", None

    text = None
    for block in message.content:
        if getattr(block, "type", None) == "text":
            text = block.text
            break
    if not text:
        return "parse_failure", None

    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return "parse_failure", None

    ranking = data.get("ranking")
    if not isinstance(ranking, list):
        return "parse_failure", None

    # Two distinct failures, kept apart because they mean different things:
    # a non-integer entry is a schema violation (the model did not follow the
    # output contract), whereas an integer outside the candidate range is a
    # hallucinated id. Collapsing them would hide which one a backend suffers
    # from. Silently dropping either would understate the failure rate.
    for r in ranking:
        if isinstance(r, bool) or not isinstance(r, int):
            return "parse_failure", None
        if not (1 <= r <= n_candidates):
            return "out_of_set_id", None

    return "valid", {
        "ranking": ranking[:5],
        "abstain": bool(data.get("abstain", False)),
        "confidence": float(data.get("confidence", 0.0)),
    }


def rerank_one(client, requirement, candidates, effort, budget,
               bypass_cache=False, run_id=0):
    """Rerank one requirement. Returns a record ready for the per-query CSV."""
    prompt = build_prompt(requirement, candidates)
    key = cache_key(prompt, effort)

    if not bypass_cache:
        hit = load_cached(key)
        if hit is not None:
            hit["cached"] = True
            return hit

    started = time.perf_counter()
    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        output_config={"format": {"type": "json_schema",
                                  "schema": RESPONSE_SCHEMA},
                       "effort": effort},
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.perf_counter() - started

    usage = getattr(message, "usage", None)
    in_tok = getattr(usage, "input_tokens", 0) or 0
    out_tok = getattr(usage, "output_tokens", 0) or 0
    budget.charge(in_tok + out_tok)

    availability, payload = parse_response(message, len(candidates))
    record = {
        "availability": availability,
        "payload": payload,
        "stop_reason": getattr(message, "stop_reason", None),
        "model": getattr(message, "model", MODEL),
        "effort": effort,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "latency_s": round(latency, 4),
        "run_id": run_id,
        "cached": False,
    }
    if not bypass_cache:
        save_cached(key, record)
    return record


def submit_batch(client, reg_text, per_req, effort, rids=None):
    """Submit the main accuracy pass as a batch at 50% of standard pricing.

    Batches are the right tool for a 30-seed ablation: nothing here is
    latency-sensitive. Latency is measured separately with synchronous calls
    on a fixed sample, because batch turnaround is asynchronous (usually
    under an hour, permitted up to 24) and is not comparable to per-query
    latency.

    Identical (requirement, candidate list) inputs are deduplicated first;
    overlapping seed splits produce many repeats and each would otherwise be
    billed separately. Note that server-side fallback is rejected by the
    Batches API, which matches this study's design: refusals are counted,
    not re-served.
    """
    from anthropic.types.message_create_params import (
        MessageCreateParamsNonStreaming)
    from anthropic.types.messages.batch_create_params import Request

    rids = sorted(per_req) if rids is None else rids
    seen, requests, mapping = {}, [], {}
    for rid in rids:
        prompt = build_prompt(reg_text[rid], per_req[rid])
        key = cache_key(prompt, effort)
        if key in seen:
            mapping[rid] = seen[key]      # reuse another rid's result
            continue
        custom_id = f"rid-{rid}"
        seen[key] = custom_id
        mapping[rid] = custom_id
        requests.append(Request(
            custom_id=custom_id,
            params=MessageCreateParamsNonStreaming(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                output_config={"format": {"type": "json_schema",
                                          "schema": RESPONSE_SCHEMA},
                               "effort": effort},
                messages=[{"role": "user", "content": prompt}],
            ),
        ))

    print(f"  {len(rids)} requirements -> {len(requests)} unique requests "
          f"({len(rids) - len(requests)} deduplicated)")
    batch = client.messages.batches.create(requests=requests)
    print(f"  batch {batch.id} submitted, status={batch.processing_status}")
    return batch.id, mapping


def poll_batch(client, batch_id, interval=30, timeout=86400):
    """Wait for a batch to finish. Returns the terminal batch object."""
    waited = 0
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            print(f"  batch ended: {batch.request_counts}")
            return batch
        if waited >= timeout:
            raise TimeoutError(f"batch {batch_id} still "
                               f"{batch.processing_status} after {waited}s")
        print(f"  {batch.processing_status}, waited {waited}s", flush=True)
        time.sleep(interval)
        waited += interval


def collect_batch(client, batch_id, per_req, mapping):
    """Turn batch results into per-query records under the same taxonomy.

    Results arrive in arbitrary order, so they are keyed by custom_id and
    never by position.
    """
    by_id = {}
    for result in client.messages.batches.results(batch_id):
        by_id[result.custom_id] = result

    rows = []
    for rid, custom_id in mapping.items():
        result = by_id.get(custom_id)
        if result is None or result.result.type != "succeeded":
            rows.append({"rid": rid, "availability": "api_failure",
                         "stop_reason": None, "top1_control": None,
                         "abstain": None, "confidence": None,
                         "input_tokens": 0, "output_tokens": 0})
            continue
        message = result.result.message
        availability, payload = parse_response(message, len(per_req[rid]))
        top = None
        if availability == "valid" and payload["ranking"]:
            top = per_req[rid][payload["ranking"][0] - 1][0]
        usage = getattr(message, "usage", None)
        rows.append({
            "rid": rid, "availability": availability,
            "stop_reason": getattr(message, "stop_reason", None),
            "top1_control": top,
            "abstain": payload["abstain"] if payload else None,
            "confidence": payload["confidence"] if payload else None,
            "input_tokens": getattr(usage, "input_tokens", 0) or 0,
            "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        })
    return rows


def load_candidates(corpus_dir, prefix="", k=20):
    """Load the shared first-stage candidate set and corpus texts."""
    d = corpus_dir.rstrip("/\\")
    cands = pd.read_csv(f"{d}/candidates_k{k}.csv")
    regs = pd.read_csv(f"{d}/{prefix}regs.csv")
    ctls = pd.read_csv(f"{d}/{prefix}controls.csv")
    reg_text = dict(zip(regs["id"], regs["text"]))
    ctl_text = dict(zip(ctls["id"], ctls["text"]))

    per_req = {}
    for rid, grp in cands.groupby("rid"):
        grp = grp.sort_values("rank")
        per_req[rid] = [(int(c), ctl_text[int(c)]) for c in grp["control_id"]]
    return reg_text, per_req, cands


def estimate_cost(client, reg_text, per_req, effort, limit=None):
    """Dry run: count tokens over the planned request set and spend nothing."""
    rids = sorted(per_req)
    if limit:
        rids = rids[:limit]

    # Deduplicate identical (requirement, candidate list) inputs. Overlapping
    # seed splits produce many repeats and each would otherwise be billed.
    seen, unique = set(), []
    for rid in rids:
        prompt = build_prompt(reg_text[rid], per_req[rid])
        key = cache_key(prompt, effort)
        if key in seen:
            continue
        seen.add(key)
        unique.append((rid, prompt))

    total_in = 0
    for _, prompt in unique:
        count = client.messages.count_tokens(
            model=MODEL,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        total_in += count.input_tokens

    return {"requests_planned": len(rids), "unique_requests": len(unique),
            "deduplicated": len(rids) - len(unique), "input_tokens": total_in}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--effort", default=DEFAULT_EFFORT,
                    choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="count tokens and exit without spending")
    ap.add_argument("--bypass-cache", action="store_true",
                    help="for the stability trial only")
    ap.add_argument("--batch", action="store_true",
                    help="submit the pass as a batch at 50%% cost")
    ap.add_argument("--run-id", type=int, default=0)
    args = ap.parse_args()

    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set in the environment.\n"
              "Set it there; never pass a key as a flag or a literal.",
              file=sys.stderr)
        return 2
    client = anthropic.Anthropic()

    reg_text, per_req, _ = load_candidates(args.corpus, args.prefix)
    print(f"{args.corpus}: {len(per_req)} requirements with candidates, "
          f"model={MODEL}, effort={args.effort}")

    if args.dry_run:
        est = estimate_cost(client, reg_text, per_req, args.effort, args.limit)
        print("\nDRY RUN, nothing spent:")
        for k, v in est.items():
            print(f"  {k:<20} {v:,}")
        print(f"\n  input tokens are exact; output is capped at "
              f"{MAX_TOKENS:,}/request but typically far lower.")
        return 0

    rids = sorted(per_req)
    if args.limit:
        rids = rids[:args.limit]

    if args.batch:
        batch_id, mapping = submit_batch(client, reg_text, per_req,
                                         args.effort, rids)
        poll_batch(client, batch_id)
        rows = collect_batch(client, batch_id, per_req, mapping)
        out = pd.DataFrame(rows)
        out["model"] = MODEL
        out["effort"] = args.effort
        out["run_id"] = args.run_id
        out["batch_id"] = batch_id
        path = f"{args.corpus.rstrip('/')}/llm_batch_run{args.run_id}.csv"
        out.to_csv(path, index=False)
        print(f"\nwrote {path}")
        print(f"availability: {out['availability'].value_counts().to_dict()}")
        return 0

    budget = Budget()

    rows = []
    for i, rid in enumerate(rids, start=1):
        rec = rerank_one(client, reg_text[rid], per_req[rid], args.effort,
                         budget, args.bypass_cache, args.run_id)
        top = None
        if rec["availability"] == "valid" and rec["payload"]["ranking"]:
            top = per_req[rid][rec["payload"]["ranking"][0] - 1][0]
        rows.append({
            "rid": rid, "availability": rec["availability"],
            "stop_reason": rec["stop_reason"], "top1_control": top,
            "abstain": (rec["payload"]["abstain"]
                        if rec["availability"] == "valid" else None),
            "confidence": (rec["payload"]["confidence"]
                           if rec["availability"] == "valid" else None),
            "input_tokens": rec["input_tokens"],
            "output_tokens": rec["output_tokens"],
            "latency_s": rec["latency_s"], "cached": rec["cached"],
            "model": rec["model"], "effort": rec["effort"],
            "run_id": args.run_id,
        })
        if i % 10 == 0:
            print(f"  {i}/{len(rids)}  requests={budget.requests} "
                  f"tokens={budget.tokens:,}")

    out = pd.DataFrame(rows)
    path = f"{args.corpus.rstrip('/')}/llm_rerank_run{args.run_id}.csv"
    out.to_csv(path, index=False)

    avail = out["availability"].value_counts().to_dict()
    print(f"\nwrote {path}")
    print(f"availability: {avail}")
    print(f"spent: {budget.requests} requests, {budget.tokens:,} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())

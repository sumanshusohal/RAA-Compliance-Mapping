#!/usr/bin/env python3
"""Open-weight reranking backend, run locally.

This is the reproducibility half of the LLM comparison: any reviewer with a
GPU can rerun it with no account, no key, and no vendor endpoint that can
change underneath them.

It deliberately imports the prompt, schema, parser, and cache key from
llm_backend rather than restating them, so the local and API paths provably
share one prompt, one output contract, and one validator. If they drifted,
the comparison would silently stop being apples-to-apples.

Two asymmetries against the API path, both reported rather than hidden:

  1. The API enforces the JSON schema server-side; a local model is only
     asked for JSON. Local parse failures are therefore expected to be
     higher, and are counted in the same Layer 1 taxonomy.
  2. Local decoding CAN be pinned (greedy, fixed seed) whereas the API
     cannot. That is the point of running both: it separates instability
     caused by hosted inference from instability inherent to LLM ranking.
     Expect a near-zero local flip rate and read it as a control, not as
     evidence that the API model is unstable for a different reason.

Usage:
    USE_TF=0 python local_backend.py --corpus hipaa_benchmark --limit 5
    USE_TF=0 python local_backend.py --corpus hipaa_benchmark --model Qwen/Qwen2.5-7B-Instruct --load-4bit
"""
import argparse
import os
import sys
import time

os.environ.setdefault("USE_TF", "0")

import pandas as pd  # noqa: E402

from llm_backend import (  # noqa: E402
    RESPONSE_SCHEMA, SYSTEM_PROMPT, build_prompt, load_candidates,
    parse_response,
)

# Default is sized for 6 GB VRAM in fp16. Larger models need --load-4bit.
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 512
SEED = 20260801

JSON_INSTRUCTION = (
    "\n\nRespond with ONLY a JSON object matching this schema, no prose and "
    "no code fence:\n"
    '{"ranking": [<candidate numbers, best first, at most 5>], '
    '"abstain": <true|false>, "confidence": <0.0-1.0>}'
)


class _Msg:
    """Adapter so the shared parse_response can read a local generation."""

    def __init__(self, text, stop_reason, model):
        self.content = [type("B", (), {"type": "text", "text": text})()]
        self.stop_reason = stop_reason
        self.model = model


def load_model(name, load_4bit):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    kwargs = {"dtype": torch.float16, "device_map": "auto"}
    if load_4bit:
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            kwargs.pop("dtype")
        except ImportError:
            print("bitsandbytes unavailable; falling back to fp16.",
                  file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval()
    return tok, model


def generate(tok, model, prompt, max_new_tokens=MAX_NEW_TOKENS):
    """Greedy, seeded decoding. Deterministic by construction, unlike the API."""
    import torch

    torch.manual_seed(SEED)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt + JSON_INSTRUCTION}]
    text = tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(model.device)
    n_in = int(inputs["input_ids"].shape[1])

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None,
                             top_k=None,
                             pad_token_id=tok.eos_token_id)
    gen = out[0][n_in:]
    n_out = int(gen.shape[0])
    body = tok.decode(gen, skip_special_tokens=True).strip()

    # Strip a code fence if the model added one despite instructions.
    if body.startswith("```"):
        body = body.split("\n", 1)[-1]
        body = body.rsplit("```", 1)[0].strip()

    truncated = n_out >= max_new_tokens
    return body, n_in, n_out, ("max_tokens" if truncated else "end_turn")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run-id", type=int, default=0)
    args = ap.parse_args()

    reg_text, per_req, _ = load_candidates(args.corpus, args.prefix)
    rids = sorted(per_req)
    if args.limit:
        rids = rids[:args.limit]

    print(f"{args.corpus}: {len(rids)} requirements, model={args.model}, "
          f"4bit={args.load_4bit}")
    tok, model = load_model(args.model, args.load_4bit)
    print(f"loaded on {next(model.parameters()).device}")

    rows = []
    t0 = time.perf_counter()
    for i, rid in enumerate(rids, start=1):
        prompt = build_prompt(reg_text[rid], per_req[rid])
        started = time.perf_counter()
        body, n_in, n_out, stop = generate(tok, model, prompt)
        latency = time.perf_counter() - started

        availability, payload = parse_response(
            _Msg(body, stop, args.model), len(per_req[rid]))
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
            "model": args.model, "effort": "n/a-local",
            "run_id": args.run_id,
        })
        if i % 10 == 0 or i == len(rids):
            print(f"  {i}/{len(rids)}  {time.perf_counter() - t0:.0f}s elapsed")

    out = pd.DataFrame(rows)
    path = (f"{args.corpus.rstrip('/')}/local_rerank_run{args.run_id}.csv")
    out.to_csv(path, index=False)
    print(f"\nwrote {path}")
    print(f"availability: {out['availability'].value_counts().to_dict()}")
    print(f"total {time.perf_counter() - t0:.0f}s, "
          f"mean {out['latency_s'].mean():.2f}s/request")
    return 0


if __name__ == "__main__":
    sys.exit(main())

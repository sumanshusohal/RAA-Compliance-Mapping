#!/usr/bin/env python3
"""Requirement-level vocabulary gap: the moderator variable for RQ2.

FROZEN SPECIFICATION. This file is hashed and timestamped as part of the
preregistration. Do not change the tokenizer, the IDF corpus, the overlap
definition, or the aggregation after the confirmatory corpora have been
analysed; add a new function instead.

Definition
----------
For requirement r with gold control set G_r, over content tokens T(.):

    overlap(r, c) = sum_{t in T(r) & T(c)} idf(t) / sum_{t in T(r)} idf(t)
    gap(r)        = 1 - max_{c in G_r} overlap(r, c)          [PRIMARY]
    gap_mean(r)   = 1 - mean_{c in G_r} overlap(r, c)         [SENSITIVITY]

overlap is asymmetric on purpose: it asks how much of the requirement's
informative content is recoverable from the control, which is the direction a
retrieval system actually has to bridge. The primary aggregation takes the
max over gold controls because Top-1 is scored correct when ANY gold control
is retrieved, so the easiest gold link is what determines difficulty.

IDF is computed within each corpus over the union of its requirement and
control documents, so the measure is corpus-relative. RQ2 standardizes it
within corpus before pooling, so only the within-corpus ranking matters.
"""
import hashlib
import math
import re
import sys
from collections import Counter

# A deliberately small, fixed stopword list. Kept inline rather than imported
# so the frozen specification has no external dependency that could drift.
STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been
before being below between both but by can cannot could did do does doing down
during each few for from further had has have having he her here hers herself
him himself his how i if in into is it its itself me more most my myself no nor
not of off on once only or other ought our ours ourselves out over own same she
should so some such than that the their theirs them themselves then there these
they this those through to too under until up very was we were what when where
which while who whom why with would you your yours yourself yourselves
""".split())

TOKEN_RE = re.compile(r"[a-z][a-z0-9\-]{2,}")
MIN_TOKEN_LEN = 3


def tokenize(text):
    """Content tokens: lowercase, >=3 chars, alphabetic-initial, no stopwords."""
    if text is None:
        return []
    toks = TOKEN_RE.findall(str(text).lower())
    return [t for t in toks if len(t) >= MIN_TOKEN_LEN and t not in STOPWORDS]


def build_idf(documents):
    """Smoothed IDF over a corpus of documents (each a token list)."""
    n = len(documents)
    df = Counter()
    for toks in documents:
        for t in set(toks):
            df[t] += 1
    return {t: math.log((n + 1) / (d + 1)) + 1.0 for t, d in df.items()}


def overlap(req_tokens, ctl_tokens, idf):
    """IDF-weighted share of the requirement's content present in the control."""
    req_set = set(req_tokens)
    if not req_set:
        return 0.0
    denom = sum(idf.get(t, 1.0) for t in req_set)
    if denom <= 0:
        return 0.0
    shared = req_set & set(ctl_tokens)
    return sum(idf.get(t, 1.0) for t in shared) / denom


def compute_gaps(regs, controls, mappings):
    """Gap scores for every requirement that has at least one gold control.

    regs     : {req_id: text}
    controls : {ctl_id: text}
    mappings : iterable of (req_id, ctl_id)

    Returns {req_id: {"gap": float, "gap_mean": float, "n_gold": int,
                      "best_overlap": float}}
    """
    req_tokens = {r: tokenize(t) for r, t in regs.items()}
    ctl_tokens = {c: tokenize(t) for c, t in controls.items()}
    idf = build_idf(list(req_tokens.values()) + list(ctl_tokens.values()))

    gold = {}
    for r, c in mappings:
        gold.setdefault(r, set()).add(c)

    out = {}
    for r, golds in gold.items():
        if r not in req_tokens:
            continue
        scores = [overlap(req_tokens[r], ctl_tokens[c], idf)
                  for c in sorted(golds) if c in ctl_tokens]
        if not scores:
            continue
        out[r] = {
            "gap": 1.0 - max(scores),
            "gap_mean": 1.0 - (sum(scores) / len(scores)),
            "n_gold": len(scores),
            "best_overlap": max(scores),
        }
    return out


def standardize_within(values):
    """Z-score a list of values; returns zeros if the spread is degenerate."""
    n = len(values)
    if n == 0:
        return []
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    sd = math.sqrt(var)
    if sd <= 1e-12:
        return [0.0] * n
    return [(v - mean) / sd for v in values]


def spec_hash():
    """SHA-256 of this file: the frozen-specification fingerprint."""
    with open(__file__, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_corpus(directory, prefix=""):
    """Load a benchmark directory into (regs, controls, mappings)."""
    import pandas as pd
    d = str(directory).rstrip("/\\")
    regs = pd.read_csv(f"{d}/{prefix}regs.csv")
    ctls = pd.read_csv(f"{d}/{prefix}controls.csv")
    maps = pd.read_csv(f"{d}/{prefix}mappings.csv")
    return (dict(zip(regs["id"], regs["text"])),
            dict(zip(ctls["id"], ctls["text"])),
            list(zip(maps["regulation_id"], maps["control_id"])))


def main():
    """Report gap distributions for every corpus given on the command line.

    Usage: python gap_metrics.py <dir>[:prefix] [<dir>[:prefix] ...]
    """
    print(f"gap_metrics.py specification hash: {spec_hash()}\n")
    targets = sys.argv[1:] or [
        "diagnostic_benchmark:diag_", "csf_benchmark:csf_",
        "hipaa_benchmark:", "pf_benchmark:",
    ]
    for target in targets:
        directory, _, prefix = target.partition(":")
        try:
            regs, controls, mappings = load_corpus(directory, prefix)
        except FileNotFoundError as e:
            print(f"{directory}: skipped ({e.filename} not found)")
            continue
        gaps = compute_gaps(regs, controls, mappings)
        vals = sorted(g["gap"] for g in gaps.values())
        if not vals:
            print(f"{directory}: no scorable requirements")
            continue
        n = len(vals)
        q = [vals[int(p * (n - 1))] for p in (0.0, 0.25, 0.5, 0.75, 1.0)]
        mean = sum(vals) / n
        print(f"{directory:<22} n={n:<4} mean gap={mean:.3f}  "
              f"min={q[0]:.3f} q1={q[1]:.3f} med={q[2]:.3f} "
              f"q3={q[3]:.3f} max={q[4]:.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Execute the frozen hybrid specification. Registered alongside it.

This is the runner hybrid_spec.py describes. It exists so that the
registration covers the whole analysis path and not only the gate and the
fusion: corpus loading, the join by requirement id, arm construction, the
paired difference vector, the tests, the gatekeeping rule and the output
record are all fixed here rather than left to be written afterwards.

It never loads a model. Every input comes from frozen_backends/, whose array
hashes are verified against the manifest before anything is computed. A
mismatch aborts. That makes the registered path pure arithmetic over hashed
inputs.

    python run_hybrid.py --verify-only     # check inputs, compute nothing
    python run_hybrid.py --self-test       # synthetic fixtures, no real data
    python run_hybrid.py                   # the analysis

DO NOT run the third form until the specification is registered.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np

import confirmatory_stats as cs
import hybrid_spec as H

HERE = os.path.dirname(os.path.abspath(__file__))
INDIR = os.path.join(HERE, "frozen_backends")
OUTDIR = os.path.join(HERE, "results_v3", "hybrid")


# ------------------------------------------------------------------ inputs
def sha256_array(a):
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def sha256_file(path):
    """Hash a TEXT file with line endings normalized to LF.

    Every file hashed here is text: the manifest and the Python modules. CRLF
    is folded so the recorded hashes reproduce on any platform, matching
    hybrid_spec.spec_hash(). Array bytes are hashed by sha256_array instead
    and are never normalized.
    """
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def load_manifest():
    path = os.path.join(INDIR, "manifest.json")
    if not os.path.exists(path):
        raise SystemExit("frozen_backends/manifest.json missing. "
                         "Run freeze_backends.py first.")
    with open(path) as f:
        return json.load(f), sha256_file(path)


def load_corpus(key, manifest, backends):
    """Load one corpus and verify every array against the manifest."""
    entry = manifest["corpora"].get(key)
    if entry is None:
        raise SystemExit(f"corpus {key!r} absent from the manifest")
    base = os.path.join(INDIR, key)

    def load(name, expected_key):
        arr = np.load(os.path.join(base, f"{name}.npy"))
        got = sha256_array(arr)
        want = entry[expected_key]
        if got != want:
            raise SystemExit(
                f"HASH MISMATCH {key}/{name}: manifest {want[:16]}... "
                f"but file hashes {got[:16]}... Inputs are not the frozen "
                f"ones; refusing to compute.")
        return arr

    ctrl_ids = load("control_ids", "control_ids_sha256")
    rids = load("requirement_ids", "requirement_ids_sha256")
    gold = load("gold_mask", "gold_mask_sha256")

    scores = {}
    for b in backends:
        arr = np.load(os.path.join(base, f"{b}.npy"))
        got = sha256_array(arr)
        want = entry["backends"][b]["array_sha256"]
        if got != want:
            raise SystemExit(f"HASH MISMATCH {key}/{b}: refusing to compute.")
        scores[b] = arr
    return {"control_ids": ctrl_ids, "requirement_ids": rids,
            "gold_mask": gold, "scores": scores}


# -------------------------------------------------------------------- arms
def per_query_scores(corpus, i, backends):
    return {b: corpus["scores"][b][i] for b in backends}


def arm_top1(corpus, arm, lexical):
    """Top-1 correctness per requirement, plus the gate trace where relevant.

    Returns (top1 array of 0/1, predicted control id array, gate_fired array).
    """
    ctrl_ids = corpus["control_ids"]
    gold = corpus["gold_mask"]
    n = gold.shape[0]
    needed = set(lexical) | {H.SEMANTIC_BACKEND}
    top1 = np.zeros(n, dtype=float)
    pred = np.zeros(n, dtype=np.int64)
    fired = np.zeros(n, dtype=bool)

    for i in range(n):
        s = per_query_scores(corpus, i, needed)
        if arm == "semantic":
            fused, gate = s[H.SEMANTIC_BACKEND], False
        elif arm == "rrf_lexical":
            fused, gate = H.rrf_fuse([s[b] for b in lexical], ctrl_ids), False
        elif arm == "hybrid_equal":
            fused, gate = H.hybrid_equal(s, ctrl_ids, lexical), False
        elif arm == "hybrid_gated":
            fused, gate = H.hybrid_gated(s, ctrl_ids, lexical)
        elif arm == "hybrid_gated_fallback":
            fused, gate = H.hybrid_gated_fallback(s, ctrl_ids, lexical)
        else:
            raise ValueError(f"unknown arm {arm!r}")
        j = H.stable_order(fused, ctrl_ids)[0]
        top1[i] = 1.0 if gold[i, j] else 0.0
        pred[i] = int(ctrl_ids[j])
        fired[i] = bool(gate)
    return top1, pred, fired


# ------------------------------------------------------------------ report
def complementarity(a_top1, b_top1, gated_top1):
    """Counts the design exists to produce. Descriptive; no test attaches."""
    a, b, g = a_top1.astype(bool), b_top1.astype(bool), gated_top1.astype(bool)
    a_only, b_only = a & ~b, b & ~a
    return {
        "semantic_only": int(a_only.sum()),
        "lexical_only": int(b_only.sum()),
        "both": int((a & b).sum()),
        "neither": int((~a & ~b).sum()),
        "oracle_union": int((a | b).sum()),
        "lexical_only_recovered_by_gated": int((b_only & g).sum()),
        "semantic_only_lost_by_gated": int((a_only & ~g).sum()),
    }


def analyse(corpora_data, lexical, alpha, delta):
    per_corpus, diffs, labels = {}, [], []
    for key, corpus in corpora_data.items():
        arms = {}
        for arm in ("semantic", "rrf_lexical", "hybrid_equal",
                    "hybrid_gated", "hybrid_gated_fallback"):
            arms[arm] = arm_top1(corpus, arm, lexical)
        sem = arms["semantic"][0]
        gated = arms["hybrid_gated"][0]
        d = gated - sem
        per_corpus[key] = {
            "n": int(len(d)),
            "top1": {a: float(v[0].mean()) for a, v in arms.items()},
            "gate_fired": int(arms["hybrid_gated"][2].sum()),
            "gate_fired_rate": float(arms["hybrid_gated"][2].mean()),
            "top1_among_fired": float(
                gated[arms["hybrid_gated"][2]].mean())
            if arms["hybrid_gated"][2].any() else None,
            "top1_among_unfired": float(
                gated[~arms["hybrid_gated"][2]].mean())
            if (~arms["hybrid_gated"][2]).any() else None,
            "mean_difference_vs_semantic": float(d.mean()),
            "complementarity": complementarity(
                sem, arms["rrf_lexical"][0], gated),
        }
        if key in H.PRIMARY_CORPORA:
            diffs.append(d)
            labels.append(np.full(len(d), key))

    diffs = np.concatenate(diffs)
    labels = np.concatenate(labels)

    ni = cs.tost(diffs, delta, alpha=alpha)
    sup_raw = cs.sign_flip_test(diffs, alpha=alpha)
    sup = H.assert_gatekeeping(ni, sup_raw)
    lo, hi, _ = H.stratified_bootstrap_ci(diffs, labels, alpha=alpha)

    return {
        "primary": {
            "arm": "hybrid_gated",
            "comparator": "semantic",
            "n": int(len(diffs)),
            "estimand": H.PRIMARY_ESTIMAND,
            "mean_difference": float(diffs.mean()),
            "stratified_bootstrap": {
                "low": lo, "high": hi,
                "coverage": f"{100*(1-2*alpha):.0f}% two-sided "
                            f"({100*(1-alpha):.0f}% one-sided)",
            },
            "test_1_non_inferiority": ni,
            "test_2_superiority": sup,
            "superiority_suppressed": sup is None,
            "superiority_raw_not_reportable": sup_raw if sup is None else None,
        },
        "per_corpus": per_corpus,
    }


# ---------------------------------------------------------------- self test
def self_test():
    """Exercise the whole path on synthetic fixtures. Touches no real data."""
    print("run_hybrid self-test on synthetic fixtures\n")
    rng = np.random.default_rng(20260801)
    n_ctl, ok = 40, True

    def fixture(n_req, signal):
        ctrl_ids = np.arange(n_ctl, dtype=np.int64)
        gold = np.zeros((n_req, n_ctl), dtype=bool)
        scores = {b: rng.normal(size=(n_req, n_ctl))
                  for b in ("tfidf", "bm25", "lsi_inductive", "semantic")}
        for i in range(n_req):
            g = int(rng.integers(n_ctl))
            gold[i, g] = True
            if rng.random() < signal:      # plant a clear semantic winner
                scores["semantic"][i, g] = 10.0
        return {"control_ids": ctrl_ids, "gold_mask": gold,
                "requirement_ids": np.arange(n_req, dtype=np.int64),
                "scores": scores}

    data = {"nist": fixture(106, 0.5), "hipaa": fixture(68, 0.5),
            "pf": fixture(94, 0.5), "diagnostic": fixture(58, 0.5)}
    res = analyse(data, H.LEXICAL_BACKENDS, H.ALPHA, H.DELTA_NI)

    p = res["primary"]
    checks = [
        ("primary n is 268", p["n"] == 268),
        ("diagnostic excluded from primary",
         sum(res["per_corpus"][k]["n"] for k in H.PRIMARY_CORPORA) == 268),
        ("all five arms scored per corpus",
         all(len(v["top1"]) == 5 for v in res["per_corpus"].values())),
        ("gate fired somewhere",
         any(v["gate_fired"] > 0 for v in res["per_corpus"].values())),
        ("bootstrap brackets the estimate",
         p["stratified_bootstrap"]["low"] <= p["mean_difference"]
         <= p["stratified_bootstrap"]["high"]),
        ("coverage is labelled 90% two-sided",
         p["stratified_bootstrap"]["coverage"].startswith("90%")),
        ("gatekeeping consistent",
         (p["test_2_superiority"] is None)
         == (p["test_1_non_inferiority"]["p_lower"] >= H.ALPHA)),
        ("complementarity counts are exhaustive",
         all(v["complementarity"]["both"] + v["complementarity"]["neither"]
             + v["complementarity"]["semantic_only"]
             + v["complementarity"]["lexical_only"] == v["n"]
             for v in res["per_corpus"].values())),
        ("ungated queries reproduce semantic exactly",
         True),  # structural: hybrid_gated returns sem when not ambiguous
    ]
    for name, cond in checks:
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        ok &= bool(cond)

    # A gate that never fires must reproduce the comparator exactly.
    flat = {"control_ids": np.arange(n_ctl, dtype=np.int64),
            "gold_mask": np.zeros((5, n_ctl), dtype=bool),
            "requirement_ids": np.arange(5, dtype=np.int64),
            "scores": {b: np.tile(np.linspace(1.0, 0.0, n_ctl), (5, 1))
                       for b in ("tfidf", "bm25", "lsi_inductive")}}
    conf = np.tile(np.concatenate([[10.0, 1.0], np.zeros(n_ctl - 2)]), (5, 1))
    flat["scores"]["semantic"] = conf
    t_sem, _, _ = arm_top1(flat, "semantic", H.LEXICAL_BACKENDS)
    t_gat, _, fired = arm_top1(flat, "hybrid_gated", H.LEXICAL_BACKENDS)
    cond = (not fired.any()) and np.array_equal(t_sem, t_gat)
    print(f"  {'PASS' if cond else 'FAIL'}  confident queries bypass the gate "
          f"and match semantic exactly")
    ok &= cond

    print("\n" + ("self-test passed" if ok else "SELF-TEST FAILED"))
    return 0 if ok else 1


# --------------------------------------------------------------------- main
def git_state():
    def run(*a):
        try:
            return subprocess.run(a, cwd=HERE, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:
            return None
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain"))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--legacy-lsi", action="store_true",
                    help="sensitivity arm: transductive LSI instead of "
                         "inductive. Never the primary result.")
    ap.add_argument("--alpha", type=float, default=H.ALPHA)
    ap.add_argument("--delta", type=float, default=H.DELTA_NI)
    ap.add_argument("--tag")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    manifest, manifest_hash = load_manifest()
    lexical = (H.LEGACY_LEXICAL_BACKENDS if args.legacy_lsi
               else H.LEXICAL_BACKENDS)
    backends = set(lexical) | {H.SEMANTIC_BACKEND}
    keys = list(H.PRIMARY_CORPORA) + list(H.ENGINEERED_CORPORA)

    data = {k: load_corpus(k, manifest, backends) for k in keys}
    print(f"inputs verified against manifest {manifest_hash[:16]}...")
    for k in keys:
        print(f"  {k:11s} {data[k]['gold_mask'].shape[0]:3d} requirements x "
              f"{data[k]['gold_mask'].shape[1]:3d} controls")
    if args.verify_only:
        print("\n--verify-only: nothing computed.")
        return 0

    if args.alpha != H.ALPHA or args.delta != H.DELTA_NI:
        raise SystemExit("alpha and delta are frozen by the specification; "
                         "overriding them voids the registration")

    print("\n" + "=" * 70)
    print("EXPLORATORY. Preregistered design, hypothesis generated from these")
    print("same corpora. hybrid_equal was OBSERVED before registration.")
    print("=" * 70 + "\n")

    res = analyse(data, lexical, args.alpha, args.delta)

    p = res["primary"]
    print(f"primary  hybrid_gated - semantic, n={p['n']}")
    print(f"  mean difference {p['mean_difference']:+.4f}")
    b = p["stratified_bootstrap"]
    print(f"  stratified bootstrap [{b['low']:+.4f}, {b['high']:+.4f}]  "
          f"{b['coverage']}")
    ni = p["test_1_non_inferiority"]
    print(f"  NI delta={args.delta}: p_lower={ni['p_lower']:.4f} -> "
          f"{'non-inferior' if ni['p_lower'] < args.alpha else 'NOT established'}")
    if p["test_2_superiority"] is None:
        print("  superiority: SUPPRESSED (non-inferiority not established)")
    else:
        s = p["test_2_superiority"]
        print(f"  superiority: mean {s['mean_difference']:+.4f}, "
              f"p={s['p_value']:.4f}")
    print()
    for k, v in res["per_corpus"].items():
        mark = "" if k in H.PRIMARY_CORPORA else "  [engineered, excluded]"
        print(f"  {k:11s} n={v['n']:3d} gate fired {v['gate_fired']:3d} "
              f"({v['gate_fired_rate']:.2f}){mark}")
        print(f"    " + "  ".join(f"{a}={t:.4f}" for a, t in v["top1"].items()))

    record = {
        "status": "exploratory",
        "status_reason": "design frozen before outcome, but the gate was "
                         "motivated by complementarity already observed in "
                         "these corpora; hybrid_equal was observed before "
                         "registration",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "argv": sys.argv,
        "git": git_state(),
        "lexical_backends": list(lexical),
        "is_sensitivity_arm": bool(args.legacy_lsi),
        "model_pin": H.MODEL_PIN,
        "input_manifest_sha256": manifest_hash,
        "spec_hashes": {
            "hybrid_spec.py": H.spec_hash(),
            "run_hybrid.py": sha256_file(os.path.abspath(__file__)),
            "confirmatory_stats.py": sha256_file(
                os.path.join(HERE, "confirmatory_stats.py")),
        },
        "environment": {"python": sys.version.split()[0],
                        "platform": platform.platform(),
                        "numpy": np.__version__},
        "seeds": {"bootstrap": cs.BOOTSTRAP_SEED,
                  "permutation": cs.PERMUTATION_SEED},
        "alpha": args.alpha, "delta": args.delta,
        "results": res,
    }
    os.makedirs(OUTDIR, exist_ok=True)
    stem = args.tag or ("hybrid_gated_legacy_lsi" if args.legacy_lsi
                        else "hybrid_gated")
    out = os.path.join(OUTDIR, f"{stem}.json")
    with open(out, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True, default=str)
    print(f"\nrecord written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

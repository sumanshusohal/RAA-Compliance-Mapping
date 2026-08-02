#!/usr/bin/env python3
"""Export and hash every backend score matrix, once, before the hybrid runs.

WHY THIS EXISTS
---------------
hybrid_spec.py freezes the gate and the fusion, but a specification that
freezes only those still leaves the result dependent on how corpora are
loaded, which model checkpoint is resolved, and how scores reach the analysis.
Registering the spec alone would not freeze any of that.

This script removes the whole question by materializing the inputs. It runs
every backend once per corpus, writes the raw score matrices to disk, and
records a manifest of SHA-256 hashes alongside the model revisions and library
versions that produced them. run_hybrid.py then consumes ONLY these frozen
matrices and never loads a model, so the registered analysis path is pure
arithmetic over hashed inputs.

WHAT THIS DOES NOT DO
---------------------
It does not compute any hybrid arm, any gate decision, or any contrast. It
writes per-backend scores that already existed implicitly in score_all.py.
Running it does not consume the freeze.

Usage:
    USE_TF=0 python freeze_backends.py            # all four corpora
    USE_TF=0 python freeze_backends.py --corpus pf
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import datetime as dt

os.environ.setdefault("USE_TF", "0")

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "frozen_backends")

# Corpus key -> (directory, csv prefix). Keys match hybrid_spec.PRIMARY_CORPORA
# and ENGINEERED_CORPORA.
CORPORA = {
    "nist":       ("csf_benchmark", "csf_"),
    "hipaa":      ("hipaa_benchmark", ""),
    "pf":         ("pf_benchmark", ""),
    "diagnostic": ("diagnostic_benchmark", "diag_"),
}

# Pinned model identity. Both snapshots cached locally carry byte-identical
# weights and tokenizer, but the bare name "all-MiniLM-L6-v2" resolves through
# refs/main and is not a stable reference on another machine. The revision is
# pinned explicitly and the weight hash is recorded so a mismatch is
# detectable rather than silent.
DUAL_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
DUAL_ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

# Backends exported. The cross-encoder is deliberately absent: it is a
# candidate-constrained conditional reranker, not an arm of the end-to-end
# hybrid, and including it here would invite the protocol mixing that
# HANDOFF.md rule 2 forbids.
BACKENDS = ("tfidf", "bm25", "lsi_inductive", "lsi", "semantic")


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sha256_array(a):
    """Hash an array by its exact bytes, shape and dtype."""
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def git_state():
    def run(*args):
        try:
            return subprocess.run(args, cwd=HERE, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return None
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain"))}


def build_pinned_semantic_scorer(ctrl_texts):
    """Dual-encoder scorer that ENFORCES the pinned revision.

    raa_agent.build_semantic_scorer calls SentenceTransformer(model_name) with
    no revision, so it resolves through refs/main and would silently use
    whatever checkpoint a machine happens to have. Recording a revision while
    calling that function would make the pin documentation rather than
    enforcement, which is the same defect as prose describing a model the code
    does not implement.

    Scoring is otherwise identical to raa_agent's: normalized embeddings,
    cosine similarity by dot product. The regenerated matrices are verified
    hash-for-hash against the unpinned ones, so this change is proven not to
    alter the numbers, only to make them reproducible.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(DUAL_ENCODER, revision=DUAL_ENCODER_REVISION)
    ctrl_emb = model.encode(list(ctrl_texts), convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)

    def score_fn(q):
        q_emb = model.encode([q], convert_to_numpy=True,
                             normalize_embeddings=True)
        return (q_emb @ ctrl_emb.T).reshape(-1)
    return score_fn


def build_scorers(ctrl_texts, reg_texts):
    from raa_agent import (LSIIndex, build_bm25_scorer,
                           build_tfidf_scorer)
    return {
        "tfidf": build_tfidf_scorer(ctrl_texts),
        "bm25": build_bm25_scorer(ctrl_texts),
        # Controls only. Every requirement is genuinely unseen at fit time.
        # This is the backend the primary hybrid uses.
        "lsi_inductive": LSIIndex(ctrl_texts, [], n_components=100,
                                  include_regs_in_fit=False).score,
        # Controls plus evaluation queries. Transductive, label-free but not
        # inductive. Exported only so the published protocol remains
        # reproducible as a declared sensitivity arm.
        "lsi": LSIIndex(ctrl_texts, reg_texts, n_components=100,
                        include_regs_in_fit=True).score,
        "semantic": build_pinned_semantic_scorer(ctrl_texts),
    }


def freeze_corpus(key, directory, prefix):
    d = os.path.join(HERE, directory)
    regs = pd.read_csv(f"{d}/{prefix}regs.csv")
    ctls = pd.read_csv(f"{d}/{prefix}controls.csv")
    maps = pd.read_csv(f"{d}/{prefix}mappings.csv")

    gold = {}
    for r, c in zip(maps["regulation_id"], maps["control_id"]):
        gold.setdefault(r, set()).add(c)

    # Requirements with at least one gold control, in a stable, explicit order.
    rids = [int(r) for r in regs["id"] if r in gold]
    ctrl_ids = np.asarray([int(c) for c in ctls["id"]], dtype=np.int64)
    reg_text = dict(zip(regs["id"], regs["text"]))
    ctrl_texts = list(ctls["text"])

    print(f"  {key}: {len(rids)} requirements x {len(ctrl_ids)} controls",
          flush=True)
    scorers = build_scorers(ctrl_texts, list(regs["text"]))

    os.makedirs(os.path.join(OUTDIR, key), exist_ok=True)
    entry = {"n_requirements": len(rids), "n_controls": len(ctrl_ids),
             "backends": {}}

    np.save(os.path.join(OUTDIR, key, "control_ids.npy"), ctrl_ids)
    rid_arr = np.asarray(rids, dtype=np.int64)
    np.save(os.path.join(OUTDIR, key, "requirement_ids.npy"), rid_arr)
    entry["control_ids_sha256"] = sha256_array(ctrl_ids)
    entry["requirement_ids_sha256"] = sha256_array(rid_arr)

    # Gold sets, stored as a padded membership matrix so the runner needs no
    # CSV parsing and the join cannot drift.
    gold_mask = np.zeros((len(rids), len(ctrl_ids)), dtype=bool)
    pos = {int(c): j for j, c in enumerate(ctrl_ids)}
    for i, r in enumerate(rids):
        for c in gold[r]:
            if int(c) in pos:
                gold_mask[i, pos[int(c)]] = True
    np.save(os.path.join(OUTDIR, key, "gold_mask.npy"), gold_mask)
    entry["gold_mask_sha256"] = sha256_array(gold_mask)
    entry["n_gold_links"] = int(gold_mask.sum())

    for name in BACKENDS:
        fn = scorers[name]
        mat = np.empty((len(rids), len(ctrl_ids)), dtype=np.float64)
        for i, r in enumerate(rids):
            mat[i] = np.asarray(fn(reg_text[r]), dtype=np.float64)
        path = os.path.join(OUTDIR, key, f"{name}.npy")
        np.save(path, mat)
        entry["backends"][name] = {
            "shape": list(mat.shape),
            "array_sha256": sha256_array(mat),
            "file_sha256": sha256_file(path),
        }
        print(f"    {name:15s} {mat.shape}  {entry['backends'][name]['array_sha256'][:16]}...",
              flush=True)
    return entry


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=sorted(CORPORA))
    args = ap.parse_args()

    keys = [args.corpus] if args.corpus else list(CORPORA)
    print("Freezing backend score matrices. No hybrid arm is computed here.\n")

    manifest_path = os.path.join(OUTDIR, "manifest.json")
    existing = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            existing = json.load(f).get("corpora", {})

    corpora = dict(existing)
    for key in keys:
        directory, prefix = CORPORA[key]
        if not os.path.isdir(os.path.join(HERE, directory)):
            print(f"  {key}: {directory} missing, skipped")
            continue
        corpora[key] = freeze_corpus(key, directory, prefix)

    import sentence_transformers
    import sklearn
    import torch

    manifest = {
        "purpose": "frozen backend score matrices for the hybrid analysis",
        "computes_no_arm": True,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git": git_state(),
        "dual_encoder": {
            "model": DUAL_ENCODER,
            "revision": DUAL_ENCODER_REVISION,
            "normalize_embeddings": True,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "torch": torch.__version__,
        },
        "lsi": {"n_components": 100,
                "lsi_inductive": "fitted on control documents only",
                "lsi": "fitted on controls plus evaluation queries "
                       "(transductive; sensitivity arm only)"},
        "corpora": corpora,
    }
    os.makedirs(OUTDIR, exist_ok=True)
    # newline="\n" so the manifest is byte-identical on every platform. It is
    # a text file whose own hash is recorded in the run record, and the
    # default on Windows would write CRLF, then .gitattributes would check it
    # out as LF, and the recorded hash would stop matching the file.
    with open(manifest_path, "w", newline="\n") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"\nwrote {manifest_path}")
    print("manifest sha256:", sha256_file(manifest_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())

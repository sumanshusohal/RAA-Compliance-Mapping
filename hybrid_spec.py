#!/usr/bin/env python3
"""Hybrid lexical-semantic retrieval: the frozen analysis specification.

FROZEN SPECIFICATION. This file is hashed and timestamped BEFORE any arm
defined in it is run. Nothing here may change after the timestamp: not the
fusion, not the gate, not the endpoint, not the margin, not the decision
rule. Changes go in a new file with a new hash and a new timestamp, and the
change is reported.

WHY THIS FILE EXISTS AND WHY IT IS FIRST
----------------------------------------
HANDOFF.md opens by recording that all four corpora lost confirmatory status
because the HIPAA and PF ablations ran before any spec was timestamped, and
those ablations already contained the reform-vs-multi contrast. The hybrid is
the one contrast in the pending list that has not been computed yet, so it is
the only place where that status is still available to lose.

It is losable in a specific way. The pending list puts full RAA (item 2)
before the hybrid (item 3), but full RAA is an ARM of the hybrid comparison.
Running it first and then writing this file would set the gate design and the
margin with one arm's results already visible, which is the original failure
in miniature. Hence: this file first, timestamped, then any arm.

WHAT CONFIRMATORY STATUS THIS CAN AND CANNOT CARRY
--------------------------------------------------
Freezing now is necessary but not sufficient, and the writeup must not
overclaim it.

The gate below is motivated by complementarity ALREADY OBSERVED on these
corpora: five methods tie at 35/94 on PF while TF-IDF and the dual-encoder
each uniquely solve 15 requirements the other misses, and the oracle union of
lexical RRF and the cross-encoder reaches 53/94. That observation is why a
gated hybrid is worth testing rather than an equal-weight one. It is also why
a result on these four corpora is PREREGISTERED BUT STILL EXPLORATORY: the
design was frozen before the outcome, but the hypothesis was generated from
the same corpora that will test it.

    status on the four existing corpora : preregistered, exploratory
    status available on a future corpus : confirmatory

Confirmatory status is regained only by applying this file, unchanged and at
this hash, to a corpus that does not exist yet. RESERVED_CONFIRMATORY_CORPUS
below is the placeholder for it. Do not quietly promote the exploratory run.

WHAT IS BEING ASKED
-------------------
One question, stated before the answer is known: does a gate recover the
requirements lexical RRF uniquely solves without giving up the ones semantic
retrieval already solves? A hybrid that gains on one side and loses the same
amount on the other is a null result, and the design has to be able to say so,
which is why non-inferiority is tested alongside superiority rather than
accuracy alone being reported.
"""
import hashlib

import numpy as np

# =====================================================================
# Inherited constants. NOT free parameters.
# =====================================================================
# Both are adopted unchanged from code that predates this specification, so
# neither is a number chosen with the hybrid outcome in view.

# score_all.py:52. The fusion constant used by every existing RRF result.
RRF_K = 60

# raa_agent.py:1230, rel_gap_retry_threshold. The relative top-2 margin below
# which RAA already declares a query ambiguous and reformulates. The gated
# hybrid fires on the same statistic at the same threshold, so the gate
# introduces no tuned constant of its own.
REL_MARGIN_GATE = 0.10

# score_all.py:51. Rank cutoff for the secondary rank metrics.
TOPK = 5


# =====================================================================
# Gate statistic
# =====================================================================

def rel_margin(scores):
    """Relative top-2 margin of a score vector. Label-free by construction.

    THE GATE CANNOT USE THE GAP METRIC. gap_metrics.compute_gaps maximizes
    overlap over the gold control set G_r, so it is not computable at query
    time. It stays what it already is, the RQ2 moderator, and plays no part in
    routing.

    Deviation from raa_agent._ambiguous, stated because it is a deviation:
    that method computes gap / max(conf, 1e-9) on raw backend scores, which is
    safe for BM25 and RRF because both are non-negative. The dual-encoder
    returns cosine similarities that can be near zero or negative, and there
    the raw form degenerates: a negative top score collapses the denominator
    to 1e-9, the ratio explodes, and the gate silently never fires exactly on
    the low-confidence queries it exists to catch. Scores are therefore
    shifted to non-negative per query before the ratio is taken. The statistic
    remains scale-invariant and the threshold remains 0.10.
    """
    s = np.asarray(scores, dtype=float)
    if s.size < 2:
        return float("inf")
    s = s - s.min()
    order = np.sort(s)[::-1]
    top = order[0]
    if top <= 1e-12:
        # Every control scored identically. There is no signal to be
        # confident about, so the query is maximally ambiguous.
        return 0.0
    return float((top - order[1]) / top)


def is_ambiguous(scores):
    """Gate predicate: True when the query should consult the lexical side."""
    return rel_margin(scores) < REL_MARGIN_GATE


# =====================================================================
# Fusion
# =====================================================================

def rrf_fuse(score_arrays):
    """Reciprocal rank fusion. Replicates score_all.rrf_fuse exactly.

    Held byte-identical in behaviour to the existing implementation rather
    than improved, so hybrid arms and the existing rrf_lexical rows differ
    only in which backends are fused. verify_fusion_matches_score_all()
    asserts this and must pass before any arm is run.

    Known and deliberately preserved: np.argsort defaults to quicksort, so
    ties between equal scores break by an arbitrary but deterministic order.
    Making the sort stable would be an improvement and would also make every
    existing RRF number incomparable, so it is not done here.
    """
    arrays = [np.asarray(s, dtype=float) for s in score_arrays]
    n = len(arrays[0])
    fused = np.zeros(n, dtype=float)
    for scores in arrays:
        order = np.argsort(-scores)
        ranks = np.empty(len(scores), dtype=float)
        ranks[order] = np.arange(1, len(scores) + 1)
        fused += 1.0 / (RRF_K + ranks)
    return fused


def verify_fusion_matches_score_all(n_trials=25, n_items=300, seed=20260801):
    """Assert this module's fusion agrees with score_all.py's on random input.

    Run as a precondition, not at import: score_all imports raa_agent, which
    loads models.
    """
    import score_all
    rng = np.random.default_rng(seed)
    ids = list(range(n_items))
    for _ in range(n_trials):
        arrays = [rng.normal(size=n_items) for _ in range(3)]
        mine = rrf_fuse(arrays)
        theirs = score_all.rrf_fuse(arrays, ids)
        if not np.array_equal(np.argsort(-mine), np.argsort(-theirs)):
            raise AssertionError("fusion diverged from score_all.rrf_fuse")
    return True


# =====================================================================
# The two hybrid arms
# =====================================================================
# Both take the per-query score vectors already produced by score_all.py's
# scorers, over the SAME control ordering, and return a fused score vector.
# Neither sees a label.

LEXICAL_BACKENDS = ("tfidf", "bm25", "lsi")
SEMANTIC_BACKEND = "semantic"


def hybrid_equal(scores_by_backend):
    """Equal-weight arm: one RRF over the three lexical backends and semantic.

    The straw man the gated arm has to beat. If complementarity were uniform
    across queries this would capture it, and no gate would be needed.
    """
    keys = list(LEXICAL_BACKENDS) + [SEMANTIC_BACKEND]
    return rrf_fuse([scores_by_backend[k] for k in keys])


def hybrid_gated(scores_by_backend):
    """Semantic-primary gated arm. THE PRIMARY ARM.

    Semantic retrieval ranks the corpus. Only when its own top-2 relative
    margin falls below REL_MARGIN_GATE is the lexical side consulted, and then
    the query is re-ranked by RRF over all four backends.

    The gate reads ONLY the semantic score vector, so routing is decided
    before any lexical work and before any label is touched.

    Returns (fused_scores, gate_fired).
    """
    sem = np.asarray(scores_by_backend[SEMANTIC_BACKEND], dtype=float)
    if not is_ambiguous(sem):
        return sem, False
    return hybrid_equal(scores_by_backend), True


def hybrid_gated_fallback(scores_by_backend):
    """Pre-declared SECONDARY variant: hard fallback instead of fusion.

    Identical gate, different action. When the gate fires this hands the query
    to lexical RRF outright rather than fusing all four. Declared here, before
    running, precisely so that whichever of the two does better cannot be
    presented afterwards as the design that was intended all along.

    Returns (fused_scores, gate_fired).
    """
    sem = np.asarray(scores_by_backend[SEMANTIC_BACKEND], dtype=float)
    if not is_ambiguous(sem):
        return sem, False
    return rrf_fuse([scores_by_backend[k] for k in LEXICAL_BACKENDS]), True


# =====================================================================
# Arms and protocol sections
# =====================================================================
# Protocol rule 2 in HANDOFF.md: end-to-end methods rank the whole corpus,
# conditional rerankers see only the dual-encoder top-20 and are bounded by
# its Recall@20 ceiling (0.983 diagnostic, 0.906 CSF, 0.882 HIPAA, 0.840 PF).
# The primary comparison is END-TO-END ONLY. Cross-encoder and LLM rows may be
# reported, never in the same table.

ARMS = {
    # name                     section        status      source
    "semantic":               ("end_to_end", "reference", "score_all.py"),
    "rrf_lexical":            ("end_to_end", "reference", "score_all.py"),
    "hybrid_equal":           ("end_to_end", "new",       "hybrid_spec.py"),
    "hybrid_gated":           ("end_to_end", "PRIMARY",   "hybrid_spec.py"),
    "hybrid_gated_fallback":  ("end_to_end", "secondary", "hybrid_spec.py"),
    "raa_full":               ("end_to_end", "reference", "pending item 2"),
    "raa_no_reform":          ("end_to_end", "reference", "pending item 2"),
    "reranker":               ("conditional", "context",  "score_all.py"),
}

# raa_full and raa_no_reform are the reformulation-off control pair. Because
# RAA's reformulation trigger and this gate are the SAME statistic at the SAME
# threshold, the pair also answers whether any hybrid gain is the gate doing
# work or merely reformulation firing on the same queries. That confound is
# named here rather than discovered later.

# =====================================================================
# Corpora
# =====================================================================
# Fixed now. No corpus may be added, dropped, or re-derived after the
# timestamp on the basis of what any arm produced.

PRIMARY_CORPORA = ("nist", "hipaa", "pf")     # CSF 1.1, HIPAA, PF: real text
ENGINEERED_CORPORA = ("diagnostic",)          # author-built, reported apart

# The diagnostic corpus is NEVER pooled into the primary estimate. It is the
# corpus that carries the retired reformulation effect (+0.1405, sign-flip
# p=0.0094, against +0.0088 / +0.0078 / +0.0027 on the real three), so pooling
# it would let an engineered corpus drive a conclusion about real text a
# second time. It is reported as a labelled contrast.

RESERVED_CONFIRMATORY_CORPUS = None   # set only when such a corpus exists


# =====================================================================
# Endpoint
# =====================================================================

PRIMARY_ENDPOINT = "top1"
ENDPOINT_UNIT = "one value per requirement, scored once over the full corpus"
SECONDARY_ENDPOINTS = ("rr@5", "recall@5", "first_gold_rank")

# Protocol rule 1: one estimand. Every arm is scored once per requirement over
# the full control corpus via score_all.py. The repeated-holdout rows are not
# admissible here; on HIPAA they test requirements between 2 and 12 times
# across 30 seeds, and pooling them weights requirements unevenly.
#
# This also removes a caveat that applies to the existing reform-vs-multi
# records. Those run on per-requirement MEANS over an uneven seed count, so
# the per-requirement variances are unequal and the t approximation in TOST is
# approximate. Scored once per requirement, Top-1 is a clean 0/1 per
# requirement and the paired difference is exactly {-1, 0, +1}.


# =====================================================================
# Hypotheses, tests and margins
# =====================================================================
# ONE primary test. Everything else is secondary or descriptive, and is
# labelled as such in the output. This mirrors the discipline already imposed
# on RQ2 in confirmatory_stats.py, where an effect in one stratum plus none in
# another does not constitute a finding.

ALPHA = 0.05

# Non-inferiority margin, absolute Top-1. Adopted from the equivalence margin
# already fixed for the reform-vs-multi contrast (precision_analysis.py:43,
# run_confirmatory.py --delta default), so it is not a number invented for
# this test with the answer in view.
DELTA_NI = 0.05

PRIMARY_HYPOTHESIS = {
    "arm": "hybrid_gated",
    "comparator": "semantic",
    "population": "PRIMARY_CORPORA pooled, corpus fixed effects",
    "endpoint": PRIMARY_ENDPOINT,
    # Non-inferiority FIRST. The question is whether gating costs anything,
    # and a gate that gains on lexical-friendly queries while losing an equal
    # number of semantic-friendly ones is the outcome most worth detecting.
    "test_1_non_inferiority": {
        "null": "mean(hybrid_gated - semantic) <= -DELTA_NI",
        "method": "confirmatory_stats.tost, one-sided lower arm (p_lower)",
        "margin": DELTA_NI,
        "reject_if": "p_lower < ALPHA",
    },
    # Superiority is tested only if non-inferiority is established. A gated
    # hybrid that is not non-inferior has no superiority claim worth making,
    # and this ordering fixes the multiplicity without a correction.
    "test_2_superiority": {
        "null": "mean(hybrid_gated - semantic) = 0",
        "method": "confirmatory_stats.sign_flip_test",
        "gated_on": "test_1 rejects",
        "reject_if": "p_value < ALPHA and mean_difference > 0",
    },
}

SECONDARY_ANALYSES = (
    # Pre-declared, reported with their own labels, never promoted to primary.
    "hybrid_equal vs semantic, same two tests",
    "hybrid_gated vs rrf_lexical, same two tests",
    "hybrid_gated_fallback vs hybrid_gated, descriptive only",
    "per-corpus breakdown of the primary contrast, descriptive only",
    "gate firing rate per corpus, with Top-1 among fired and unfired queries",
    "diagnostic corpus, all of the above, labelled engineered",
    "raa_full and raa_no_reform against hybrid_gated, once item 2 lands",
)

# The complementarity check the whole design exists to answer. Descriptive by
# construction: an oracle is not an achievable method, and no p-value attaches
# to it. Reported as counts of requirements, from predicted_top1 in
# shared_ranking_scores.csv.
COMPLEMENTARITY_REPORT = (
    "n solved by semantic only",
    "n solved by rrf_lexical only",
    "n solved by both",
    "n solved by neither",
    "n of the rrf_lexical-only set recovered by hybrid_gated",   # the gain
    "n of the semantic-only set lost by hybrid_gated",           # the cost
    "oracle union of semantic and rrf_lexical, as the ceiling",
)

STOPPING_RULE = (
    "Every arm is scored once, over the full corpus, in a single pass. There "
    "is no interim look, no seed sweep, and no re-run on a different corpus "
    "subset. If an arm errors, the fix is committed and ALL arms are re-run "
    "together from the same commit."
)

PROHIBITED = (
    "Adding a backend to a hybrid arm after seeing any arm's Top-1.",
    "Moving REL_MARGIN_GATE off 0.10.",
    "Moving DELTA_NI off 0.05.",
    "Promoting hybrid_gated_fallback to primary if it does better.",
    "Pooling the diagnostic corpus into the primary estimate.",
    "Reporting end-to-end and conditional arms in one table.",
    "Calling any arm here RAA. rrf_lexical is not RAA and neither is a "
    "hybrid built on it; only pending item 2 produces an arm that may carry "
    "the name.",
    "Describing a NIST crosswalk prediction as a false positive merely "
    "because it is unlisted. The crosswalks are a silver standard and the "
    "HIPAA OLIR is marked 'Comprehensive: No'.",
)


def spec_hash():
    """SHA-256 of this file: the frozen-specification fingerprint."""
    with open(__file__, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


if __name__ == "__main__":
    print(__doc__)
    print(f"hybrid_spec.py specification hash: {spec_hash()}")
    print(f"\nprimary arm      : {PRIMARY_HYPOTHESIS['arm']}")
    print(f"comparator       : {PRIMARY_HYPOTHESIS['comparator']}")
    print(f"endpoint         : {PRIMARY_ENDPOINT}")
    print(f"NI margin        : {DELTA_NI}")
    print(f"gate threshold   : {REL_MARGIN_GATE} (inherited)")
    print(f"primary corpora  : {', '.join(PRIMARY_CORPORA)}")
    print(f"\nstatus on existing corpora: preregistered, EXPLORATORY")
    print(f"reserved confirmatory corpus: {RESERVED_CONFIRMATORY_CORPUS}")

#!/usr/bin/env python3
"""Hybrid lexical-semantic retrieval: the frozen analysis specification.

FROZEN SPECIFICATION. This file is hashed and timestamped BEFORE any arm
defined in it is run. Nothing here may change after the timestamp: not the
fusion, not the gate, not the endpoint, not the margin, not the decision
rule. Changes go in a new file with a new hash and a new timestamp, and the
change is reported.

Revision note: an earlier draft of this file was committed and reviewed
BEFORE any public timestamp, and four defects were corrected as a result
(gate normalization, an estimand that the named test did not implement, a
backend conflation in the motivation, and unspecified tie-breaking). That
history is deliberately left visible in the repository rather than amended
away. Nothing in this file has been run against any arm.

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
corpora (see MOTIVATING_COMPLEMENTARITY). That observation is why a gated
hybrid is worth testing rather than an equal-weight one. It is also why a
result on these four corpora is PREREGISTERED BUT STILL EXPLORATORY: the
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
requirements lexical RRF uniquely solves without giving up the ones the
dual-encoder already solves? A hybrid that gains on one side and loses the
same amount on the other is a null result, and the design has to be able to
say so, which is why non-inferiority is tested alongside superiority rather
than accuracy alone being reported.
"""
import hashlib

import numpy as np

# =====================================================================
# Motivating observation, stated with the exact pairing it came from
# =====================================================================
# These two pairings are DIFFERENT and an earlier draft conflated them. The
# 53/94 union belongs to the cross-encoder; the symmetric 15/15 split belongs
# to the dual-encoder. The primary hybrid uses the DUAL-ENCODER, so the second
# row is the one that motivates it.
#
#   pairing                                  union   unique A / unique B
#   rrf_lexical + reranker (cross-encoder)   53/94        10 / 18
#   rrf_lexical + semantic (dual-encoder)    50/94        15 / 15   <- primary
#
# Both are on PF, where five methods tie at 35/94. The dual-encoder pairing is
# the relevant one because the primary comparison is END-TO-END: the
# cross-encoder only reranks a dual-encoder top-20 and is bounded by that
# candidate ceiling, so it cannot be an arm in a full-corpus comparison.
#
# Consequence for interpretation, stated now: a successful primary hybrid
# retains DUAL-ENCODER performance while recovering lexical-only wins. It is
# not required, and should not be reported as failing, if it does not reach
# the cross-encoder numbers observed on CSF and PF.
MOTIVATING_COMPLEMENTARITY = {
    "corpus": "pf",
    "primary_pairing": {"a": "rrf_lexical", "b": "semantic",
                        "union": 50, "n": 94, "a_only": 15, "b_only": 15},
    "conditional_pairing": {"a": "rrf_lexical", "b": "reranker",
                            "union": 53, "n": 94, "a_only": 10, "b_only": 18},
}

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

    Definition:

        rel_margin = (s_1 - s_2) / max(|s_1|, |s_2|, 1e-9)

    Relationship to raa_agent._ambiguous, which computes gap / max(conf,
    1e-9) on raw scores. For a NON-NEGATIVE score vector s_1 >= s_2 >= 0, so
    max(|s_1|, |s_2|) = s_1 and this reduces to the RAA definition exactly.
    BM25 and RRF are non-negative, so on every backend RAA actually uses, the
    statistic and the 0.10 threshold are unchanged.

    Why the extension is needed: the dual-encoder returns cosine similarities
    that can be near zero or negative, and there the raw form degenerates. A
    negative top score collapses the denominator to 1e-9, the ratio explodes,
    and the gate silently never fires exactly on the low-confidence queries it
    exists to catch. Top-2 scores of -0.30 and -0.31 are a 3% margin and
    return 1e7 under the raw form.

    Why NOT min-subtraction: an earlier draft normalized by subtracting the
    per-query minimum. That also fixes the sign problem, but it makes the
    statistic depend on the WORST-scoring control, so a single extreme
    negative score compresses every margin toward zero and the gate fires
    almost everywhere. It would also mean 0.10 no longer denotes the quantity
    RAA thresholded, which would break the claim that no constant here is new.
    The max-magnitude denominator depends only on the top two scores.
    """
    s = np.asarray(scores, dtype=float)
    if s.size < 2:
        return float("inf")
    if not np.all(np.isfinite(s)):
        raise ValueError("gate received non-finite scores")
    top2 = np.sort(s)[::-1][:2]
    s1, s2 = float(top2[0]), float(top2[1])
    return (s1 - s2) / max(abs(s1), abs(s2), 1e-9)


def is_ambiguous(scores):
    """Gate predicate: True when the query should consult the lexical side."""
    return rel_margin(scores) < REL_MARGIN_GATE


# =====================================================================
# Deterministic ordering
# =====================================================================

def stable_order(scores, control_ids):
    """Rank controls by score descending, breaking ties by control id.

    score_all.rrf_fuse uses np.argsort's default quicksort, so ties break by
    an order that is arbitrary, undocumented, and not guaranteed stable across
    NumPy versions or platforms. That is tolerable for an exploratory result
    and NOT tolerable for a registered one, which a third party has to be able
    to reproduce exactly.

    Ties are common: on PF, TF-IDF assigns exactly zero to a mean of 211 of
    300 controls per query, and RRF ranks all 300, so tail placement feeds
    back into every fused score.

    Adopting this rule was measured before it was adopted, not assumed safe.
    Re-running lexical RRF under four random control re-orderings, which is
    what an alternative tie-break amounts to, changed the fused top-1 on ZERO
    queries across all three real corpora (PF 35/94, CSF 43/106, HIPAA 19/68,
    identical every time). So the deterministic rule costs nothing in
    comparability with existing numbers, and the reason the earlier draft gave
    for keeping the unstable sort does not survive contact with the data.
    """
    scores = np.asarray(scores, dtype=float)
    ids = np.asarray(control_ids)
    return np.lexsort((ids, -scores))


# =====================================================================
# Fusion
# =====================================================================

def rrf_fuse(score_arrays, control_ids):
    """Reciprocal rank fusion with a deterministic tie-break.

    Same constant and same formula as score_all.rrf_fuse; the only difference
    is that ties resolve by control id instead of by quicksort accident. See
    stable_order for the evidence that this changes no existing result.
    """
    arrays = [np.asarray(s, dtype=float) for s in score_arrays]
    n = len(arrays[0])
    if any(a.shape != (n,) for a in arrays):
        raise ValueError("backend score vectors have mismatched lengths")
    if len(control_ids) != n:
        raise ValueError("control_ids length does not match score vectors")
    fused = np.zeros(n, dtype=float)
    for scores in arrays:
        order = stable_order(scores, control_ids)
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(1, n + 1)
        fused += 1.0 / (RRF_K + ranks)
    return fused


def verify_fusion_matches_score_all(n_trials=25, n_items=300, seed=20260801):
    """Assert this module's fusion agrees with score_all.py's on TIE-FREE input.

    Restricted to continuous random scores, where ties have probability zero,
    because on tied input the two are INTENDED to differ: that is the whole
    point of stable_order. This checks that the constant and the formula match,
    not that the tie-break matches.

    Run as a precondition, not at import: score_all imports raa_agent, which
    loads models.
    """
    import score_all
    rng = np.random.default_rng(seed)
    ids = np.arange(n_items)
    for _ in range(n_trials):
        arrays = [rng.normal(size=n_items) for _ in range(3)]
        mine = rrf_fuse(arrays, ids)
        theirs = score_all.rrf_fuse(arrays, list(ids))
        if not np.allclose(np.sort(mine), np.sort(theirs)):
            raise AssertionError("fusion diverged from score_all.rrf_fuse")
        if not np.array_equal(stable_order(mine, ids),
                              np.argsort(-theirs, kind="stable")):
            raise AssertionError("ordering diverged on tie-free input")
    return True


# =====================================================================
# The two hybrid arms
# =====================================================================
# Both take the per-query score vectors already produced by score_all.py's
# scorers, over the SAME control ordering, and return a fused score vector.
# Neither sees a label.

LEXICAL_BACKENDS = ("tfidf", "bm25", "lsi")
SEMANTIC_BACKEND = "semantic"   # the DUAL-ENCODER, all-MiniLM-L6-v2


def hybrid_equal(scores_by_backend, control_ids):
    """Equal-weight arm: one RRF over the three lexical backends and semantic.

    The straw man the gated arm has to beat. If complementarity were uniform
    across queries this would capture it, and no gate would be needed.
    """
    keys = list(LEXICAL_BACKENDS) + [SEMANTIC_BACKEND]
    return rrf_fuse([scores_by_backend[k] for k in keys], control_ids)


def hybrid_gated(scores_by_backend, control_ids):
    """Semantic-primary gated arm. THE PRIMARY ARM.

    The dual-encoder ranks the corpus. Only when its own top-2 relative margin
    falls below REL_MARGIN_GATE is the lexical side consulted, and then the
    query is re-ranked by RRF over all four backends.

    The gate reads ONLY the dual-encoder score vector, so routing is decided
    before any lexical work and before any label is touched.

    Returns (fused_scores, gate_fired).
    """
    sem = np.asarray(scores_by_backend[SEMANTIC_BACKEND], dtype=float)
    if not is_ambiguous(sem):
        return sem, False
    return hybrid_equal(scores_by_backend, control_ids), True


def hybrid_gated_fallback(scores_by_backend, control_ids):
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
    return rrf_fuse([scores_by_backend[k] for k in LEXICAL_BACKENDS],
                    control_ids), True


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
    "raa_full":               ("end_to_end", "reference", "pending item 3"),
    "raa_no_reform":          ("end_to_end", "reference", "pending item 3"),
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
# Endpoint and estimand
# =====================================================================

PRIMARY_ENDPOINT = "top1"
SECONDARY_ENDPOINTS = ("rr@5", "recall@5", "first_gold_rank")

# ESTIMAND: requirement-weighted. The primary quantity is the mean paired
# Top-1 difference over the concatenated requirements of the three real
# corpora (n = 106 + 68 + 94 = 268), each requirement contributing once.
#
# An earlier draft said "pooled, corpus fixed effects" while naming
# confirmatory_stats.tost, which takes a vector of differences and runs a
# one-sample paired t-test with no fixed effects anywhere in it. The prose
# described a model the code did not implement. Corrected by choosing the
# estimand the code actually computes and saying so plainly.
#
# Consequence, stated rather than hidden: requirement weighting means CSF
# contributes 106/268 = 40% of the primary estimate and HIPAA 25%. Corpora are
# NOT equally weighted. Per-corpus results are reported descriptively so an
# unequal-weighting objection can be checked directly against them.
PRIMARY_ESTIMAND = "mean paired Top-1 difference over 268 requirements"
CORPUS_WEIGHTING = "requirement-proportional, not equal-per-corpus"
ENDPOINT_UNIT = "one value per requirement, scored once over the full corpus"

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


def stratified_bootstrap_ci(diffs, corpus_labels, alpha=0.05,
                            n_boot=10000, seed=20260801):
    """Requirement-level bootstrap resampled WITHIN corpus.

    Lives here rather than in confirmatory_stats.py so that module stays
    byte-stable: its SHA-256 is recorded inside the already-committed
    reform-vs-multi records, and those hashes should keep resolving.

    Resampling within corpus holds the corpus composition of each replicate
    fixed at the observed 106/68/94, so the interval reflects sampling of
    requirements and not sampling of how much of each corpus happened to land
    in a replicate. The point estimate is unaffected; only the interval is.
    """
    diffs = np.asarray(diffs, dtype=float)
    labels = np.asarray(corpus_labels)
    if diffs.shape != labels.shape:
        raise ValueError("diffs and corpus_labels must align")
    blocks = [np.flatnonzero(labels == c) for c in np.unique(labels)]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(bl, size=len(bl), replace=True)
                              for bl in blocks])
        draws[b] = diffs[idx].mean()
    lo, hi = np.quantile(draws, [alpha, 1.0 - alpha])
    return float(lo), float(hi), draws


# =====================================================================
# Hypotheses, tests and margins
# =====================================================================
# ONE primary test. Everything else is secondary or descriptive, and is
# labelled as such in the output. This mirrors the discipline already imposed
# on RQ2 in confirmatory_stats.py, where an effect in one stratum plus none in
# another does not constitute a finding.

ALPHA = 0.05

# Non-inferiority margin, absolute Top-1.
#
# Provenance: 0.05 is the equivalence margin already fixed for the
# reform-vs-multi contrast (precision_analysis.py:43, run_confirmatory.py
# --delta default). Reusing it means it was not chosen with this outcome in
# view. That is a provenance argument, and provenance alone is NOT a
# justification, so the operational one follows.
#
# Operational justification: the endpoint is Top-1 over a 300-control
# catalogue, and the deliverable is a ranked shortlist an analyst reviews, not
# an autonomous decision. Five absolute Top-1 points on a corpus of this size
# is three to five requirements whose top suggestion changes position while
# the gold control remains available further down the same shortlist; the
# secondary endpoints rr@5 and recall@5 are reported precisely so that a
# hybrid which trades Top-1 for shortlist quality is visible rather than
# scored as a loss. A margin materially tighter than 0.05 is not decidable at
# these sample sizes, as the precision statement below shows, so declaring one
# would be declaring a test that cannot return an answer.
DELTA_NI = 0.05

# PRECISION, computed before running and outcome-blind.
#
# Estimated from the DIAGNOSTIC corpus only, which is the one corpus excluded
# from the primary estimand, so the planning estimate does not touch the test
# set. Note the honest limitation: when precision_analysis.py was written,
# HIPAA and PF were the confirmatory corpora and diagnostic plus CSF were the
# exploratory sources. All four are now exploratory, so no fully clean
# planning source exists; the diagnostic corpus is the least contaminated one
# available because it is excluded from the primary estimate by design.
#
#   discordance (semantic vs rrf_lexical, diagnostic) = 0.241
#   sd of the paired difference                       = 0.476
#   SE at n = 268                                     = 0.0291
#   one-sided 95% half-width                          = 0.0478
#
# The margin therefore clears the half-width by 0.0022. Read plainly: at
# n = 268 non-inferiority at delta = 0.05 can be established only if the true
# difference is at least -0.0022, which is to say only if the gated hybrid is
# essentially not worse at all. This test can confirm "no cost"; it CANNOT
# confirm "a cost smaller than 5 points". Stated here so the result is not
# over-read later, and so that a failure to establish non-inferiority is
# understood as possibly a power limit rather than evidence of harm.
PRECISION_NOTE = {
    "planning_source": "diagnostic corpus only (excluded from primary)",
    "discordance": 0.241,
    "sd_paired_diff": 0.476,
    "se_at_n_268": 0.0291,
    "one_sided_95_half_width": 0.0478,
    "margin": DELTA_NI,
    "slack": 0.0022,
    "reading": "confirms 'no cost'; cannot confirm 'cost below 5 points'",
}

PRIMARY_HYPOTHESIS = {
    "arm": "hybrid_gated",
    "comparator": "semantic",
    "population": "268 requirements from PRIMARY_CORPORA, concatenated",
    "estimand": PRIMARY_ESTIMAND,
    "endpoint": PRIMARY_ENDPOINT,
    "interval": "hybrid_spec.stratified_bootstrap_ci, resampled within corpus",
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
    #
    # BINDING ON THE IMPLEMENTATION: when test 1 does not reject, the
    # superiority p-value must not be reported as a finding at all. Gatekeeping
    # controls the family-wise error rate only if the second claim is actually
    # suppressed, not merely annotated. assert_gatekeeping below enforces it.
    "test_2_superiority": {
        "null": "mean(hybrid_gated - semantic) = 0",
        "method": "confirmatory_stats.sign_flip_test",
        "gated_on": "test_1 rejects",
        "reject_if": "p_value < ALPHA and mean_difference > 0",
    },
}


def assert_gatekeeping(ni_result, superiority_result):
    """Suppress the superiority claim unless non-inferiority was established.

    Returns the reportable superiority result, or None. Call this instead of
    reading the superiority dict directly, so the gatekeeping rule cannot be
    satisfied by a footnote.
    """
    if not ni_result.get("p_lower", 1.0) < ALPHA:
        return None
    return superiority_result


SECONDARY_ANALYSES = (
    # Pre-declared, reported with their own labels, never promoted to primary.
    "hybrid_equal vs semantic, same two tests",
    "hybrid_gated vs rrf_lexical, same two tests",
    "hybrid_gated_fallback vs hybrid_gated, descriptive only",
    "per-corpus breakdown of the primary contrast, descriptive only",
    "gate firing rate per corpus, with Top-1 among fired and unfired queries",
    "diagnostic corpus, all of the above, labelled engineered",
    "raa_full and raa_no_reform against hybrid_gated, once that arm lands",
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
    "Reporting the superiority p-value when non-inferiority did not reject.",
    "Judging the primary hybrid against cross-encoder numbers; it is an "
    "end-to-end arm and the cross-encoder is candidate-constrained.",
    "Calling any arm here RAA. rrf_lexical is not RAA and neither is a "
    "hybrid built on it; only the pending full-RAA item produces an arm that "
    "may carry the name.",
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
    print(f"comparator       : {PRIMARY_HYPOTHESIS['comparator']} "
          f"(dual-encoder, end-to-end)")
    print(f"estimand         : {PRIMARY_ESTIMAND}")
    print(f"weighting        : {CORPUS_WEIGHTING}")
    print(f"NI margin        : {DELTA_NI}  (half-width {PRECISION_NOTE['one_sided_95_half_width']}, "
          f"slack {PRECISION_NOTE['slack']})")
    print(f"gate threshold   : {REL_MARGIN_GATE} (inherited)")
    print(f"primary corpora  : {', '.join(PRIMARY_CORPORA)}")
    print(f"\nstatus on existing corpora: preregistered, EXPLORATORY")
    print(f"reserved confirmatory corpus: {RESERVED_CONFIRMATORY_CORPUS}")

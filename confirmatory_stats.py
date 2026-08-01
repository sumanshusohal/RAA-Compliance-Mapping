#!/usr/bin/env python3
"""Confirmatory tests for RQ2 and RQ3, plus the outcome taxonomy.

Kept separate from query_level_stats.py so the existing paired sign-flip and
bootstrap results stay reproducible byte-for-byte. This module adds only what
the confirmatory design needs:

  * moderation_test  : the SINGLE primary RQ2 test. Paired Top-1 difference
                       regressed on within-corpus standardized continuous gap
                       with corpus fixed effects, requirement-level bootstrap.
                       Stratum contrasts are descriptive and are NOT tests.
  * tost             : equivalence with exhaustive verdicts on a 90% CI.
  * risk_coverage    : risk-coverage curve and AURC from a continuous
                       selective score.
  * classify_outcome : the two-layer taxonomy, asserted exhaustive and
                       mutually exclusive.

An effect in the high-gap stratum plus no effect in the low-gap stratum does
not demonstrate moderation; only the interaction does. That is why the
continuous model is primary.
"""
import math

import numpy as np

BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260801

# --- Layer 1: availability -------------------------------------------------
VALID = "valid"
INVALID_SUBTYPES = ("refusal", "truncation", "parse_failure",
                    "out_of_set_id", "api_failure")

# --- Layer 2: outcome, for valid responses only ----------------------------
OUTCOME_CANDIDATE_MISS = "candidate_generation_miss"
OUTCOME_CORRECT = "correct"
OUTCOME_RERANK_MISS = "reranking_miss"
OUTCOME_INCORRECT_ABSTENTION = "incorrect_abstention"
OUTCOME_CORRECT_GAP = "correct_gap_detection"
OUTCOME_FALSE_ACCEPT = "false_acceptance"

OUTCOMES = (OUTCOME_CANDIDATE_MISS, OUTCOME_CORRECT, OUTCOME_RERANK_MISS,
            OUTCOME_INCORRECT_ABSTENTION, OUTCOME_CORRECT_GAP,
            OUTCOME_FALSE_ACCEPT)


def classify_outcome(is_mapped, gold_in_candidates, accepted, selected_is_gold):
    """Assign exactly one Layer 2 outcome. See OUTCOMES for the partition.

    is_mapped         : the requirement has at least one gold control
    gold_in_candidates: some gold control reached the candidate set
    accepted          : the system committed rather than abstaining
    selected_is_gold  : the committed top-1 is a gold control
    """
    if not is_mapped:
        return OUTCOME_CORRECT_GAP if not accepted else OUTCOME_FALSE_ACCEPT
    if not gold_in_candidates:
        return OUTCOME_CANDIDATE_MISS
    if not accepted:
        return OUTCOME_INCORRECT_ABSTENTION
    return OUTCOME_CORRECT if selected_is_gold else OUTCOME_RERANK_MISS


def assert_partition(records):
    """Verify both taxonomy layers partition the records exactly."""
    n = len(records)
    valid = [r for r in records if r["availability"] == VALID]
    invalid = [r for r in records if r["availability"] != VALID]
    assert len(valid) + len(invalid) == n, "Layer 1 is not exhaustive"
    for r in invalid:
        assert r["availability"] in INVALID_SUBTYPES, \
            f"unknown invalid subtype {r['availability']!r}"
        assert r.get("outcome") is None, \
            "invalid responses must not carry a Layer 2 outcome"
    for r in valid:
        assert r["outcome"] in OUTCOMES, f"unknown outcome {r['outcome']!r}"
    return {"n": n, "valid": len(valid), "invalid": len(invalid)}


# --------------------------------------------------------------- inference --

def _bootstrap_ci(stat_fn, n, rng, alpha):
    """Percentile bootstrap over requirement indices."""
    draws = np.empty(BOOTSTRAP_N)
    for b in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, size=n)
        draws[b] = stat_fn(idx)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), draws


def moderation_test(diffs, gaps, corpora, alpha=0.05):
    """PRIMARY RQ2 TEST: does gap moderate the reformulation effect?

    diffs   : per-requirement paired Top-1 difference (treatment - baseline)
    gaps    : per-requirement raw gap score
    corpora : per-requirement corpus label, for fixed effects and for
              within-corpus standardization of the gap

    Returns the interaction coefficient with a bootstrap CI and a two-sided
    p-value from the CI's percentile position. A positive coefficient means
    the reformulation benefit grows with the vocabulary gap.
    """
    diffs = np.asarray(diffs, dtype=float)
    gaps = np.asarray(gaps, dtype=float)
    corpora = np.asarray(corpora)
    n = len(diffs)
    if n == 0:
        raise ValueError("no requirements supplied")

    labels = sorted(set(corpora.tolist()))

    def design(idx):
        """Standardize gap WITHIN corpus on the resampled data, then build X."""
        g = np.zeros(len(idx))
        for lab in labels:
            m = corpora[idx] == lab
            if not m.any():
                continue
            v = gaps[idx][m]
            sd = v.std()
            g[m] = 0.0 if sd <= 1e-12 else (v - v.mean()) / sd
        # intercept + corpus fixed effects (drop first) + standardized gap
        cols = [np.ones(len(idx))]
        for lab in labels[1:]:
            cols.append((corpora[idx] == lab).astype(float))
        cols.append(g)
        return np.column_stack(cols)

    def coef(idx):
        X = design(idx)
        y = diffs[idx]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta[-1]

    point = coef(np.arange(n))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    lo, hi, draws = _bootstrap_ci(coef, n, rng, alpha)
    # Two-sided bootstrap p: how often the sign flips relative to the point.
    p = 2.0 * min((draws <= 0).mean(), (draws >= 0).mean())
    return {"coefficient": float(point), "ci_low": lo, "ci_high": hi,
            "p_value": float(min(1.0, p)), "n": n,
            "interpretation": ("benefit grows with gap" if point > 0
                               else "benefit shrinks with gap")}


def tost(diffs, delta, conf=0.90):
    """Equivalence test with EXHAUSTIVE verdicts on a two-sided CI.

    Verdicts, which cover the whole real line:
      practically_equivalent : CI entirely inside (-delta, +delta)
      meaningfully_positive  : CI entirely above +delta
      meaningfully_negative  : CI entirely below -delta
      inconclusive           : anything else

    'inconclusive' is a legitimate pre-specified outcome, not a failure. It
    says the design cannot separate equivalence from a real effect at this
    sample size, which is itself worth reporting.
    """
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    if n < 2:
        raise ValueError("need at least two requirements")
    alpha = 1.0 - conf
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    mean = float(diffs.mean())
    lo, hi, _ = _bootstrap_ci(lambda idx: diffs[idx].mean(), n, rng, alpha)

    if lo > -delta and hi < delta:
        verdict = "practically_equivalent"
    elif lo > delta:
        verdict = "meaningfully_positive"
    elif hi < -delta:
        verdict = "meaningfully_negative"
    else:
        verdict = "inconclusive"
    return {"mean_difference": mean, "ci_low": lo, "ci_high": hi,
            "delta": delta, "confidence": conf, "n": n, "verdict": verdict}


def risk_coverage(correct, score):
    """Risk-coverage curve and AURC from a continuous selective score.

    correct : 1 if the committed answer is right
    score   : confidence, higher meaning more confident

    Accepting only the most confident predictions traces risk against
    coverage; AURC is the area under it, lower being better. A single
    accept/abstain threshold gives one point on this curve, not the curve,
    which is why a continuous score is required.
    """
    correct = np.asarray(correct, dtype=float)
    score = np.asarray(score, dtype=float)
    n = len(correct)
    if n == 0:
        raise ValueError("no predictions supplied")
    order = np.argsort(-score)
    c = correct[order]
    cum_err = np.cumsum(1.0 - c)
    k = np.arange(1, n + 1)
    risk = cum_err / k
    coverage = k / n
    aurc = float(np.trapezoid(risk, coverage) if hasattr(np, "trapezoid")
                 else np.trapz(risk, coverage))
    return {"coverage": coverage.tolist(), "risk": risk.tolist(),
            "aurc": aurc, "n": n}


def holm(pvalues, labels=None):
    """Holm correction within a family of secondary endpoints."""
    m = len(pvalues)
    labels = labels or [str(i) for i in range(m)]
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvalues[i]
        running = max(running, min(1.0, val))
        adjusted[i] = running
    return {labels[i]: adjusted[i] for i in range(m)}

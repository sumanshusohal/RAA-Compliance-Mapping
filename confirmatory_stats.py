#!/usr/bin/env python3
"""Confirmatory tests for RQ2 and RQ3, plus the outcome taxonomy.

Kept separate from query_level_stats.py so the existing paired sign-flip and
bootstrap results stay reproducible byte-for-byte. This module adds only what
the confirmatory design needs:

  * moderation_test    : the SINGLE primary RQ2 test. Paired Top-1 difference
                         regressed on within-corpus standardized continuous gap
                         with corpus fixed effects. Sign-flip permutation
                         p-value, requirement-level bootstrap CI. Stratum
                         contrasts are descriptive and are NOT tests.
  * sign_flip_test     : paired sign-flip permutation test of a mean
                         difference, for the superiority prediction.
  * tost               : real two one-sided tests for equivalence.
  * equivalence_verdict: exhaustive descriptive verdicts read off a bootstrap
                         CI. This is NOT TOST and must not be reported as one.
  * risk_coverage      : risk-coverage curve and AURC from a continuous
                         selective score.
  * classify_outcome   : the two-layer taxonomy, asserted exhaustive and
                         mutually exclusive.

An effect in the high-gap stratum plus no effect in the low-gap stratum does
not demonstrate moderation; only the interaction does. That is why the
continuous model is primary.

On p-values. An earlier version of moderation_test derived its p-value from
how often the bootstrap distribution crossed zero. That is not a test: the
bootstrap resamples under the observed data, not under the null, so the
quantity has no calibrated Type I error rate. It is replaced here by a
Freedman-Lane sign-flip permutation test, which builds a genuine null by
re-signing the residuals of the nuisance-only model. The bootstrap is kept,
but only for the interval it can legitimately produce.
"""
import math

import numpy as np
from scipy import stats

BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260801
PERMUTATION_N = 100000
PERMUTATION_SEED = 20260801

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

    full = np.arange(n)
    point = coef(full)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    lo, hi, _ = _bootstrap_ci(coef, n, rng, alpha)

    # --- Freedman-Lane sign-flip permutation p-value ----------------------
    # X = [Z | g] with Z the nuisance block (intercept + corpus fixed
    # effects). Let c be the row of pinv(X) that produces the gap
    # coefficient, so coef = c @ y and, because pinv(X) @ X = I, c @ Z = 0.
    # Regress y on Z alone, keep the residuals e, and re-sign them: the
    # permuted coefficient is c @ (Z b_z + s * e) = (c * e) @ s, since the
    # nuisance fit is annihilated by c. Under H0 (no moderation) the errors
    # are symmetric about the nuisance fit, so every sign vector is equally
    # likely and the resulting distribution is a real null.
    X = design(full)
    Z = X[:, :-1]
    c = np.linalg.pinv(X)[-1]
    resid = diffs - Z @ np.linalg.lstsq(Z, diffs, rcond=None)[0]
    weights = c * resid
    prng = np.random.default_rng(PERMUTATION_SEED)
    signs = prng.choice([-1.0, 1.0], size=(PERMUTATION_N, n))
    null = signs @ weights
    # +1 in both terms: the observed sign vector is itself a valid draw, so
    # the p-value is never exactly zero.
    p = (1.0 + np.sum(np.abs(null) >= abs(point) - 1e-12)) / (PERMUTATION_N + 1.0)

    # Do not name a direction the interval does not support. An earlier
    # version reported "benefit grows with gap" from the sign of the point
    # estimate alone, which reads a finding off a coefficient that may be
    # indistinguishable from zero, and that string is written into records
    # the manuscript cites.
    if lo <= 0.0 <= hi:
        interp = "no direction identified; interval includes zero"
    elif point > 0:
        interp = "benefit grows with gap"
    else:
        interp = "benefit shrinks with gap"

    return {"coefficient": float(point), "ci_low": lo, "ci_high": hi,
            "p_value": float(p), "p_method": "sign_flip_permutation",
            "n_permutations": PERMUTATION_N, "n": n,
            "interpretation": interp}


def sign_flip_test(diffs, alpha=0.05):
    """Paired sign-flip permutation test of a mean difference.

    Used for the HIPAA superiority prediction. The null is that the paired
    per-requirement differences are symmetric about zero, so re-signing any
    subset of them is equally likely; the two-sided p-value is the share of
    re-signed means at least as extreme as the observed one. Reported with a
    requirement-level bootstrap CI, which is an interval, not the test.
    """
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    if n < 2:
        raise ValueError("need at least two requirements")
    obs = float(diffs.mean())
    prng = np.random.default_rng(PERMUTATION_SEED)
    signs = prng.choice([-1.0, 1.0], size=(PERMUTATION_N, n))
    null = (signs @ diffs) / n
    p = (1.0 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (PERMUTATION_N + 1.0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    lo, hi, _ = _bootstrap_ci(lambda idx: diffs[idx].mean(), n, rng, alpha)
    return {"mean_difference": obs, "p_value": float(p),
            "p_method": "sign_flip_permutation",
            "n_permutations": PERMUTATION_N,
            "ci_low": lo, "ci_high": hi, "confidence": 1.0 - alpha, "n": n}


def tost(diffs, delta, alpha=0.05):
    """Two one-sided tests for equivalence. This is the real TOST.

    H01: mu <= -delta  tested against mu > -delta
    H02: mu >= +delta  tested against mu < +delta

    Equivalence within +/- delta is concluded only when BOTH nulls are
    rejected, so p_tost = max(p_lower, p_upper) and the decision rule is
    p_tost < alpha. Each one-sided test is a paired t-test on the shifted
    differences; with n in the sixties to nineties and a bounded outcome the
    t approximation is adequate, and the equivalent (1 - 2*alpha) CI is
    returned alongside so the reader can check the decision by eye.

    Note the relationship to equivalence_verdict below: the two agree by
    construction on the equivalence call when the same interval is used, but
    only this function produces p-values, and only this function is TOST.
    """
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    if n < 2:
        raise ValueError("need at least two requirements")
    mean = float(diffs.mean())
    se = float(diffs.std(ddof=1) / math.sqrt(n))
    df = n - 1

    if se <= 0.0:
        # Every requirement moved identically, so the mean is known exactly.
        p_lower = 0.0 if mean > -delta else 1.0
        p_upper = 0.0 if mean < delta else 1.0
        lo = hi = mean
        t_lower = t_upper = float("inf")
    else:
        t_lower = (mean + delta) / se     # against H01: mu <= -delta
        t_upper = (mean - delta) / se     # against H02: mu >= +delta
        p_lower = float(stats.t.sf(t_lower, df))
        p_upper = float(stats.t.cdf(t_upper, df))
        crit = stats.t.ppf(1.0 - alpha, df)
        lo, hi = mean - crit * se, mean + crit * se

    p_tost = max(p_lower, p_upper)
    return {"mean_difference": mean, "se": se, "df": df, "delta": delta,
            "t_lower": float(t_lower), "p_lower": float(p_lower),
            "t_upper": float(t_upper), "p_upper": float(p_upper),
            "p_tost": float(p_tost), "alpha": alpha,
            "equivalent": bool(p_tost < alpha),
            "ci_low": float(lo), "ci_high": float(hi),
            "ci_confidence": 1.0 - 2.0 * alpha, "n": n}


def equivalence_verdict(diffs, delta, conf=0.90):
    """EXHAUSTIVE descriptive verdicts read off a bootstrap CI.

    This is a CI-inclusion check, NOT an equivalence test. It was called
    tost() in an earlier version, which was wrong: it computes no one-sided
    test and produces no p-value. Report it as a descriptive interval reading
    and cite tost() above for the equivalence decision.

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
            "delta": delta, "confidence": conf, "n": n, "verdict": verdict,
            "method": "bootstrap_ci_inclusion_not_tost"}


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

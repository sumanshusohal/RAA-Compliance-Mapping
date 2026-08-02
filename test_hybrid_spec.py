#!/usr/bin/env python3
"""Tests for the frozen hybrid specification.

Run before the spec hash is timestamped, and again before any arm is run:

    python test_hybrid_spec.py            # gate, ordering, fusion, gatekeeping
    USE_TF=0 python test_hybrid_spec.py --with-score-all   # + cross-check

The second form imports score_all, which imports raa_agent and loads models.
"""
import sys

import numpy as np

import hybrid_spec as H

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


# ---------------------------------------------------------------- gate
def test_gate():
    print("\nrel_margin: positive scores")
    check("clear winner does not fire",
          approx(H.rel_margin(np.array([0.9, 0.5, 0.4])), (0.9 - 0.5) / 0.9))
    check("near tie fires", H.is_ambiguous(np.array([0.90, 0.89, 0.4])))
    check("scale invariance",
          approx(H.rel_margin(np.array([0.9, 0.5])),
                 H.rel_margin(np.array([90.0, 50.0]))))

    print("\nrel_margin: reduces to raa_agent._ambiguous on non-negative input")
    for s in ([5.0, 3.0, 1.0], [0.4, 0.4, 0.1], [12.0, 0.0], [1e-6, 1e-9]):
        arr = np.array(s)
        raw = (arr[0] - arr[1]) / max(arr[0], 1e-9)   # the RAA form
        check(f"matches RAA form on {s}", approx(H.rel_margin(arr), raw),
              f"got {H.rel_margin(arr)} want {raw}")

    print("\nrel_margin: negative scores (the dual-encoder case)")
    neg = np.array([-0.30, -0.31, -0.80])
    raw = (neg[0] - neg[1]) / max(neg[0], 1e-9)
    check("raw RAA form degenerates", raw > 1e6, f"raw={raw}")
    check("spec form is a small margin",
          approx(H.rel_margin(neg), 0.01 / 0.31, tol=1e-6),
          f"got {H.rel_margin(neg)}")
    check("spec form fires the gate", H.is_ambiguous(neg))
    check("all-negative clear winner does not fire",
          not H.is_ambiguous(np.array([-0.10, -0.90, -0.95])))
    check("straddling zero does not fire",
          not H.is_ambiguous(np.array([0.30, -0.30])))

    print("\nrel_margin: does not depend on the worst control")
    #  the defect in the min-subtraction draft: one extreme low score must not
    #  change the verdict for the top two
    base = np.array([0.50, 0.49, 0.10])
    with_outlier = np.array([0.50, 0.49, -50.0])
    check("outlier leaves margin unchanged",
          approx(H.rel_margin(base), H.rel_margin(with_outlier)),
          f"{H.rel_margin(base)} vs {H.rel_margin(with_outlier)}")

    print("\nrel_margin: degenerate input")
    check("all equal fires", H.is_ambiguous(np.array([0.3, 0.3, 0.3])))
    check("all equal is exactly zero",
          approx(H.rel_margin(np.array([0.3, 0.3, 0.3])), 0.0))
    check("all zero fires", H.is_ambiguous(np.zeros(5)))
    check("single element is infinite",
          H.rel_margin(np.array([0.5])) == float("inf"))
    check("single element does not fire",
          not H.is_ambiguous(np.array([0.5])))
    check("empty is infinite", H.rel_margin(np.array([])) == float("inf"))

    print("\nrel_margin: non-finite input is rejected, not silently ranked")
    for bad, label in ((np.array([np.nan, 0.1]), "NaN"),
                       (np.array([np.inf, 0.1]), "inf"),
                       (np.array([0.5, -np.inf]), "-inf")):
        try:
            H.rel_margin(bad)
            check(f"{label} raises", False, "no exception")
        except ValueError:
            check(f"{label} raises", True)


# ------------------------------------------------------------ ordering
def test_ordering():
    print("\nstable_order: deterministic tie-break by control id")
    scores = np.array([0.5, 0.9, 0.5, 0.9])
    ids = np.array([103, 101, 102, 100])
    order = H.stable_order(scores, ids)
    got = [int(ids[i]) for i in order]
    check("ties break by ascending control id", got == [100, 101, 102, 103],
          f"got {got}")

    print("\nstable_order: invariant to input permutation")
    rng = np.random.default_rng(3)
    n = 200
    scores = rng.choice([0.0, 0.0, 0.0, 0.5, 1.0], size=n)  # many ties
    ids = np.arange(n)
    ref = [int(ids[i]) for i in H.stable_order(scores, ids)]
    for _ in range(5):
        p = rng.permutation(n)
        got = [int(ids[p][i]) for i in H.stable_order(scores[p], ids[p])]
        if got != ref:
            check("permuted input yields identical ranking", False)
            break
    else:
        check("permuted input yields identical ranking", True)


# -------------------------------------------------------------- fusion
def test_fusion():
    print("\nrrf_fuse: shape validation")
    ids = np.arange(4)
    try:
        H.rrf_fuse([np.zeros(4), np.zeros(3)], ids)
        check("mismatched backend lengths raise", False, "no exception")
    except ValueError:
        check("mismatched backend lengths raise", True)
    try:
        H.rrf_fuse([np.zeros(4)], np.arange(3))
        check("mismatched control_ids raise", False, "no exception")
    except ValueError:
        check("mismatched control_ids raise", True)

    print("\nrrf_fuse: known value")
    #  one backend, 3 items, ranks 1..3 -> 1/61, 1/62, 1/63
    f = H.rrf_fuse([np.array([3.0, 2.0, 1.0])], np.arange(3))
    check("single backend gives 1/(K+rank)",
          approx(f[0], 1 / 61) and approx(f[1], 1 / 62) and approx(f[2], 1 / 63),
          f"got {f}")

    print("\nrrf_fuse: determinism under permutation with ties")
    rng = np.random.default_rng(11)
    n = 120
    ids = np.arange(n)
    arrays = [rng.choice([0.0, 0.0, 0.25, 1.0], size=n) for _ in range(3)]
    base = H.rrf_fuse(arrays, ids)
    base_top = int(ids[H.stable_order(base, ids)[0]])
    ok = True
    for _ in range(5):
        p = rng.permutation(n)
        f = H.rrf_fuse([a[p] for a in arrays], ids[p])
        top = int(ids[p][H.stable_order(f, ids[p])[0]])
        ok &= (top == base_top)
    check("tied fusion top-1 is permutation invariant", ok)


# ---------------------------------------------------------------- arms
def test_arms():
    print("\nhybrid arms: gate routing")
    n = 50
    ids = np.arange(n)
    # NOTE: linspace(1.0, 0.0, 50) is NOT a confident vector. Its top-2 gap is
    # 1/49 = 0.0204, well under the 0.10 threshold, so the gate fires. An
    # earlier draft of this test used it and read the resulting failure as a
    # spec bug. The margin is between the top TWO scores only; the shape of
    # the tail is irrelevant.
    confident = np.concatenate([[1.0, 0.5], np.linspace(0.4, 0.0, n - 2)])
    ambiguous = np.concatenate([[0.90, 0.895], np.linspace(0.5, 0.0, n - 2)])
    lex = {k: np.linspace(0.0, 1.0, n) for k in H.LEXICAL_BACKENDS}

    s_conf = dict(lex, semantic=confident)
    out, fired = H.hybrid_gated(s_conf, ids)
    check("confident query is not gated", not fired)
    check("confident query returns the semantic vector unchanged",
          np.array_equal(out, confident))

    s_amb = dict(lex, semantic=ambiguous)
    out, fired = H.hybrid_gated(s_amb, ids)
    check("ambiguous query is gated", fired)
    check("gated query returns the four-way fusion",
          np.allclose(out, H.hybrid_equal(s_amb, ids)))

    out_fb, fired_fb = H.hybrid_gated_fallback(s_amb, ids)
    check("fallback variant gates on the same queries", fired_fb)
    check("fallback returns lexical-only fusion",
          np.allclose(out_fb, H.rrf_fuse(
              [s_amb[k] for k in H.LEXICAL_BACKENDS], ids)))

    print("\nhybrid arms: gate reads only the semantic vector")
    altered = dict(s_conf)
    altered["tfidf"] = np.zeros(n)
    _, fired_alt = H.hybrid_gated(altered, ids)
    check("changing a lexical backend cannot change routing",
          fired_alt is False)


# --------------------------------------------------------- gatekeeping
def test_gatekeeping():
    print("\nassert_gatekeeping: superiority is suppressed, not annotated")
    sup = {"mean_difference": 0.04, "p_value": 0.001}
    check("suppressed when NI fails",
          H.assert_gatekeeping({"p_lower": 0.40}, sup) is None)
    check("released when NI succeeds",
          H.assert_gatekeeping({"p_lower": 0.01}, sup) is sup)
    check("suppressed when NI p is exactly alpha",
          H.assert_gatekeeping({"p_lower": H.ALPHA}, sup) is None)
    check("suppressed when NI result is empty",
          H.assert_gatekeeping({}, sup) is None)


# ------------------------------------------------------------ metadata
def test_metadata():
    print("\nspec metadata: internal consistency")
    check("primary arm is declared PRIMARY",
          H.ARMS[H.PRIMARY_HYPOTHESIS["arm"]][1] == "PRIMARY")
    check("comparator is an end-to-end arm",
          H.ARMS[H.PRIMARY_HYPOTHESIS["comparator"]][0] == "end_to_end")
    check("cross-encoder is conditional",
          H.ARMS["reranker"][0] == "conditional")
    check("diagnostic corpus is not in the primary set",
          not set(H.ENGINEERED_CORPORA) & set(H.PRIMARY_CORPORA))
    check("primary corpora total 268 requirements is documented",
          "268" in H.PRIMARY_ESTIMAND)
    check("margin matches the precision note",
          H.PRECISION_NOTE["margin"] == H.DELTA_NI)
    check("precision slack is margin minus half-width",
          approx(H.PRECISION_NOTE["slack"],
                 H.DELTA_NI - H.PRECISION_NOTE["one_sided_95_half_width"],
                 tol=1e-4))
    check("motivating pairing uses the dual-encoder",
          H.MOTIVATING_COMPLEMENTARITY["primary_pairing"]["b"]
          == H.SEMANTIC_BACKEND)
    check("the two pairings have different unions",
          H.MOTIVATING_COMPLEMENTARITY["primary_pairing"]["union"] !=
          H.MOTIVATING_COMPLEMENTARITY["conditional_pairing"]["union"])

    print("\nspec metadata: the registered primary is not the known-bad setup")
    check("primary lexical backends use INDUCTIVE lsi",
          "lsi_inductive" in H.LEXICAL_BACKENDS)
    check("primary lexical backends exclude transductive lsi",
          "lsi" not in H.LEXICAL_BACKENDS)
    check("transductive variant survives only as a sensitivity arm",
          "lsi" in H.LEGACY_LEXICAL_BACKENDS
          and any(v is H.LEGACY_LEXICAL_BACKENDS
                  for v in H.SENSITIVITY_ARMS.values()))
    check("dual-encoder revision is pinned",
          len(H.MODEL_PIN["revision"]) == 40)
    check("dual-encoder weights are hashed",
          len(H.MODEL_PIN["weights_sha256"]) == 64)
    check("frozen inputs name a manifest",
          H.FROZEN_INPUTS["manifest"].endswith("manifest.json"))
    check("hybrid_equal is marked observed, not preregistered",
          H.ARMS["hybrid_equal"][1] == "observed")
    check("the pre-registration disclosure is in the docstring",
          "OBSERVED before registration" in H.__doc__
          or "WAS COMPUTED before" in H.__doc__)
    check("no reserved confirmatory corpus is claimed",
          not hasattr(H, "RESERVED_CONFIRMATORY_CORPUS"))
    check("confirmatory replication requires its own registration",
          "separate prospective registration" in H.CONFIRMATORY_REPLICATION)
    check("spec hash is a sha256 hex digest",
          len(H.spec_hash()) == 64 and all(c in "0123456789abcdef"
                                           for c in H.spec_hash()))


# ------------------------------------------------- stratified bootstrap
def test_bootstrap():
    print("\nstratified_bootstrap_ci")
    rng = np.random.default_rng(5)
    diffs = np.concatenate([rng.normal(0.10, 0.4, 106),
                            rng.normal(0.0, 0.4, 68),
                            rng.normal(0.0, 0.4, 94)])
    labels = np.array(["nist"] * 106 + ["hipaa"] * 68 + ["pf"] * 94)
    lo, hi, draws = H.stratified_bootstrap_ci(diffs, labels, n_boot=2000)
    check("interval brackets the point estimate",
          lo < diffs.mean() < hi, f"[{lo:.4f}, {hi:.4f}]")
    check("draw count matches", len(draws) == 2000)
    lo2, hi2, _ = H.stratified_bootstrap_ci(diffs, labels, n_boot=2000)
    check("seeded and reproducible", approx(lo, lo2) and approx(hi, hi2))
    try:
        H.stratified_bootstrap_ci(diffs, labels[:-1])
        check("misaligned labels raise", False, "no exception")
    except ValueError:
        check("misaligned labels raise", True)

    print("\nstratified_bootstrap_ci: composition is held fixed")
    # An earlier version of this test only asserted len(diffs) == 268, which
    # is a property of the input and would pass even if the function ignored
    # the strata entirely. Constant values per corpus make the check real: if
    # composition is held at 106/68/94 then EVERY replicate mean equals the
    # observed weighted mean exactly, and any cross-stratum resampling moves
    # it.
    const = np.concatenate([np.full(106, 1.0), np.full(68, 2.0),
                            np.full(94, 3.0)])
    expected = (106 * 1.0 + 68 * 2.0 + 94 * 3.0) / 268
    lo, hi, draws = H.stratified_bootstrap_ci(const, labels, n_boot=500)
    check("every stratified replicate reproduces the weighted mean",
          np.allclose(draws, expected),
          f"expected {expected:.6f}, draws span "
          f"[{draws.min():.6f}, {draws.max():.6f}]")
    check("degenerate interval collapses to the point",
          approx(lo, expected, 1e-9) and approx(hi, expected, 1e-9))

    # Control: an UNSTRATIFIED bootstrap on the same input must vary, which
    # shows the check above has power to detect the failure it targets.
    rng2 = np.random.default_rng(1)
    naive = np.array([const[rng2.integers(0, 268, 268)].mean()
                      for _ in range(200)])
    check("unstratified resampling would vary (test has power)",
          naive.std() > 1e-6)

    print("\nstratified_bootstrap_ci: interval coverage is labelled")
    check("docstring states the 90% two-sided reading",
          "90 percent TWO-SIDED" in H.stratified_bootstrap_ci.__doc__)


def test_against_score_all():
    print("\ncross-check against score_all.rrf_fuse (tie-free input)")
    check("formula and constant agree", H.verify_fusion_matches_score_all())


def main():
    test_gate()
    test_ordering()
    test_fusion()
    test_arms()
    test_gatekeeping()
    test_metadata()
    test_bootstrap()
    if "--with-score-all" in sys.argv:
        test_against_score_all()
    else:
        print("\n(skipping score_all cross-check; pass --with-score-all)")

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {', '.join(FAILURES)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

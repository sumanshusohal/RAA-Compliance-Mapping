#!/usr/bin/env python3
"""Outcome-blind precision analysis for the two confirmatory tests.

Run BEFORE the preregistration is timestamped. It answers one question: given
the sample sizes the confirmatory corpora actually have, what effect can the
HIPAA superiority test detect, and can the PF equivalence test fit inside the
proposed margin at all?

Discordance is estimated ONLY from the exploratory corpora (diagnostic, CSF),
never from HIPAA or PF outcomes, so running this does not burn the
confirmatory corpora.

The paired Top-1 difference per requirement takes values in {-1, 0, +1}, so
its standard deviation is driven by the discordance rate p (the share of
requirements where the two treatments disagree). For a paired difference with
discordance p split into p_plus and p_minus:

    Var = (p_plus + p_minus) - (p_plus - p_minus)^2

which under a null-ish split is approximately p. Half-widths below use the
observed discordance directly rather than that approximation.
"""
import math
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# Treatment contrast under test: reformulation vs multi-backend fusion.
BASELINE, TREATMENT = "multi", "reform"

# Exploratory sources for the discordance estimate.
EXPLORATORY = [
    ("diagnostic", os.path.join(HERE, "results_v3", "diagnostic")),
    ("csf/nist", os.path.join(HERE, "results_v3", "nist")),
]

# Confirmatory corpus sizes, from the built corpora.
CONFIRMATORY = [("HIPAA (superiority)", 68), ("PF (equivalence)", 94)]

DELTA = 0.05          # proposed equivalence margin, absolute Top-1
Z90 = 1.6448536269514722   # 90% CI, used for TOST
Z95 = 1.959963984540054    # 95% CI, used for the superiority estimate


def per_query_top1(directory, variant):
    """Mean Top-1 per unique requirement for one variant."""
    path = os.path.join(directory, f"perquery_{variant}.csv")
    df = pd.read_csv(path)
    return df.groupby("rid")["top1"].mean()


def discordance(directory):
    """Observed paired-difference statistics on an exploratory corpus."""
    a = per_query_top1(directory, BASELINE)
    b = per_query_top1(directory, TREATMENT)
    shared = sorted(set(a.index) & set(b.index))
    diff = (b.loc[shared] - a.loc[shared]).to_numpy()
    n = len(diff)
    nonzero = float((abs(diff) > 1e-9).sum()) / n
    return {
        "n": n,
        "discordance": nonzero,
        "mean_diff": float(diff.mean()),
        "sd_diff": float(diff.std(ddof=1)),
    }


def half_width(sd, n, z):
    return z * sd / math.sqrt(n)


def main():
    print("=" * 74)
    print("OUTCOME-BLIND PRECISION ANALYSIS")
    print("Discordance estimated from exploratory corpora only.")
    print("=" * 74)

    print(f"\nContrast: {TREATMENT} vs {BASELINE} (paired, per requirement)\n")
    stats = []
    for label, directory in EXPLORATORY:
        try:
            s = discordance(directory)
        except FileNotFoundError:
            print(f"  {label:<12} results not found, skipped")
            continue
        stats.append(s)
        print(f"  {label:<12} n={s['n']:<4} discordance={s['discordance']:.3f}  "
              f"sd(diff)={s['sd_diff']:.3f}  mean(diff)={s['mean_diff']:+.3f}")

    if not stats:
        print("\nNo exploratory results available; cannot proceed.")
        return 1

    # Conservative planning value: the largest observed spread.
    sd_plan = max(s["sd_diff"] for s in stats)
    sd_low = min(s["sd_diff"] for s in stats)
    print(f"\nPlanning sd(diff): {sd_plan:.3f} (conservative, max observed); "
          f"optimistic {sd_low:.3f}")

    print("\n" + "-" * 74)
    print("ACHIEVABLE PRECISION AT CONFIRMATORY SAMPLE SIZES")
    print("-" * 74)
    for label, n in CONFIRMATORY:
        hw95 = half_width(sd_plan, n, Z95)
        hw90 = half_width(sd_plan, n, Z90)
        hw90_opt = half_width(sd_low, n, Z90)
        print(f"\n{label}, n={n}")
        print(f"  95% CI half-width (superiority) : {hw95:.3f}"
              f"   -> smallest reliably detectable effect ~{hw95:.2f} Top-1")
        print(f"  90% CI half-width (TOST)        : {hw90:.3f} "
              f"(optimistic {hw90_opt:.3f})")
        verdict = ("FEASIBLE" if hw90 < DELTA else
                   "NOT FEASIBLE" if hw90_opt >= DELTA else "MARGINAL")
        print(f"  equivalence at delta={DELTA}          : {verdict}")
        if hw90 >= DELTA:
            need = math.ceil((Z90 * sd_plan / DELTA) ** 2)
            print(f"     n required for delta={DELTA}: ~{need} "
                  f"({need - n} more than available)")
            print(f"     smallest supportable delta at n={n}: {hw90:.3f}")

    print("\n" + "=" * 74)
    print("DECISION REQUIRED BEFORE PREREGISTRATION")
    print("=" * 74)
    hw90_pf = half_width(sd_plan, 94, Z90)
    if hw90_pf >= DELTA:
        print(f"""
The PF equivalence test cannot support delta={DELTA} at n=94: the 90% CI
half-width is {hw90_pf:.3f} even before any effect is observed, so the
interval cannot fit inside (-{DELTA}, +{DELTA}). Pre-commit to ONE of:

  (1) keep delta={DELTA} and report whatever verdict the CI supports,
      accepting that "inconclusive" is the likely and honest outcome;
  (2) widen delta to {hw90_pf:.2f} (the smallest the design can support) and
      justify it operationally, noting it is looser than preferred;
  (3) declare the PF analysis descriptive rather than confirmatory, resting
      confirmatory weight on the HIPAA superiority test and the continuous
      moderation model.

Recommendation: (1). It is honest, requires no post-hoc margin change, and an
inconclusive equivalence result is itself a reportable finding about what
this corpus size can and cannot establish.""")
    else:
        print(f"\nPF equivalence at delta={DELTA} is feasible "
              f"(half-width {hw90_pf:.3f}). No margin change needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# OSF Open-Ended Registration: text to submit

Paste the block below into the Summary field of an OSF Open-Ended
Registration. Attach `hybrid_spec.py` and `test_hybrid_spec.py` to the OSF
project first, so the registration freezes the file content itself and does
not depend on the GitHub repository remaining available or unaltered.

Fill the two bracketed values before submitting. Consider registering with an
embargo, which records the timestamp while keeping the content private until
the paper is out.

---

## Title

Preregistered analysis specification: gated hybrid lexical-semantic retrieval
for regulatory traceability

## Summary

**What this registers.** A frozen analysis specification for one comparison
that has not yet been run. The specification is the file `hybrid_spec.py`,
whose content is fixed at:

    SHA-256  4e454e79d12756d4d38eb29d53cca662b5062ebb013d62d15478c84c362685b5
    commit   [FILL: full 40-character commit SHA]
    repo     [FILL: public repository URL]

The hash is computed over the file with line endings normalized to LF, so it
reproduces on any platform. Running `python hybrid_spec.py` prints it.

**Status, stated plainly.** This is a preregistration of the analysis, not a
claim of confirmatory status for the result it will produce. The gate design
tested here was motivated by complementarity we had already observed in the
same four corpora: on the Privacy Framework corpus, five methods tie at 35 of
94 requirements while lexical rank fusion and a dual-encoder each uniquely
solve 15 the other misses. The design is therefore frozen before the outcome,
but the hypothesis was generated from the data that will test it. On these
four corpora the result is preregistered and exploratory. Confirmatory status
is available only by applying this specification, unchanged and at this hash,
to a corpus that does not yet exist. The specification reserves a slot for
that corpus and does not permit the exploratory run to be relabelled.

**Question.** Does routing a query to lexical retrieval only when the
dual-encoder is uncertain recover the requirements lexical fusion uniquely
solves, without giving up those the dual-encoder already solves?

**Primary arm.** `hybrid_gated`. A dual-encoder ranks the full control
corpus. Only when its own top-2 relative margin falls below 0.10 is the
lexical side consulted, and the query is then re-ranked by reciprocal rank
fusion over TF-IDF, BM25, LSI and the dual-encoder. The gate reads only the
dual-encoder scores, so routing is decided before any label is touched.

**Neither constant is new.** The 0.10 gate threshold is the relative top-2
margin the existing agent already uses to trigger query reformulation. The
0.05 margin below is the equivalence margin already fixed for an earlier,
unrelated contrast in the same project. Both predate this specification, so
neither was chosen with this outcome in view.

**Comparator.** `semantic`, the dual-encoder alone.

**Endpoint.** Top-1 accuracy, scored once per requirement over the full
control corpus. No repeated holdout, no seed sweep.

**Estimand.** The mean paired Top-1 difference over the 268 concatenated
requirements of the three real corpora: NIST CSF 1.1 (106), HIPAA (68) and the
NIST Privacy Framework (94). Requirement-weighted, so corpora are not equally
weighted and CSF contributes 40 percent of the estimate. Per-corpus results
are reported descriptively. Intervals come from a bootstrap resampled within
corpus.

**The engineered corpus is excluded from the primary estimate.** A fourth,
author-constructed diagnostic corpus is analysed and reported separately and
is never pooled into the primary estimand.

**Tests, in this fixed order.**

1. Non-inferiority. Null: the mean difference is at or below -0.05. Tested by
   the lower one-sided arm of a two one-sided tests procedure. Rejected at
   p < 0.05.
2. Superiority, tested only if test 1 rejects. Null: the mean difference is
   zero. Tested by a paired sign-flip permutation test. Rejected at p < 0.05
   with a positive mean difference.

If test 1 does not reject, the superiority result is suppressed entirely and
not reported as a finding. This gatekeeping is enforced in code, not by a
footnote.

**Precision, computed before running and outcome-blind.** Estimated from the
engineered corpus only, which is excluded from the primary estimate.
Discordance 0.241, standard deviation of the paired difference 0.476, standard
error 0.0291 at n = 268, one-sided 95 percent half-width 0.0478 against a 0.05
margin. Non-inferiority is therefore establishable only if the true difference
is at least -0.0022. This design can confirm "no cost". It cannot confirm "a
cost smaller than five points". A failure to establish non-inferiority may be a
power limit rather than evidence of harm, and will be reported as such.

**Pre-declared secondary analyses.** An equal-weight four-backend fusion
without a gate; a hard-fallback variant of the gate that hands the query to
lexical fusion instead of fusing; the same contrasts against lexical fusion;
per-corpus breakdowns; gate firing rates with accuracy among fired and unfired
queries; the engineered corpus, labelled as such. The hard-fallback variant is
declared secondary here, before running, so that it cannot be presented later
as the design that was intended all along.

**Stopping rule.** Every arm is scored once, in a single pass, over the full
corpus. No interim analysis. If an arm errors, the fix is committed and all
arms are re-run together from one commit.

**Explicitly prohibited.** Adding a backend after seeing any arm's accuracy;
moving the gate threshold off 0.10 or the margin off 0.05; promoting the
secondary variant to primary; pooling the engineered corpus into the primary
estimate; reporting end-to-end and candidate-constrained methods in one table;
reporting the superiority p-value when non-inferiority did not reject.

**Verification.** `python hybrid_spec.py` prints the hash above.
`python test_hybrid_spec.py` runs the specification's own test suite, which
covers the gate statistic on positive, negative, tied, single-element, empty
and non-finite inputs, deterministic tie-breaking, fusion behaviour, gate
routing, gatekeeping suppression, and internal consistency of the declared
metadata.

---

## After registering

1. Record the registration DOI and date in `HANDOFF.md`.
2. Optionally add an OpenTimestamps proof over the same file as a second,
   purely cryptographic timestamp. It is cheap and independent of OSF, but it
   is not a substitute: reviewers recognize a registration, and few will
   verify a Bitcoin attestation.
3. Only then run the arms.

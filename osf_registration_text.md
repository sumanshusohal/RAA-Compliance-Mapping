# OSF Open-Ended Registration: text to submit

Paste the block below into the Summary field of an OSF Open-Ended
Registration.

**Attach everything listed under "What this registers", including the 32 .npy
arrays in `frozen_backends/` (3.6 MB) and `confirmatory_stats.py`.** A
manifest freezes hashes, not availability: without the arrays and the imported
statistics module, the claim that the registration does not depend on GitHub
is false, because nobody could re-run the analysis from the registration
alone. A deterministic ZIP of `frozen_backends/` is an acceptable substitute
if its own SHA-256 is registered.

Fill the two bracketed values before submitting.

An embargo adds little here: the repository and the specification are public
already, so the registration is not revealing anything the repo does not.

---

## Title

Prospectively registered exploratory analysis: gated hybrid lexical-semantic
retrieval for regulatory traceability

## Summary

**What this registers.** A frozen analysis specification, its runner, its
tests, its input generator, and the inputs themselves, for a comparison whose
primary arm has not been computed.

    hybrid_spec.py                 a1b4c78966321f37bf17fc019333363f1e819a552c5ea60dfbd39e5be190597a
    run_hybrid.py                  b33459d9b132277a2c73c2c7f4e2be91b17648448d1b08e46b468e492866fc8f
    test_hybrid_spec.py            d0336d6c4c185d21ec782c956e6c1b57f0ba51395acc909e2c1cf72c9e07b827
    freeze_backends.py             b20d326ed3576b1c4f118cc347d4e9e35144fa2371517f1906b5433c37741482
    confirmatory_stats.py          79ec294a77c12c79baf0df6187315affa521d8585f520b273050b67f27e11040
    frozen_backends/manifest.json  93259026e5cc90998a2ab182a21b95660c36b67f20907a3c38cece128913fa0c

    frozen_backends/  32 .npy arrays, 3.6 MB, individually hashed in the
                      manifest above: one score matrix per backend per
                      corpus, plus control ids, requirement ids and a gold
                      membership mask

    commit  25ce391d5e00bbb8218cf46878c256c88b2655f0
    tag     hybrid-analysis-spec-v1
    repo    https://github.com/sumanshusohal/RAA-Compliance-Mapping

Cite the tag rather than the commit where possible. The tag is annotated
and carries these hashes in its own message, and it stays fixed at the
frozen state even as later commits land on master. The specification's
SHA-256 is content-addressed and is the strongest anchor of the three:
it does not depend on git at all.

Text hashes are computed with line endings normalized to LF, so they reproduce
on any platform. `python hybrid_spec.py` prints the specification hash.
Array hashes cover the raw bytes with dtype and shape and are never
normalized.

**The analysis does not load a model.** `freeze_backends.py` runs every
retrieval backend once, writes the raw score matrices, and hashes them
alongside the model revision and library versions. `run_hybrid.py` consumes
only those matrices, verifies every array hash against the manifest, and
aborts on any mismatch. The registered path is arithmetic over hashed inputs.

The dual-encoder is pinned to sentence-transformers/all-MiniLM-L6-v2 at
revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41, and the generator passes
that revision explicitly rather than resolving the bare model name. The
matrices were regenerated from a clean commit under the enforced pin and all
32 array hashes were byte-identical to those produced before it, which
confirms the unpinned path had been resolving to the same checkpoint. The
weights and tokenizer file hashes are recorded in the specification.

**Status.** Exploratory. This registration does not reserve confirmatory
status for itself. The gate was motivated by complementarity we had already
observed in these corpora: on the Privacy Framework corpus five methods tie at
35 of 94 requirements while lexical fusion and the dual-encoder each uniquely
solve 15 the other misses. The design is frozen before the outcome, but the
hypothesis was generated from the data that will test it. A future
confirmatory replication would require a separate prospective registration
identifying a new corpus and its construction, carrying the gate, fusion,
endpoint, margin and test order below over unchanged.

**Disclosure, recorded before registering.** One declared secondary arm,
`hybrid_equal`, was computed during method development, before this
registration. It was produced as a byproduct of a tie-breaking sensitivity
check that did not require it, and the values were seen: Top-1 0.4623 (CSF),
0.3676 (HIPAA), 0.3936 (PF), 0.5517 (diagnostic).

That arm is therefore an observed result, not a preregistered one. It is
retained, reported descriptively, and excluded from preregistered inferential
claims. It is not removed: it is part of the primary arm's own mechanism,
since the gated arm applies exactly this fusion to gated queries, and deleting
an observed result would conceal that.

The primary arm has not been computed, and no contrast, test, or gate firing
rate has been produced. Because the primary arm returns the dual-encoder
ranking on ungated queries and this fusion on gated ones, knowing
`hybrid_equal` constrains the primary result without determining it: the
gate's firing pattern, and therefore the mixture and the contrast, remain
unknown. The primary claim is weakened by this disclosure rather than
unaffected by it.

The repository history is unsquashed and shows that the gated arm and its
0.10 threshold were committed before the secondary outcome was observed.

**Question.** Does routing a query to lexical retrieval only when the
dual-encoder is uncertain recover the requirements lexical fusion uniquely
solves, without giving up those the dual-encoder already solves?

**Primary arm.** `hybrid_gated`. A dual-encoder ranks the full control corpus.
Only when its own top-2 relative margin falls below 0.10 is the lexical side
consulted, and the query is then re-ranked by reciprocal rank fusion over
TF-IDF, BM25, inductive LSI and the dual-encoder. The gate reads only the
dual-encoder scores, so routing is decided before any label is touched.

The LSI component is fitted on control documents only. A transductive fit on
controls plus all evaluation queries is available as a declared sensitivity
arm and is never primary, because that fit is known in this project to reduce
accuracy.

**Neither constant is new.** The 0.10 gate threshold is the relative top-2
margin the existing agent already uses to trigger query reformulation. The
0.05 margin is the equivalence margin already fixed for an earlier, unrelated
contrast in the same project. Both predate this specification.

**Comparator.** `semantic`, the dual-encoder alone.

**Endpoint.** Top-1 accuracy, scored once per requirement over the full
control corpus. No repeated holdout, no seed sweep.

**Estimand.** The mean paired Top-1 difference over the 268 concatenated
requirements of the three real corpora: NIST CSF 1.1 (106), HIPAA (68) and the
NIST Privacy Framework (94). Requirement-weighted, so corpora are not equally
weighted and CSF contributes 40 percent of the estimate. Per-corpus results
are reported descriptively. Intervals come from a bootstrap resampled within
corpus, reported as its 5th and 95th percentiles, a 90 percent two-sided
interval equivalently a 95 percent one-sided bound, matching the one-sided
test below.

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
not reported as a finding. This is enforced in code, not by a footnote.

**Margin justification.** Over the pooled 268 requirements, five absolute
Top-1 points is about 13 requirements whose top suggestion is wrong. Whether
relevant controls survive further down the shortlist is not established by a
Top-1 loss and is reported separately through MRR@5 and Recall@5, whichever
way the primary test lands.

**Precision, computed before running and outcome-blind.** Estimated from the
engineered corpus only, which is excluded from the primary estimate.
Discordance 0.241, standard deviation of the paired difference 0.476, standard
error 0.0291 at n = 268, one-sided 95 percent half-width 0.0478 against a 0.05
margin. Non-inferiority is therefore establishable only if the true difference
is at least -0.0022. This design can confirm "no cost". It cannot confirm "a
cost smaller than five points". A failure to establish non-inferiority may be a
power limit rather than evidence of harm, and will be reported as such.

**Pre-declared secondary analyses.** A hard-fallback variant of the gate that
hands the query to lexical fusion instead of fusing; the same contrasts
against lexical fusion; the transductive-LSI sensitivity arm; per-corpus
breakdowns; gate firing rates with accuracy among fired and unfired queries;
the engineered corpus, labelled as such. The hard-fallback variant is declared
secondary here, before running, so it cannot be presented later as the design
that was intended all along.

**Stopping rule.** Every arm is scored once, in a single pass, over the full
corpus. No interim analysis. If an arm errors, the fix is committed and all
arms are re-run together from one commit.

**Explicitly prohibited.** Adding a backend after seeing any arm's accuracy;
moving the gate threshold off 0.10 or the margin off 0.05; promoting a
secondary or sensitivity arm to primary; pooling the engineered corpus into
the primary estimate; reporting end-to-end and candidate-constrained methods
in one table; reporting the superiority p-value when non-inferiority did not
reject.

**Verification.** `python hybrid_spec.py` prints the specification hash.
`python test_hybrid_spec.py --with-score-all` runs the test suite, covering
the gate statistic on positive, negative, tied, single-element, empty and
non-finite inputs, its reduction to the pre-existing agent formula on
non-negative input, deterministic tie-breaking, fusion behaviour, gate
routing, gatekeeping suppression, stratified resampling, and internal
consistency of the declared metadata. `python run_hybrid.py --self-test`
exercises the full analysis path on synthetic fixtures.
`python run_hybrid.py --verify-only` checks the frozen inputs against the
manifest without computing anything.

---

## After registering

1. Record the registration DOI and date in `HANDOFF.md`.
2. Optionally add an OpenTimestamps proof over `hybrid_spec.py` as a second,
   purely cryptographic timestamp. It is independent of OSF, but it is not a
   substitute: it proves only that a file existed, not what was intended with
   it, and few reviewers will verify a Bitcoin attestation.
3. Only then run `python run_hybrid.py`.

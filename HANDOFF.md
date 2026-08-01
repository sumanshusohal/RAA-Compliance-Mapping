# Handoff: exploratory checkpoint

State as of this commit. Read this before continuing the work.

## What this snapshot is, and is not

It is an **exploratory checkpoint**, not a preregistration. The intended
sequence was to freeze the analysis spec, timestamp it publicly, and only
then compute confirmatory results. That did not happen: the HIPAA and PF
ablations were run before anything was timestamped, and those ablations
compute the `reform` vs `multi` contrast, which is the confirmatory test
itself. There is therefore no evidence that the spec predated the outcomes.

**Consequence: all four corpora are exploratory.** Confirmatory status can
only be regained on a corpus that does not yet exist, with a spec timestamped
first.

## Corpora

| Corpus | Role | Reqs | Links | Notes |
|---|---|---|---|---|
| CSF 1.1 | exploratory | 106 | 495 | reproduces published figures exactly |
| HIPAA | exploratory | 68 | 274 | regulation-to-control, statutory source text |
| PF | exploratory | 94 | 456 | narrow gap, NIST-authored |
| diagnostic | sanity only | 58 | 86 | author-built, never confirmatory |

All share one 300-control SP 800-53 5.2.0 corpus (OSCAL tag `v1.5.0`).
Zero unresolved control references anywhere. Every builder writes
`provenance.json`; the HIPAA builder also snapshots and hashes all six raw
CPRT responses under `cprt_snapshots/`.

CSF stays at 1.1 deliberately: CPRT documents a near 1:1 derivation of CSF
2.0 subcategories from CSF 1.1, so 2.0 cannot be an independent corpus.

## Headline findings

**The reformulation effect does not replicate on real text.** +0.140 on the
engineered diagnostic corpus versus +0.009 (CSF), +0.008 (HIPAA), +0.003 (PF).
HIPAA has the largest measured gap and produces roughly a seventh of the
diagnostic effect. Engineered-versus-real separates the data; measured gap
does not.

**Lexical distance predicts lexical failure, but not semantic advantage.**
Spearman gap vs TF-IDF correctness: −0.754, −0.359, −0.272, −0.475, all
p < 0.05. Gap vs dual-encoder advantage over TF-IDF: +0.335, −0.082, −0.143,
+0.302, inconsistent and two not distinguishable from zero.

**Lexical RRF trails semantic retrieval on every real corpus** by 0.141
(CSF), 0.147 (HIPAA, after the LSI fix), 0.085 (PF). It does beat its own
lexical components on CSF and HIPAA, so fusion is not failing; it is fusing
only weak representations.

**Aggregate ties conceal complementary errors.** On PF five methods each
score exactly 35/94 while TF-IDF and the dual-encoder each uniquely solve 15
requirements the other misses. Oracle union of lexical RRF and the
cross-encoder reaches 53/94. This motivates a **gated** hybrid, not an
equal-weight one.

**The transductive LSI fit was hurting, not helping.** Fitting on controls
only improves LSI by +0.015 to +0.019 on three corpora and lifts lexical RRF
on HIPAA from 0.279 to 0.309. The published protocol fitted LSI on controls
plus evaluation queries (`include_regs_in_fit=True`) and never said so.

## Claims that were made and RETRACTED. Do not reintroduce.

- "LLM wins three of four corpora" — wins 1, ties 1, loses 2 on a common
  estimand.
- "HIPAA is the widest-gap corpus" — IDF is fitted per corpus, so gap means
  are not on a common scale.
- "The gap metric is validated, 5.8x separation" — face validity against
  author-assigned labels on the author-built corpus only.
- "Engineered benchmarks inflate gains ~7x" — causal overreach; the three
  real corpora are correlated replications, all NIST, all to SP 800-53.
- "RRF never beats its own best component" — false, it does on CSF and HIPAA.
- "PF is degenerate" — disproven; the equal means conceal different successes.
- "Poor accuracy supports the qualitative-value conclusion" — it refutes an
  accuracy claim and establishes nothing about provenance.
- "No method's ordering across corpora tracks the gap measure" — four corpus
  averages on a corpus-relative scale cannot support this.

## Protocol rules that must hold

1. **One estimand.** Score every frozen method once per requirement over the
   full corpus (`score_all.py`). The old repeated-holdout design evaluates
   requirements unevenly (HIPAA: 2 to 12 times across 30 seeds); pooling
   those rows and comparing against a one-pass LLM mixes estimands.
2. **Never mix protocols in one table.** End-to-end methods rank the whole
   corpus. Conditional rerankers (cross-encoder, LLM) see only the
   dual-encoder top-20 and are bounded by its Recall@20 ceiling: 0.983
   diagnostic, 0.906 CSF, 0.882 HIPAA, 0.840 PF.
3. **`rrf_lexical` is not RAA.** RAA adds conditional reformulation and
   decomposition, which can reorder. Do not label it RAA until the full
   pipeline is regenerated on the shared protocol.
4. **NIST crosswalks are a silver standard.** NIST calls them concept
   relationship mappings and marks the HIPAA OLIR "Comprehensive: No". Say
   "supports" or "addresses", not "implements". An unlisted plausible
   prediction is not automatically a false positive.

## Pending work, in priority order

1. ~~**Statistics**~~ **DONE.** `moderation_test` now derives its p-value
   from a Freedman-Lane sign-flip permutation test (100,000 sign vectors,
   seed 20260801) instead of the bootstrap-crossing-zero rule; the bootstrap
   is kept only for the CI. `sign_flip_test` added for the superiority
   contrast. `tost()` is now real two one-sided tests returning `p_lower`,
   `p_upper`, `p_tost`; the old CI-inclusion check is renamed
   `equivalence_verdict` and marked descriptive. `run_confirmatory.py`
   regenerates every number into a JSON record under
   `results_v3/confirmatory/` carrying argv, UTC timestamp, git commit and
   dirty flag, library versions, module and input-CSV SHA-256 hashes, and
   seeds.

   **The "+0.0043, p=0.82" figure does not reproduce and is retired.** On the
   three real corpora the pooled moderation coefficient is **+0.0137, 95% CI
   [-0.0139, +0.0434], sign-flip p=0.376** (n=268). Adding the engineered
   diagnostic corpus moves it to **+0.0236, p=0.089** (n=326), which is
   itself evidence for the headline finding: the apparent moderation is
   carried by the engineered corpus. Neither matches the scratchpad value
   under any corpus set tried, so the old number has no recoverable
   provenance. Do not cite it, and do not treat the new values as a
   correction of it.

   Two caveats to carry into the writeup. Both TOST and the moderation model
   run on per-requirement means over an uneven number of seeds (protocol rule
   1), so the per-requirement variances are not equal and the t
   approximation is approximate; and PF and CSF both return
   `equivalent=True` at δ=0.05 while HIPAA does not (p_tost=0.114), so the
   superiority prediction is neither supported nor excluded there.
2. ~~**Hybrid spec**~~ **DRAFTED, NOT YET TIMESTAMPED.** `hybrid_spec.py`
   freezes the design: equal-weight semantic RRF and a semantic-primary gated
   variant, against semantic alone, lexical RRF, full RAA, and the
   reformulation-off control. Primary arm `hybrid_gated`, primary endpoint
   Top-1, non-inferiority tested before superiority at δ=0.05, which is the
   margin already fixed for the reform-vs-multi contrast rather than a new
   one. The gate reuses RAA's existing relative top-2 margin at 0.10, so it
   adds no tuned constant.

   **This item moved ahead of full RAA deliberately.** Full RAA is an ARM of
   the hybrid comparison, so running it first and writing the spec afterwards
   would set the gate and the margin with one arm's results already visible.
   That is the failure at the top of this file, in miniature.

   Two things to carry forward. The gap metric CANNOT be the gate:
   `compute_gaps` maximizes overlap over the gold set, so it is not computable
   at query time and stays the RQ2 moderator. And freezing now does not buy
   confirmatory status on these four corpora, because the gate is motivated by
   complementarity already observed on them; the result is preregistered but
   still exploratory, and confirmatory status needs this file applied
   unchanged to a corpus that does not exist yet.

   Remaining before any arm runs: a PUBLIC timestamp of the spec hash. A local
   commit is not one.
3. **Full RAA on the shared protocol**, so the pipeline can be labelled
   honestly. Runs only after the spec above is timestamped.
4. **Corpus sensitivities**: 800-53 5.1.1 as the HIPAA-native target
   (5.2.0 differs by one control, SA-24, and all 1,241 links resolve in
   both); enhancement collapsing; a `vocab_regime` field for the diagnostic
   corpus.

   ~~66 vs 59 vocabulary-matched positives~~ **RESOLVED.** The released data
   is authoritative and the docs were wrong. `match_type` is the vocabulary
   regime label: `good` averages 0.043 IDF-weighted overlap with its
   requirement against 0.250 for `perfect` (medians 0.000 and 0.197), and the
   control the manuscript cites as its mismatch example is labelled `good`.
   Corrected to 59 / 27 / 20 / 4 in manuscript3_revised.tex, SOURCES.md and
   README.md. The results claim "23% of positive links (20/86)" was wrong in
   the understating direction and is now 41% (35/86), because 14 matched
   positives also share no content words. `dke_manuscript2.tex` is left
   uncorrected on purpose: it is the version that was submitted to DKE.
5. **LLM**: candidate-order sensitivity (candidates are currently supplied in
   first-stage rank order), pinned Hugging Face model and tokenizer
   revisions, a strict shared validator rejecting duplicate IDs and
   out-of-range confidence, a contamination check by masking framework
   identifiers, and one batched API pass. Confidence is a constant 1.0 from
   the local model, so calibration is a precondition for any AURC result.

## Framing

Recommended identity, stable regardless of how the hybrid turns out:

> Benchmark Sensitivity in Regulatory Traceability: A Multi-Corpus Study of
> Query Reformulation, Neural Reranking, and LLMs

Do not make the paper's identity contingent on the hybrid winning. Do not
claim "neural accuracy without giving up determinism and auditability":
logging establishes provenance, not audit usefulness, and determinism must
be measured in a locked environment. Audit usefulness needs a human study.

## Running things

```
USE_TF=0 python score_all.py                 # shared-population ranking table
USE_TF=0 python run_local_all.py             # open-weight LLM, all corpora
python run_confirmatory.py                   # RQ2 moderation + TOST, recorded
python precision_analysis.py                 # outcome-blind power analysis
python gap_metrics.py                        # gap distributions + spec hash
python scan_secrets.py                       # before every push
```

Credentials come from `ANTHROPIC_API_KEY` in the environment only. Never a
flag, a literal, or a notebook cell.

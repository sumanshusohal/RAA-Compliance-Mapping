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
first. The one exception is the gated hybrid of item 2, which is
preregistered but, because its hypothesis was generated from these corpora,
still exploratory.

## Corpora

| Corpus | Role | Reqs | Links | Notes |
|---|---|---|---|---|
| CSF 1.1 | exploratory | 106 | 495 | reproduces earlier figures exactly |
| HIPAA | exploratory | 68 | 274 | regulation-to-control, statutory source text |
| PF | exploratory | 94 | 456 | NIST-authored |
| diagnostic | sanity only | 58 | 86 | author-built, never confirmatory |

All share one 300-control SP 800-53 5.2.0 corpus (OSCAL tag `v1.5.0`).
Zero unresolved control references anywhere. Every builder writes
`provenance.json`; the HIPAA builder also snapshots and hashes all six raw
CPRT responses under `cprt_snapshots/`.

CSF stays at 1.1 deliberately: CPRT documents a near 1:1 derivation of CSF
2.0 subcategories from CSF 1.1, so 2.0 cannot be an independent corpus.

## Headline findings

All ranking figures below are the **primary protocol**: one scoring pass per
requirement over the full control corpus, LSI fitted on control documents
only. Record: `results_v3/shared/reform_vs_multi_onepass.json`.

**The reformulation effect did not generalize.** +0.121 on the diagnostic
corpus (p=0.039) against +0.028 (CSF), 0.000 (HIPAA), +0.032 (PF), none
distinguishable from zero. Conditional engineered-vs-external contrast
+0.098, bootstrap 95% CI [+0.001, +0.206]; the lower bound sits on zero.

**Each estimate turns on very few requirements.** Wins/losses/ties:
diagnostic 8/1/49, CSF 4/1/101, HIPAA 2/2/64, PF 4/1/89. Reformulation
reorders the top-1 far more often than it changes correctness: 15, 14, 10
and 18 identity changes against 9, 5, 4 and 5 correctness changes.

**The evaluation protocol contributes exactly nothing; the LSI fit does.**
Three-cell factorial, `results_v3/shared/holdout_*.json`:

| corpus | A hold/ctrl | B hold/train | C 1pass/ctrl | LSI (B−A) | protocol (C−A) |
|---|---|---|---|---|---|
| diagnostic | +0.1207 | +0.1405 | +0.1207 | +0.0198 | 0.0000 |
| CSF | +0.0283 | +0.0088 | +0.0283 | −0.0195 | 0.0000 |
| HIPAA | 0.0000 | +0.0078 | 0.0000 | +0.0078 | 0.0000 |
| PF | +0.0319 | +0.0027 | +0.0319 | −0.0292 | 0.0000 |

A=C is close to an identity, not a discovery: with a controls-only fit
nothing in the pipeline depends on the split, so the mean over splits equals
the single-pass value. Cell B reproduces the previously reported figures and
their p-values to four decimals, which is the check that the factorial
reimplementation is faithful. The LSI column has no consistent sign and its
magnitude on the external corpora is comparable to the effect itself.

**Lexical distance predicts lexical failure, but not semantic advantage.**
Spearman gap vs TF-IDF correctness: −0.754, −0.359, −0.272, −0.475, all
p < 0.05. Gap vs dual-encoder advantage over TF-IDF: +0.335, −0.082, −0.143,
+0.302, inconsistent and two not distinguishable from zero.

**Lexical RRF trails semantic retrieval on every external corpus** by 0.141
(CSF), 0.147 (HIPAA, after the LSI fix), 0.085 (PF). It does beat its own
lexical components on CSF and HIPAA, so fusion is not failing; it is fusing
only weak representations.

**Aggregate ties conceal complementary errors.** On PF five methods each
score exactly 35/94 while lexical fusion and the dual-encoder each uniquely
solve 15 requirements the other misses. Oracle union of lexical RRF and the
cross-encoder reaches 53/94. This motivates a **gated** hybrid, not an
equal-weight one.

**The gated hybrid is non-inferior with no evidence of superiority.**
Registered primary: +0.022, p_NI=0.003, p_sup=0.48, n=268. Per corpus
+0.085 (CSF), −0.088 (HIPAA), +0.032 (PF), descriptive by prior declaration.

## Claims that were made and RETRACTED. Do not reintroduce.

- "LLM wins three of four corpora" — wins 1, ties 1, loses 2 on a common
  estimand.
- "HIPAA is the widest-gap corpus" — IDF is fitted per corpus, so gap means
  are not on a common scale. This one keeps coming back; the defensible
  statement is that HIPAA is the only corpus whose two sides come from
  different institutions, which is qualitative.
- "The gap metric is validated, 5.8x separation" — face validity against
  author-assigned labels on the author-built corpus only.
- "Engineered benchmarks inflate gains ~7x" — causal overreach; the three
  external corpora are correlated replications, all NIST, all to SP 800-53.
- "RRF never beats its own best component" — false, it does on CSF and HIPAA.
- "PF is degenerate" — disproven; the equal means conceal different successes.
- "Poor accuracy supports the qualitative-value conclusion" — it refutes an
  accuracy claim and establishes nothing about provenance.
- "No method's ordering across corpora tracks the gap measure" — four corpus
  averages on a corpus-relative scale cannot support this.
- "Three independently authored corpora" — externally authored, not mutually
  independent. Shared NIST lineage, one target catalogue.
- "CSF and PF are statistically equivalent to no effect" — true only in the
  repeated-holdout regime. Under the primary protocol p_TOST is 0.152 and
  0.224; only HIPAA meets the margin, at 0.048, which is boundary-sensitive.
- "The exact sign test confirms equivalence" — it tests superiority. No
  exact equivalence test exists at these sample sizes.
- "The transductive one-pass arm isolates the LSI fit against the holdout
  figures" — it does not; it fits on ALL requirements including the scored
  ones, whereas the holdout regime excludes them. Use the factorial.
- "Reformulation is always on" — the *gate* fires on nearly everything; an
  expansion is applied less often and changes the ranking less often still.
  Report `gate_fired`, `expanded` and `top1_changed` separately.
- "Measurement fragility" as the paper's frame — overreaches from four
  corpora to the field. The frame is cross-corpus evaluation.

## Protocol rules that must hold

1. **One estimand per comparison.** Ranking comes from one pass per
   requirement over the full corpus. Decision metrics (coverage, selective
   accuracy, gap detection) need a calibrated threshold and therefore come
   from repeated holdouts. Never mix the two in one comparison; put the
   protocol in every table caption.
2. **Never mix retrieval protocols in one block.** End-to-end methods rank
   the whole corpus. Conditional rerankers (cross-encoder, LLM) see only the
   dual-encoder top-20 and are bounded by its Recall@20 ceiling: 0.983
   diagnostic, 0.906 CSF, 0.882 HIPAA, 0.840 PF.
3. **Report wins/losses/ties with every paired effect.** Top-1 is binary, so
   the mean difference is decided entirely by the discordant requirements,
   and there are usually fewer than ten of them.
4. **NIST crosswalks are a silver standard.** NIST calls them concept
   relationship mappings and marks the HIPAA OLIR "Comprehensive: No". Say
   "supports" or "addresses", not "implements". An unlisted plausible
   prediction is not automatically a false positive.
5. **Say "submitted and rejected", not "published".** The +0.14 result was
   never published. Refer to it as initially observed or as the earlier
   version of this work.

## Pending work, in priority order

1. ~~**Statistics**~~ **DONE.** `moderation_test` derives its p-value from a
   Freedman-Lane sign-flip permutation test (100,000 sign vectors, seed
   20260801); the bootstrap is kept only for the CI. `sign_flip_test` added
   for the superiority contrast. `tost()` is real two one-sided tests; the
   old CI-inclusion check is renamed `equivalence_verdict` and marked
   descriptive. `moderation_test` no longer names a direction when its
   interval includes zero. `run_confirmatory.py` regenerates every number
   into a JSON record under `results_v3/confirmatory/` carrying argv, UTC
   timestamp, git commit and dirty flag, library versions, module and input
   hashes, and seeds.

   **The "+0.0043, p=0.82" figure does not reproduce and is retired.** Under
   the primary one-pass source the pooled moderation coefficient is
   **+0.0126, 95% CI [−0.0107, +0.0382], sign-flip p=0.311** (n=268). Adding
   the diagnostic corpus moves it to **+0.0208, CI [−0.0003, +0.0439],
   p=0.069** (n=326). Records:
   `results_v3/confirmatory/reform_vs_multi_top1_onepass*.json`. The
   holdout-source versions are kept for comparison but the manuscript cites
   the one-pass ones, because that is the source of its per-corpus effects.
2. ~~**Hybrid spec**~~ **REGISTERED AND RUN.**

   ```
   OSF project     : https://osf.io/vubf6
   registration    : https://osf.io/nzxrv
   DOI             : 10.17605/OSF.IO/NZXRV
   registered at   : 2026-08-01 21:24, public, CC0 1.0 Universal
   contributors    : Sumanshu Sohal alone. Darshankumar Prajapati is a
                     co-author on the manuscript but was not involved in the
                     post-rejection work this registration covers.
   tag             : hybrid-analysis-spec-v1 at commit
                     25ce391d5e00bbb8218cf46878c256c88b2655f0
   ```

   Cite as: Sohal, S. (2026). Prospectively registered exploratory analysis:
   gated hybrid lexical-semantic retrieval for regulatory traceability.
   https://doi.org/10.17605/OSF.IO/NZXRV

   All seven registered text artifacts were verified byte-identical to their
   registered hashes immediately before the primary arm was first run, and
   the tag still resolves to the registered commit.

   | file | sha256 |
   |---|---|
   | `hybrid_spec.py` | `a1b4c789…90597a` |
   | `run_hybrid.py` | `b33459d9…66fc8f` |
   | `test_hybrid_spec.py` | `d0336d6c…07b827` |
   | `freeze_backends.py` | `b20d326e…741482` |
   | `confirmatory_stats.py` | `79ec294a…7e1104` |
   | `frozen_backends/manifest.json` | `93259026…13fa0c` |

   **The analysis never loads a model.** `freeze_backends.py` runs every
   backend once and hashes the score matrices; `run_hybrid.py` consumes only
   those, verifies each hash against the manifest, and aborts on mismatch.
   The dual-encoder is pinned to `all-MiniLM-L6-v2` at revision
   `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, passed explicitly.

   **Disclosure that must survive into the writeup.** The secondary arm
   `hybrid_equal` was computed during method development, before
   registration: Top-1 0.4623 (CSF), 0.3676 (HIPAA), 0.3936 (PF), 0.5517
   (diagnostic). It is observed, not preregistered. Do not delete it: the
   gated arm applies exactly that fusion to gated queries, so removing it
   would hide part of the primary mechanism.

   The gap metric CANNOT be the gate: `compute_gaps` maximizes overlap over
   the gold set, so it is not computable at query time and stays the RQ2
   moderator.
3. ~~**Full RAA on the shared protocol**~~ **DONE.** `score_raa.py` scores
   `multi`, `reform` and `raa_full` once per requirement, under both LSI
   fitting regimes. Protocol rule 3 in the old numbering (`rrf_lexical` is
   not RAA) is closed for ranking. Instrumentation separates `gate_fired`,
   `expanded` and `top1_changed`.
4. **Manuscript, remaining.** The reframe to cross-corpus evaluation is done
   through abstract, RQs, contributions, protocol, results, discussion,
   threats and conclusion. Still open:
   - Qwen under the conditional-reranking block: score the 2 HIPAA parse
     failures as failures (Top-1 0.4559), report availability 0.9706 and
     valid-output accuracy 0.4697 separately, pin the revision, no AURC from
     constant confidence, run candidate-order sensitivity.
   - Document that the diagnostic corpus contains paraphrased ISO 27001, PCI
     DSS and SOC 2 statements written as author summaries.
   - Old-paper remnants: README reproduction steps, cover letter.
   - Abstract is ~350 words. Fine for Elsevier, too long for EMSE.
   - Compile and visually inspect all ten tables in Overleaf. No LaTeX
     toolchain is installed locally; `tools/check_tex.py`-style structural
     checks are all that has been run.
   - If the target becomes EMSE: Springer template, abstract 150–250 words,
     6 keywords, decimal headings, declarations section, DOI links,
     alphabetized references, drop the Elsevier highlights block.
5. **Corpus sensitivities**: 800-53 5.1.1 as the HIPAA-native target
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
6. **LLM**: candidate-order sensitivity (candidates are currently supplied in
   first-stage rank order), pinned Hugging Face model and tokenizer
   revisions, a strict shared validator rejecting duplicate IDs and
   out-of-range confidence, a contamination check by masking framework
   identifiers, and one batched API pass. Confidence is a constant 1.0 from
   the local model, so calibration is a precondition for any AURC result.

## Framing

Current identity:

> Cross-Corpus Evaluation of Domain-Aware Query Reformulation for
> Regulatory Traceability

Central claim, stated the way it can be defended:

> Under a common inductive full-population ranking protocol, the
> reformulation benefit observed on the author-constructed diagnostic corpus
> did not generalize consistently to three externally sourced, NIST-linked
> corpora. The estimates depend on few discordant requirements and are
> sensitive to LSI fitting choices.

Do not make the paper's identity contingent on the hybrid winning. Do not
claim "neural accuracy without giving up determinism and auditability":
logging establishes provenance, not audit usefulness, and determinism must
be measured in a locked environment. Audit usefulness needs a human study.
Do not claim the diagnostic effect is stable *because* the corpus was
constructed to be, or that method differences are unmeasurable at these
corpus sizes. Both overreach.

## Running things

```
USE_TF=0 python score_all.py                 # shared-population ranking table
USE_TF=0 python score_raa.py                 # full RAA, one pass, inductive
USE_TF=0 python score_raa.py --lsi-fit transductive   # writes its own CSV
python analyze_onepass.py                    # primary reformulation record
USE_TF=0 python holdout_lsi_factorial.py     # cell A (controls-only holdout)
USE_TF=0 python holdout_lsi_factorial.py --lsi-fit train_cal   # cell B
python run_confirmatory.py --source onepass  # RQ2 moderation, primary source
python run_confirmatory.py                   # same, holdout source
python heterogeneity_test.py                 # descriptive contrast only
USE_TF=0 python run_local_all.py             # open-weight LLM, all corpora
python precision_analysis.py                 # outcome-blind power analysis
python gap_metrics.py                        # gap distributions + spec hash
python scan_secrets.py                       # before every push

# Hybrid. The registration exists, so all of these are now safe.
python hybrid_spec.py                        # print the frozen spec + hash
python test_hybrid_spec.py --with-score-all  # spec test suite
python run_hybrid.py                         # the registered primary arm
python run_hybrid.py --legacy-lsi            # transductive sensitivity arm
```

Records must never carry `dirty: true` pointing at a commit that does not
contain the code that produced them. Commit the code first, then run, then
commit the record.

Credentials come from `ANTHROPIC_API_KEY` in the environment only. Never a
flag, a literal, or a notebook cell.

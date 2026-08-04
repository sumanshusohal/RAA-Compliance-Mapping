# Reproducing the results

All experiments are deterministic given a seed. The environment is pinned in
`requirements.txt` (Python 3.10.2). Neural baselines require `USE_TF=0` in the
environment to avoid the TensorFlow/Keras import path.

Every number in the manuscript comes from a JSON record under `results_v3/`
carrying the argv, a UTC timestamp, the git commit and dirty flag, library
versions, and SHA-256 hashes of the analysis modules and inputs. Check them
all at once:

```bash
python audit_records.py        # exits non-zero if any record is stale
```

A record is sound only if it was produced from a clean tree and every file it
hashed still hashes the same. Run this before any push.

## Two protocols, and which one carries which number

Ranking figures come from **one scoring pass per requirement** over the full
control corpus, with LSI fitted on control documents only. Decision figures
(coverage, selective accuracy, open-world gap detection) need a calibrated
threshold and come from **repeated stratified holdouts**, 30 seeds, holdout
0.20 and calibration 0.15. These are different estimands and are never mixed
in one comparison. Each section below says which it produces.

## 1. Benchmarks

```bash
# The diagnostic benchmark is released directly in diagnostic_benchmark/.
# The three NIST corpora are rebuilt from NIST sources; see SOURCES.md for
# download URLs and the expected SHA-256 of each artifact. Each builder
# hard-fails on a checksum mismatch and writes provenance.json.
cd csf_benchmark   && python build_csf_benchmark.py
cd hipaa_benchmark && python build_hipaa_benchmark.py
cd pf_benchmark    && python build_pf_benchmark.py
```

## 2. Primary ranking results (reformulation across four corpora)

```bash
USE_TF=0 python score_raa.py                          # inductive, controls-only fit
USE_TF=0 python score_raa.py --lsi-fit transductive   # writes its own CSV
python analyze_onepass.py
```

Writes `shared_raa_scores.csv`, `shared_raa_scores_transductive.csv`, and
`results_v3/shared/reform_vs_multi_onepass.json`, which carries the per-corpus
effects, intervals, sign-flip p-values, wins/losses/ties, the TOST secondary
results with their boundary flags, and the conditional engineered-versus-
external contrast.

`score_raa.py` also reports the reformulation funnel: gate fired, expansion
produced, ranking changed, top-1 changed. These are four different quantities
and the manuscript reports all four alongside the correctness count.

Do not pass `--lsi-fit transductive` together with an explicit
`--out shared_raa_scores.csv`: that overwrites the primary scores with the
sensitivity arm, and nothing downstream notices because the file keeps its
shape. `--out` defaults per regime, and `analyze_onepass.py` aborts if its two
inputs hash the same.

## 3. Protocol versus LSI fit (three-cell factorial)

```bash
USE_TF=0 python holdout_lsi_factorial.py                     # cell A
USE_TF=0 python holdout_lsi_factorial.py --lsi-fit train_cal # cell B
```

Cell C is the one-pass inductive arm from step 2. A vs B isolates the fitting
population; A vs C isolates the evaluation protocol. Records:
`results_v3/shared/holdout_controls_only.json` and `holdout_train_cal.json`.

Cell B reproduces the four figures reported before the reframe, to four
decimals and including their p-values. That is the check that this
reimplementation is faithful.

## 4. RQ2 moderation and the equivalence tests

```bash
python run_confirmatory.py --source onepass                        # primary
python run_confirmatory.py --source onepass --corpora nist,hipaa,pf,diagnostic
python run_confirmatory.py                                         # holdout source
```

`--source` selects where the paired differences come from. The manuscript
cites the `onepass` records, because that is the source of its per-corpus
effects; fitting the interaction to holdout differences would model
differences the paper does not otherwise show.

## 5. Gap correlations (eight Spearman coefficients)

```bash
python gap_correlations.py     # a few minutes: 8 x 100,000 permutations
```

Both outcomes are binary or ternary, so heavily tied against a continuous gap
score. p-values come from permuting the gap vector rather than from the
asymptotic approximation, and Holm-Bonferroni adjusted values are reported
within each family of four. Record:
`results_v3/confirmatory/gap_correlations.json`.

## 6. Preregistered gated hybrid

Registered at doi:10.17605/OSF.IO/NZXRV before the primary arm was run.
`confirmatory_stats.py` is one of the seven hashed artifacts and must stay
byte-identical to the registered version; `audit_records.py` warns if it
drifts, and the fix is to restore it from tag `hybrid-analysis-spec-v1`
rather than to re-record.

```bash
python hybrid_spec.py                        # print the frozen spec and its hash
python test_hybrid_spec.py --with-score-all  # spec test suite
python run_hybrid.py --verify-only           # verify inputs, compute nothing
USE_TF=0 python freeze_backends.py           # regenerate inputs (idempotent)
python run_hybrid.py                         # the registered primary arm
python run_hybrid.py --legacy-lsi            # transductive sensitivity arm
```

The analysis never loads a model. `freeze_backends.py` runs every backend once
and hashes the score matrices; `run_hybrid.py` consumes only those and aborts
on any hash mismatch.

## 7. Holdout ablations (decision metrics)

These produce coverage, selective accuracy and the per-seed logs. Their Top-1
columns are descriptive; the ranking claims come from step 2.

```bash
python raa_agent.py --regs diagnostic_benchmark/diag_regs.csv \
  --controls diagnostic_benchmark/diag_controls.csv \
  --mappings diagnostic_benchmark/diag_mappings.csv \
  --ablation --runs 30 --target-coverage 0.80 --output-dir results_v3/diagnostic

python raa_agent.py --regs csf_benchmark/csf_regs.csv \
  --controls csf_benchmark/csf_controls.csv \
  --mappings csf_benchmark/csf_mappings.csv \
  --ablation --runs 30 --target-coverage 0.80 --output-dir results_v3/nist
```

Neural baselines: append `--backend semantic` or `--backend reranker` (with
`USE_TF=0`).

## 8. Open-world gap detection

```bash
python raa_agent.py --regs diagnostic_benchmark/diag_regs.csv \
  --controls diagnostic_benchmark/diag_controls.csv \
  --mappings diagnostic_benchmark/diag_mappings.csv \
  --backend agent --runs 30 --open-world-frac 0.5 --output-dir results_v3/open_world
```

## 9. Sensitivity to the hand-set constants

```bash
# Reformulation trigger (ranking): --rel-retry in {0.05,0.10,0.15,0.20}
# Decision constants: --crossref-relax-conf, --crossref-relax-gap, --verify-tighten
python raa_agent.py ... --backend reform --rel-retry 0.15
python raa_agent.py ... --backend agent --open-world-frac 0.5 --verify-tighten 0.20
```

## 10. Reasoning traces

```bash
python raa_agent.py ... --backend agent --export-traces
```

## Requirement-level inference

The 30 stratified splits overlap heavily, so per-seed metrics are not
independent and an across-seed t-test is pseudoreplicated. All significance
claims are made at the requirement level: a paired sign-flip randomization
test for the p-value and a paired bootstrap over requirements for the 95%
interval. Wins, losses and ties accompany every paired effect, because Top-1
is binary and the mean difference is decided entirely by the requirements on
which two methods disagree.

```bash
python results_v3/query_level_stats.py results_v3/diagnostic multi reform top1
python results_v3/query_level_stats.py results_v3/nist single multi top1
```

## What is not reproduced here

The open-weight LLM reranking arm (`run_local_all.py`) is in the repository
but is not part of the manuscript. Its model and tokenizer revisions are
unpinned and its candidate-order sensitivity is untested, so no manuscript
claim rests on it.

The raw per-seed and per-query outputs behind the holdout tables are in
`results_v3/` (`perseed_*.csv`, `perquery_*.csv`). The one-pass, factorial and
hybrid results are not seed logs: they come from the CSVs and JSON records
named in the sections above.

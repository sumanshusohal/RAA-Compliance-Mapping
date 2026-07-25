# Reproducing the results

All experiments are deterministic given a seed. The environment is pinned in
`requirements.txt` (Python 3.10.2). Neural baselines require `USE_TF=0` in the
environment to avoid the TensorFlow/Keras import path.

The raw, full-precision per-seed and per-query outputs behind every table are in
`results_v3/` (`perseed_*.csv`, `perquery_*.csv`), so the tables can be checked
without rerunning anything.

## 1. Benchmarks

```bash
# Diagnostic benchmark is released directly in diagnostic_benchmark/.
# Real-world NIST benchmark: download the two NIST sources (see SOURCES.md) into
# csf_benchmark/, then:
cd csf_benchmark && python build_csf_benchmark.py   # verifies SHA-256, writes the 4 CSVs
```

## 2. Main ablations (Tables: diagnostic and NIST)

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

Neural baselines: append `--backend semantic` or `--backend reranker` (with `USE_TF=0`).

## 3. Query-level statistics (avoids seed-level pseudoreplication)

The 30 stratified splits overlap, so per-seed metrics are not independent. We
report query-level paired tests: aggregate each metric to one value per unique
query (mean over the seeds where it appears in test), then a paired sign-flip
randomization test (p-value) and paired bootstrap (95% CI) over queries.

```bash
python results_v3/query_level_stats.py results_v3/diagnostic multi reform top1
python results_v3/query_level_stats.py results_v3/nist single multi top1
```

## 4. Open-world gap detection

```bash
python raa_agent.py --regs diagnostic_benchmark/diag_regs.csv \
  --controls diagnostic_benchmark/diag_controls.csv \
  --mappings diagnostic_benchmark/diag_mappings.csv \
  --backend agent --runs 30 --open-world-frac 0.5 --output-dir results_v3/open_world
```

## 5. Sensitivity analysis

```bash
# Reformulation trigger (ranking): --rel-retry in {0.05,0.10,0.15,0.20}
# Decision constants: --crossref-relax-conf, --crossref-relax-gap, --verify-tighten
python raa_agent.py ... --backend reform --rel-retry 0.15
python raa_agent.py ... --backend agent --open-world-frac 0.5 --verify-tighten 0.20
```

## 6. Reasoning traces

```bash
python raa_agent.py ... --backend agent --export-traces   # writes traces_agent_seed<seed>.json
```

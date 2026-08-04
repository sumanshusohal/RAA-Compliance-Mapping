# RAA: Retrieval-Augmented Agentic Compliance Mapping

A deterministic, ReAct-inspired multi-stage retrieval workflow for automated regulatory
compliance traceability recovery. RAA combines domain-aware query reformulation, multi-backend
retrieval fusion, cross-framework corroboration, bidirectional verification, and calibrated
selective prediction to map regulatory requirements to implementation controls. It executes a
fixed conditional sequence of deterministic tools (not an open-ended language-model loop) and
emits a structured execution trace for every decision, recording the
inputs, transformations, scores and effective thresholds behind it. Whether
such a record is useful under audit is a question about practitioners that
this work does not test.

## Key Results (30-seed evaluation)

All ranking metrics are computed from the **full candidate ranking**, independent of the
accept/abstain decision; coverage, selective accuracy, and open-world gap detection are
reported separately as decision metrics. (Conflating the two — computing ranking only over
accepted queries — inflates ranking metrics with coverage and is a bug we fixed.)

Significance uses **query-level paired tests** (aggregate each metric to one value per unique
requirement, then a paired sign-flip randomization test + bootstrap), which avoid the
pseudoreplication of testing across overlapping seed splits. Agent variants are calibrated
end-to-end to a 0.80 coverage target.

The tables immediately below are the repeated-holdout decision-metric tables and are
descriptive. The cross-corpus ranking results further down use the one-pass protocol.

**Diagnostic benchmark** (engineered vocabulary mismatch, 58 reqs / 110 controls / 86 links):

| Method | Top-1 | MRR@5 | Coverage |
|--------|-------|-------|----------|
| TF-IDF | 0.572 | 0.662 | 0.803 |
| Dual-encoder | **0.672** | **0.762** | 0.758 |
| Cross-encoder | 0.647 | 0.733 | 0.719 |
| Full Agent (RAA) | 0.644 | 0.737 | 0.833 |

Reformulation is the one component that improves ranking here. Under the primary one-pass
protocol it gains +0.121 Top-1 over fusion (p = 0.039), by flipping correctness on 9 of 58
requirements. Neural retrieval is not out-ranked by the agent, but does not beat it either
(agent vs dual-encoder p ≈ 1.0). No method wins on coverage — LSI (0.922) is highest.

**Real-world benchmark** (NIST CSF → SP 800-53r5 official crosswalk, 106 reqs / 300 controls / 495 links):

| Method | Top-1 | MRR@5 | Coverage |
|--------|-------|-------|----------|
| TF-IDF | 0.390 | 0.505 | 0.792 |
| Dual-encoder | 0.402 | 0.530 | 0.733 |
| Cross-encoder | **0.562** | **0.666** | 0.775 |
| Full Agent (RAA) | 0.406 | 0.508 | 0.721 |

On this corpus **no deterministic component significantly improves ranking** (reformulation
+0.028 Top-1 p = 0.38 under the primary protocol; fusion +0.03 p = 0.51). The agent has no
coverage or selective-accuracy edge. As a complete system, dual-encoder retrieval followed by
cross-encoder reranking beats every single-stage method here (+0.154, p = 0.002). That result
stands as an end-to-end comparison; as a comparison between reranking methods it is bounded
by the first stage, whose Recall@20 is 0.906, and the paper reports the ceiling alongside the
score rather than discounting it.

We do not offer a qualitative advantage in place of the missing metric one. The agent records
each step and its inputs, which establishes where a mapping came from. Whether that record is
useful under audit is a question about practitioners that this work does not test, and
execution determinism was never measured under pinned dependencies.

An open-world stress test shows abstention detects only ~24% of genuine no-match gaps
(avg 4.4 gaps/seed), an open limitation.

**Since the DKE submission, the evaluation has grown to four corpora and the headline finding
has changed.** All ranking results are now measured the same way: one scoring pass per
requirement over the full control corpus, with LSI fitted on control documents only.

The +0.121 reformulation estimate on the diagnostic corpus we built (p=0.039) did not recur.
The three NIST corpora give +0.028 (CSF), 0.000 (HIPAA) and +0.032 (PF), none distinguishable
from zero. Two things qualify all four numbers:

- each turns on very few requirements. Wins/losses/ties are 8/1/49, 4/1/101, 2/2/64 and
  4/1/89, so fewer than ten decisions carry each corpus-level figure;
- each moves by up to 0.029 when the latent-semantic backend is refitted on a different
  population, which is a choice unrelated to reformulation and, on the external corpora, is
  as large as the effect itself.

The evaluation protocol contributes nothing to these particular estimates: repeated holdouts
and a single full-population pass agree to four decimals, because under a controls-only fit
no part of this deterministic pipeline depends on the split. That is a fact about this design,
not about evaluation protocols generally, and it does not hold for the decision metrics.

What separates the data is our corpus against the NIST ones. It is not measured vocabulary
gap: that measure is fitted per corpus and provides no scale on which corpora can be compared.

All four corpora are exploratory. The one preregistered analysis is the gated hybrid
(doi:10.17605/OSF.IO/NZXRV), and because its hypothesis came from these corpora it is
preregistered but still exploratory. See `HANDOFF.md`.

## Architecture

RAA is a fixed multi-stage sequence (not an open-ended reasoning loop) whose tools are split into
**ranking-shaping** and **decision-shaping** groups, so that ranking and decision effects can be
attributed separately:

Ranking tools:
1. **Retrieve + Fuse** — Multi-backend retrieval (TF-IDF, BM25, LSI) with Reciprocal Rank Fusion
2. **Reformulate** — Domain-aware query expansion via a curated compliance thesaurus (20 concept families, 32 regex patterns), triggered by a scale-invariant top-2 relative margin
3. **Decompose** — Query decomposition for compound regulations (own ablation flag). This is
   a ranking tool: on the shared population it changes the ranking for one requirement,
   HIPAA 58, and for none elsewhere.

Decision tools (measured: these reorder nothing on any of the four corpora):
4. **Cross-Reference** — Cross-framework corroboration from a static family→framework taxonomy (no ground-truth leakage)
5. **Verify** — Bidirectional check: query the requirement corpus with the control's text; if the requirement is not recovered in the top decile, tighten the acceptance threshold
6. **Selective decision** — Calibrated accept/abstain thresholds

## Benchmarks

Four corpora are included. Three are real text with NIST-authored ground truth; one is
author-built and used only as a diagnostic instrument. All three real corpora map to the same
300-control SP 800-53 5.2.0 corpus.

- **Diagnostic** (`diagnostic_benchmark/`): 58 regulations / 110 controls / 86 links,
  author-built to isolate vocabulary mismatch. Control labels in `diag_controls.csv` are
  59 `perfect`, 27 `good`, 20 `hard_neg` and 4 `neg`. These are construction labels, not
  measured lexical regimes: 35 of the 86 positive links share no content words with their
  requirement, including 14 labelled `perfect`. Never used as a confirmatory corpus.
- **Real-world NIST CSF** (`csf_benchmark/`): 106 CSF 1.1 subcategories / 300 SP 800-53
  controls / 495 links, from NIST's published OSCAL catalog and official crosswalk. Stays at
  CSF 1.1 deliberately: CPRT documents a near 1:1 derivation of CSF 2.0 subcategories from
  1.1, so 2.0 would not be an independent corpus.
- **HIPAA** (`hipaa_benchmark/`): 68 requirements / 274 links, regulation-to-control against
  statutory source text, built from the NIST OLIR crosswalk. The builder snapshots and hashes
  all six raw CPRT responses under `cprt_snapshots/`.
- **Privacy Framework** (`pf_benchmark/`): 94 requirements / 456 links, NIST-authored, with
  a different subject matter from HIPAA. We do not compare their vocabulary gaps: the gap
  measure is fitted per corpus and gives no common scale, and that comparison is on the
  retraction list in HANDOFF.md.

NIST describes these crosswalks as concept-relationship mappings (NIST IR 8477) and marks the
HIPAA OLIR "Comprehensive: No", so they are a silver standard. The task is reference-link
recovery, not implemented-compliance verification, and an unlisted plausible prediction is not
automatically a false positive.

## Project Structure

```
.
├── raa_agent.py                 # Main agent implementation (v2.0)
├── output/
│   ├── ablation_table.csv       # 30-seed ablation results (8 variants x 15 metrics)
│   ├── summary_tfidf.csv        # TF-IDF baseline results
│   ├── summary_semantic.csv     # Dual-encoder results
│   └── summary_reranker.csv     # Cross-encoder reranker results
└── README.md
```

## Usage

### Map your own regulations to controls
```bash
# Discovery mode — just provide your regulation and control files
python raa_agent.py --regs regulations.csv --controls controls.csv

# With Excel files
python raa_agent.py --regs regs.xlsx --controls controls.xlsx

# Get top-10 matches per regulation
python raa_agent.py --regs regs.csv --controls controls.csv --top-k 10
```

Output: `output/mappings.csv` with ranked control matches, confidence scores, and agent reasoning steps.

### Input file formats

**Regulations** (CSV/Excel/JSON):
| id | framework | text |
|----|-----------|------|
| 0 | GDPR | Art 32: Implement appropriate technical measures including encryption... |
| 1 | NIST | PR.AC-1: Identities and credentials are managed... |

Only `text` column is required. `id` and `framework` are optional.

**Controls** (CSV/Excel/JSON):
| id | text |
|----|------|
| 0 | AES-256 encryption applied to all personal data at rest... |
| 1 | Automated workflow for processing data deletion requests... |

Only `text` column is required.

**Mappings** (optional, for evaluation):
| regulation_id | control_id |
|---------------|------------|
| 0 | 0 |
| 0 | 3 |

### Reproduce the paper's tables
```bash
# Diagnostic benchmark ablation (Table 2/3)
python raa_agent.py --regs diagnostic_benchmark/diag_regs.csv \
  --controls diagnostic_benchmark/diag_controls.csv \
  --mappings diagnostic_benchmark/diag_mappings.csv --ablation --runs 30

# Real-world NIST ablation (Table 4)
python raa_agent.py --regs csf_benchmark/csf_regs.csv \
  --controls csf_benchmark/csf_controls.csv \
  --mappings csf_benchmark/csf_mappings.csv --ablation --runs 30

# Neural baselines (add --backend semantic or reranker)

# Open-world gap detection (Table 5): hold out 50% of controls
python raa_agent.py --regs diagnostic_benchmark/diag_regs.csv \
  --controls diagnostic_benchmark/diag_controls.csv \
  --mappings diagnostic_benchmark/diag_mappings.csv \
  --backend agent --runs 30 --open-world-frac 0.5

# Export execution traces (Table 6)
python raa_agent.py ... --backend agent --export-traces
```

### Available backends
- Baselines: `tfidf`, `bm25`, `lsi`
- Semantic: `semantic`, `reranker` (requires `sentence-transformers`)
- Agent ablation: `single`, `multi`, `reform`, `decomp`, `crossref`, `agent`

## Requirements

- Python 3.8+
- numpy, pandas, scikit-learn, scipy
- sentence-transformers (optional, for semantic backends)
- torch (CPU sufficient)


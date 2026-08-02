# RAA: Retrieval-Augmented Agentic Compliance Mapping

A deterministic, ReAct-inspired multi-stage retrieval workflow for automated regulatory
compliance traceability recovery. RAA combines domain-aware query reformulation, multi-backend
retrieval fusion, cross-framework corroboration, bidirectional verification, and calibrated
selective prediction to map regulatory requirements to implementation controls. It executes a
fixed conditional sequence of deterministic tools (not an open-ended language-model loop) and
emits a structured, auditable trace for every decision.

## Key Results (30-seed evaluation)

All ranking metrics are computed from the **full candidate ranking**, independent of the
accept/abstain decision; coverage, selective accuracy, and open-world gap detection are
reported separately as decision metrics. (Conflating the two — computing ranking only over
accepted queries — inflates ranking metrics with coverage and is a bug we fixed.)

Significance uses **query-level paired tests** (aggregate each metric to one value per unique
requirement, then a paired sign-flip randomization test + bootstrap), which avoid the
pseudoreplication of testing across overlapping seed splits. Agent variants are calibrated
end-to-end to a 0.80 coverage target.

**Diagnostic benchmark** (engineered vocabulary mismatch, 58 reqs / 110 controls / 86 links):

| Method | Top-1 | MRR@5 | Coverage |
|--------|-------|-------|----------|
| TF-IDF | 0.572 | 0.662 | 0.803 |
| Dual-encoder | **0.672** | **0.762** | 0.758 |
| Cross-encoder | 0.647 | 0.733 | 0.719 |
| Full Agent (RAA) | 0.644 | 0.737 | 0.833 |

Domain-aware reformulation is the one component that significantly improves ranking here
(query-level +0.14 Top-1 over fusion, p = 0.009). Neural retrieval is not out-ranked by the agent,
but does not beat it either (agent vs dual-encoder p ≈ 1.0). No method wins on coverage —
LSI (0.922) is highest.

**Real-world benchmark** (NIST CSF → SP 800-53r5 official crosswalk, 106 reqs / 300 controls / 495 links):

| Method | Top-1 | MRR@5 | Coverage |
|--------|-------|-------|----------|
| TF-IDF | 0.390 | 0.505 | 0.792 |
| Dual-encoder | 0.402 | 0.530 | 0.733 |
| Cross-encoder | **0.562** | **0.666** | 0.775 |
| Full Agent (RAA) | 0.406 | 0.508 | 0.721 |

On same-institution text, **no deterministic component significantly improves ranking**
(reformulation +0.01 Top-1 p = 0.65; fusion +0.03 p = 0.51). The agent has no coverage or
selective-accuracy edge. A two-stage cross-encoder pipeline scores higher (+0.154, p = 0.002),
but it reranks a 20-candidate shortlist rather than the full corpus, so it is reported in a
separate protocol section and not as a like-for-like ranking win.

We do not offer a qualitative advantage in place of the missing metric one. The agent records
each step and its inputs, which establishes where a mapping came from. Whether that record is
useful under audit is a question about practitioners that this work does not test, and
execution determinism was never measured under pinned dependencies.

An open-world stress test shows abstention detects only ~24% of genuine no-match gaps
(avg 4.4 gaps/seed), an open limitation.

**Since the DKE submission, the evaluation has grown to four corpora and the headline finding
has changed.** The reformulation effect does not replicate on real text: +0.140 on the
engineered diagnostic corpus against +0.009 (CSF), +0.008 (HIPAA) and +0.003 (PF). What
separates the data is engineered versus real, not measured vocabulary gap. All four corpora
are exploratory; no preregistration predates them. See `HANDOFF.md`.

## Architecture

RAA is a fixed multi-stage sequence (not an open-ended reasoning loop) whose tools are split into
**ranking-shaping** and **decision-shaping** groups, so that ranking and decision effects can be
attributed separately:

Ranking tools:
1. **Retrieve + Fuse** — Multi-backend retrieval (TF-IDF, BM25, LSI) with Reciprocal Rank Fusion
2. **Reformulate** — Domain-aware query expansion via a curated compliance thesaurus (20 concept families, 32 regex patterns), triggered by a scale-invariant top-2 relative margin
3. **Decompose** — Query decomposition for compound regulations (own ablation flag)

Decision tools (never reorder the ranking):
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
- **Privacy Framework** (`pf_benchmark/`): 94 requirements / 456 links, NIST-authored, a
  narrower vocabulary gap than HIPAA.

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

# Export reasoning traces (Table 6)
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


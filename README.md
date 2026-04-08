# RAA: Retrieval-Augmented Agentic Compliance Mapping

A tool-augmented agent for automated regulatory compliance traceability recovery. RAA uses a ReAct-style reasoning loop with domain-aware query reformulation, multi-backend retrieval fusion, cross-framework corroboration, and calibrated selective prediction to map regulatory requirements to implementation controls.

## Key Results (30-seed evaluation)

| Method | Top-1 | MRR@5 | nDCG@5 | Recall@5 |
|--------|-------|-------|--------|----------|
| TF-IDF | 0.511 | 0.588 | 0.548 | 0.593 |
| BM25 | 0.481 | 0.537 | 0.504 | 0.535 |
| LSI | 0.500 | 0.623 | 0.586 | 0.659 |
| Full Agent (RAA) | **0.617** | **0.705** | **0.643** | **0.706** |

The full agent achieves +21% Top-1 accuracy over the best lexical baseline, with domain-aware query reformulation as the dominant contributor (+28%).

## Architecture

RAA implements a genuine agentic reasoning loop with five tools:

1. **Retrieve** - Multi-backend retrieval (TF-IDF, BM25, LSI) with Reciprocal Rank Fusion
2. **Reformulate** - Domain-aware query expansion using a curated compliance thesaurus (20 concept families, 32 regex patterns)
3. **Decompose** - Query decomposition for complex multi-concept regulations
4. **Cross-Reference** - Cross-framework corroboration using regulatory family relationships
5. **Verify** - Bidirectional reverse-retrieval consistency check

## Benchmark

Hardened benchmark with:
- **110 controls** (66 vocabulary-matched + 20 vocabulary-mismatched + 24 adversarial hard negatives)
- **58 regulations** across 7 regulatory frameworks
- **86 ground-truth positive links**
- Stratified holdout splitting by framework family

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

### Run full ablation (30 seeds)
```bash
python raa_agent.py --ablation --runs 30
```

### Run a single variant
```bash
python raa_agent.py --backend agent --runs 30
python raa_agent.py --backend tfidf --runs 5
```

### Available backends
- Baselines: `tfidf`, `bm25`, `lsi`
- Semantic: `semantic`, `reranker` (requires `sentence-transformers`)
- Agent ablation: `single`, `multi`, `reform`, `crossref`, `agent`

## Requirements

- Python 3.8+
- numpy, scikit-learn, scipy
- sentence-transformers (optional, for semantic backends)
- torch (CPU sufficient)


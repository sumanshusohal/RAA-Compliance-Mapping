# Data sources and provenance

## Real-world NIST benchmark (`csf_benchmark/`)

The NIST benchmark is regenerated from two artifacts published by NIST. Download
them to `csf_benchmark/` and run `python build_csf_benchmark.py` to reproduce
`csf_regs.csv`, `csf_controls.csv`, `csf_control_codes.csv`, and `csf_mappings.csv`.

| File | Source URL | SHA-256 |
|------|-----------|---------|
| `csf-to-80053r5-mappings.xlsx` | https://csrc.nist.gov/files/pubs/sp/800/53/r5/upd1/final/docs/csf-pf-to-sp800-53r5-mappings.xlsx | `8b09e8c5bd11dcfa494d2f5beec2d32d98df934285c0214712d020abefc162f1` |
| `NIST_SP-800-53_rev5_catalog.json` (OSCAL) | https://raw.githubusercontent.com/usnistgov/oscal-content/main/nist.gov/SP800-53/rev5/json/NIST_SP-800-53_rev5_catalog.json | `01f37cf90ea99d92242c936cbfbdebcc338eef1f71454e2acac36cc56e9bc062` |

Requirements are the 108 CSF v1.1 subcategories from the "CSF to SP 800-53r5"
sheet; two subcategories mapped to "all controls" are dropped, leaving 106.
Controls are the 300 non-withdrawn SP 800-53r5 base controls (title + statement,
organization-defined parameters resolved to their assignment labels). Ground
truth is NIST's official crosswalk. Per NIST IR 8477 these are concept-relationship
mappings, so the task is reference-link recovery, not implemented-compliance
verification.

## Diagnostic benchmark (`diagnostic_benchmark/`)

The diagnostic benchmark is author-constructed to isolate vocabulary mismatch.
Every requirement is a simplified single-clause statement adapted from official
regulatory text (GDPR, NIST CSF/800-53, HIPAA, PCI DSS, ISO 27001, SOX, SOC 2).
Controls comprise 66 vocabulary-matched positives, 20 vocabulary-mismatched
positives, and 24 hard negatives; the thesaurus and concept patterns were frozen
before the NIST benchmark was assembled. This benchmark is a controlled diagnostic
instrument, not a validated gold standard: it has not undergone independent expert
annotation, adjudication, or inter-rater agreement, and results on it should be
read accordingly (see the manuscript's Threats to Validity).

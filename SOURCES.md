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

## HIPAA benchmark (`hipaa_benchmark/`)

Regulation-to-control, built from statutory source text rather than from
framework prose, which makes it the widest measured vocabulary gap of the
three real corpora. 68 requirements, 274 links.

| artifact | source | version | SHA-256 |
|---|---|---|---|
| SP 800-53 control catalog (OSCAL) | usnistgov/oscal-content | release `v1.5.0`, catalog 5.2.0 | `01f37cf9…9bc062` |
| HIPAA Security Rule requirements and mapping | NIST CPRT | `SP800_66_2_0_0`, OLIR `SP-800-66-Rev-2-to-SP-800-53-Rev-5.1.1` | six raw responses, individually hashed |

`build_hipaa_benchmark.py` snapshots all six raw CPRT graph responses under
`cprt_snapshots/` and records a SHA-256 for each in `provenance.json`, so the
corpus is rebuildable without re-querying a live API. Zero unresolved control
references.

Two caveats carried in `provenance.json`. The OLIR targets 800-53 **5.1.1**
while the catalog here is **5.2.0**; the two differ by one control (SA-24) and
all 1,241 links resolve in both, but 5.1.1 as the HIPAA-native target remains
an open sensitivity (see HANDOFF.md). And NIST marks this OLIR
"Comprehensive: **No**", so absence of a link is not evidence against a
prediction.

## Privacy Framework benchmark (`pf_benchmark/`)

NIST-authored throughout, with a narrower vocabulary gap than HIPAA. 94
requirements, 456 links.

| artifact | source | SHA-256 |
|---|---|---|
| SP 800-53 control catalog (OSCAL) | usnistgov/oscal-content `v1.5.0`, catalog 5.2.0 | `01f37cf9…9bc062` |
| PF to SP 800-53r5 crosswalk | `csrc.nist.gov` `csf-pf-to-sp800-53r5-mappings.xlsx`, sheet "PF to SP 800-53r5" | `8b09e8c5…c162f1` |

From `provenance.json`: 100 subcategories seen, 6 dropped for having no link,
5 blanket references skipped, 0 unresolved control references, enhancements
collapsed to their base control. PF subcategory identifiers and text have zero
overlap with the CSF corpus, so the two are independent corpora.

## Diagnostic benchmark (`diagnostic_benchmark/`)

The diagnostic benchmark is author-constructed to isolate vocabulary mismatch.
Every requirement is a simplified single-clause statement adapted from official
regulatory text (GDPR, NIST CSF/800-53, HIPAA, PCI DSS, ISO 27001, SOX, SOC 2).
Controls comprise 59 `perfect`-labelled positives, 27 `good`-labelled
positives, 20 `hard_neg` and 4 `neg`, as released in the `match_type` column of
`diag_controls.csv`; the thesaurus and concept patterns were frozen before the
NIST benchmark was assembled.

An earlier version of this file and of the manuscript stated 66 matched and 20
mismatched positives. The totals were right, 86 positives and 24 negatives, but
the split was not, and the released data is authoritative.

**These are construction labels, not measured lexical regimes, and should not
be renamed "vocabulary-matched" and "vocabulary-mismatched" as though they
were.** The two do correlate: `good` averages 0.043 IDF-weighted overlap with
its requirement against 0.250 for `perfect`, medians 0.000 and 0.197. But they
do not coincide. Under the frozen tokenizer of `gap_metrics.py`:

| quantity | count |
|---|---|
| `perfect`-labelled positive links | 59 |
| `good`-labelled positive links | 27 |
| positive links with NONZERO content-word overlap | 51 |
| positive links with ZERO content-word overlap | 35 |

21 of the 27 `good` links have zero overlap, but so do 14 of the 59 `perfect`
links. So 59/27 is the construction split and 51/35 is the observed lexical
split, and the two are different partitions of the same 86 links. Until a
`vocab_regime` field is added (see HANDOFF.md), cite the labels by name and
report the 35/86 zero-overlap figure separately.

This benchmark is a controlled diagnostic
instrument, not a validated gold standard: it has not undergone independent expert
annotation, adjudication, or inter-rater agreement, and results on it should be
read accordingly (see the manuscript's Threats to Validity).

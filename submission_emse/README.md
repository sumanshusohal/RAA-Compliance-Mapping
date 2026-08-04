# EMSE submission package

Built by `make_submission.py` from commit `ad0f9d9940b3f72635bdfbf44c212f40edf80fae`.
Regenerate with `python make_submission.py`; do not edit files here by hand,
they are copies.

Submission goes through Editorial Manager:
<https://www.editorialmanager.com/emse/>

## Contents

| file | notes |
|---|---|
| `manuscript.tex` | Springer `sn-jnl` source, generated from `manuscript3_revised.tex` by `make_emse.py` |
| `cover_letter.tex` | addressed to the EMSE editors |
| `supplementary/corpora/` | the four evaluation corpora as CSV, with each builder's `provenance.json` |
| `supplementary/REPRODUCE.md` | how to regenerate every number in the paper |
| `supplementary/SOURCES.md` | provenance and SHA-256 of every NIST source artifact |
| `supplementary/LICENSE` | MIT, for the software |
| `supplementary/LICENSE-DATA` | CC BY 4.0, for the corpora |

`sn-jnl.cls` and `sn-basic.bst` are NOT included. They come from the Springer
Nature LaTeX template; build on Overleaf using that template rather than
copying the class file around.

## Before you upload

Run `python make_submission.py --check` and clear everything it reports.
The items it cannot check for you:

1. **Compile the manuscript and read the PDF.** Springer states that files
   which do not compile will most likely fail to submit, and asks authors to
   fix errors locally first. This source has never been compiled. Check the
   eleven tables, the algorithm float, page breaks, reference wrapping and
   the author block.
2. **Add the compiled PDF to this folder** before zipping.
3. **Fill both affiliation placeholders.** `manuscript.tex` carries
   `CITY`/`COUNTRY` for the two independent-researcher affiliations. They are
   placeholders on purpose, not guesses.
4. **Archive the artifact and cite its DOI.** Tag a release, deposit it on
   Zenodo or a new OSF component, and put that DOI in the data availability
   statement. This is separate from the preregistration
   (doi:10.17605/OSF.IO/NZXRV), which is immutable and covers only the gated
   hybrid analysis.
5. **Confirm the supplementary claim is accurate.** The data availability
   statement says the corpora are provided as supplementary material. Either
   upload `supplementary/` or reword it to point at the archived release.
6. **DOIs on the references.** Springer asks for them; 56 entries need
   individual lookup.

## What the paper is

A cross-corpus evaluation of domain-aware query reformulation for regulatory
traceability. The headline result is negative: a +0.121 Top-1 gain measured
on a corpus the authors built did not recur on three externally authored
NIST-derived corpora. One analysis, a gated lexical-semantic hybrid, was
preregistered before it was run.

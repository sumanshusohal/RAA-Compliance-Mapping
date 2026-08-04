#!/usr/bin/env python3
"""Assemble the EMSE submission package, and report what is still missing.

The package is built OUTSIDE the repository, next to "DKE submission
package", because it is a deliverable rather than source. Default location:

    <project root>/EMSE submission package/
        overleaf/            everything Overleaf needs to build the PDF
        cover_letter.tex
        supplementary/       corpora and the documents that describe them
        README.md            contents and the pre-upload checklist

Submission goes through Editorial Manager at
https://www.editorialmanager.com/emse/. Springer Nature asks for the LaTeX
source, the bibliography, figures in the same directory, and a PDF you have
compiled and checked yourself, all zipped.

Nothing here compiles LaTeX. There is no toolchain in this environment, so
the PDF has to come from Overleaf, which is also where sn-jnl.cls comes from.

Usage:
    python make_submission.py
    python make_submission.py --check          # report readiness, build nothing
    python make_submission.py --out DIRECTORY  # build somewhere else
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "EMSE submission package")

CORPORA = [
    ("diagnostic", "diagnostic_benchmark", "diag_"),
    ("csf_1.1", "csf_benchmark", "csf_"),
    ("hipaa", "hipaa_benchmark", ""),
    ("privacy_framework", "pf_benchmark", ""),
]

SUPPLEMENTARY_DOCS = ["REPRODUCE.md", "SOURCES.md", "LICENSE", "LICENSE-DATA"]


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE,
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def build(out):
    if os.path.isdir(out):
        shutil.rmtree(out)
    overleaf = os.path.join(out, "overleaf")
    os.makedirs(overleaf)
    os.makedirs(os.path.join(out, "supplementary", "corpora"))

    # Everything the PDF build needs lives in overleaf/. The manuscript has
    # no figures and its bibliography is an inline thebibliography, so the
    # only thing missing is the class file, which Overleaf supplies.
    shutil.copy(os.path.join(HERE, "manuscript_emse.tex"),
                os.path.join(overleaf, "main.tex"))
    with open(os.path.join(overleaf, "README.md"), "w", newline="\n",
              encoding="utf-8") as f:
        f.write(overleaf_readme())

    shutil.copy(os.path.join(HERE, "cover_letter.tex"),
                os.path.join(out, "cover_letter.tex"))

    for label, directory, prefix in CORPORA:
        dst = os.path.join(out, "supplementary", "corpora", label)
        os.makedirs(dst)
        for stem in ("regs", "controls", "mappings"):
            src = os.path.join(HERE, directory, f"{prefix}{stem}.csv")
            if os.path.exists(src):
                shutil.copy(src, os.path.join(dst, f"{stem}.csv"))
        prov = os.path.join(HERE, directory, "provenance.json")
        if os.path.exists(prov):
            shutil.copy(prov, os.path.join(dst, "provenance.json"))

    for name in SUPPLEMENTARY_DOCS:
        src = os.path.join(HERE, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out, "supplementary", name))

    with open(os.path.join(out, "README.md"), "w", newline="\n",
              encoding="utf-8") as f:
        f.write(package_readme())
    print(f"built {out}\n  from commit {git_commit()[:8]}")


def overleaf_readme():
    return """# Overleaf build folder

Everything needed to produce the submission PDF, except the Springer class
file, which Overleaf provides.

| file | notes |
|---|---|
| `main.tex` | the manuscript, Springer `sn-jnl` with the `sn-basic` option |

No figure files: the paper has no figures, only `tabular` environments.
No `.bib` or `.bst`: the bibliography is an inline `thebibliography`
environment, so BibTeX never runs.

## How to build

1. On Overleaf, start a new project from the **Springer Nature LaTeX
   template** (the December 2024 authoring template). That project already
   contains `sn-jnl.cls` and its supporting files.
2. Replace the template's `main.tex` with the `main.tex` here.
3. Compile with **pdfLaTeX**. Springer asks authors to fix all compilation
   errors locally before uploading, so do not submit until this is clean.

## What to look at in the PDF

This source has never been compiled. Check, in rough order of risk:

- the eleven tables, especially the seven-column unified comparison and the
  factorial, which has a `\\cmidrule`-spanned header;
- the `algorithm` float, and whether it lands sensibly;
- page breaks around the wider tables;
- reference wrapping in the 56-entry bibliography;
- the author block and both affiliations.

## Before you build

`main.tex` still contains `CITY` and `COUNTRY` placeholders on the two
independent-researcher affiliations. Fill them in the repository source, not
here: edit `make_emse.py`, then run `python make_emse.py` followed by
`python make_submission.py`. Editing this copy is lost on the next rebuild.
"""


def package_readme():
    return f"""# EMSE submission package

Built by `make_submission.py` in the git repo, from commit `{git_commit()}`.
Regenerate with `python make_submission.py`. Do not edit files here by hand;
they are copies and the next build overwrites them.

Submission goes through Editorial Manager:
<https://www.editorialmanager.com/emse/>

## Contents

| path | notes |
|---|---|
| `overleaf/main.tex` | the manuscript, Springer `sn-jnl`; see `overleaf/README.md` |
| `cover_letter.tex` | addressed to the EMSE editors |
| `supplementary/corpora/` | the four evaluation corpora as CSV, with each builder's `provenance.json` |
| `supplementary/REPRODUCE.md` | how to regenerate every number in the paper |
| `supplementary/SOURCES.md` | provenance and SHA-256 of every NIST source artifact |
| `supplementary/LICENSE` | MIT, for the software |
| `supplementary/LICENSE-DATA` | CC BY 4.0, for the corpora |

## Before you upload

Run `python make_submission.py --check` and clear what it reports. The rest
it cannot check for you:

1. **Compile and read the PDF.** See `overleaf/README.md`. Springer states
   that files which do not compile will most likely fail to submit.
2. **Put the compiled PDF in this folder** before zipping.
3. **Fill both affiliation placeholders.** `overleaf/main.tex` carries
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
"""


def check(out):
    if not os.path.isdir(out):
        print(f"{out} does not exist; run without --check first")
        return 1

    problems, notes = [], []
    man = os.path.join(out, "overleaf", "main.tex")
    if not os.path.exists(man):
        print("overleaf/main.tex is missing; rebuild")
        return 1
    text = open(man, encoding="utf-8").read()

    if "CITY" in text or "COUNTRY" in text:
        problems.append("affiliation placeholders CITY/COUNTRY are unfilled")

    if not [f for f in os.listdir(out) if f.lower().endswith(".pdf")]:
        problems.append("no compiled PDF in the package; Springer expects one "
                        "you have compiled and checked yourself")

    for stray in ("cas-model2-names", "cas-sc", "\\bibliographystyle"):
        if stray in text:
            problems.append(f"Elsevier leftover in main.tex: {stray}")

    src = open(os.path.join(HERE, "manuscript_emse.tex"), encoding="utf-8").read()
    if src != text:
        problems.append("overleaf/main.tex is stale; rebuild with "
                        "python make_submission.py")

    if not os.path.isdir(os.path.join(out, "supplementary", "corpora")):
        problems.append("supplementary/corpora is missing")

    n_bib = len(re.findall(r"\\bibitem", text))
    n_doi = len(re.findall(r"doi\.org|\\doi\{", text, re.I))
    if n_doi < n_bib:
        notes.append(f"{n_doi} of {n_bib} references carry a DOI link")

    # The archived-artifact DOI must be a DIFFERENT identifier from the
    # preregistration. Counting occurrences of "OSF.IO" does not establish
    # that: the preregistration DOI alone appears more than once.
    PREREG = "OSF.IO/NZXRV"
    others = [d for d in re.findall(r"10\.\d{4,9}/[^\s{}$,)]+", text)
              if PREREG.lower() not in d.lower()]
    if not others:
        notes.append("no archived-artifact DOI in the manuscript; the only "
                     "DOI present is the preregistration, which covers one "
                     "analysis and is not the replication package")

    for p in problems:
        print(f"BLOCK  {p}")
    for n in notes:
        print(f"note   {n}")
    if not problems and not notes:
        print("ready")
    print(f"\n{len(problems)} blocking, {len(notes)} to consider")
    return 1 if problems else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="report readiness without rebuilding")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="package directory (default: alongside the repo)")
    args = ap.parse_args()
    if not args.check:
        build(args.out)
    return check(args.out)


if __name__ == "__main__":
    sys.exit(main())

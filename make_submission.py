#!/usr/bin/env python3
"""Assemble the EMSE submission folder, and report what is still missing.

Submission goes through Editorial Manager at
https://www.editorialmanager.com/emse/ and Springer Nature asks for the LaTeX
source, the bibliography, figures in the same directory, and a PDF that you
have compiled yourself, all zipped.

This builds submission_emse/ from the repository and then runs a readiness
check. The check is the point: it fails loudly on the things that are easy to
forget and impossible to fix after upload, including the affiliation
placeholders and the missing compiled PDF.

Nothing here compiles LaTeX. There is no toolchain in this environment, so
the PDF has to come from Overleaf.

Usage:
    python make_submission.py
    python make_submission.py --check    # report readiness, build nothing
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "submission_emse")

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


def build():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "supplementary", "corpora"))

    shutil.copy(os.path.join(HERE, "manuscript_emse.tex"),
                os.path.join(OUT, "manuscript.tex"))
    shutil.copy(os.path.join(HERE, "cover_letter.tex"),
                os.path.join(OUT, "cover_letter.tex"))

    for label, directory, prefix in CORPORA:
        dst = os.path.join(OUT, "supplementary", "corpora", label)
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
            shutil.copy(src, os.path.join(OUT, "supplementary", name))

    with open(os.path.join(OUT, "README.md"), "w", newline="\n",
              encoding="utf-8") as f:
        f.write(readme())
    print(f"built {os.path.relpath(OUT, HERE)}/ from commit {git_commit()[:8]}")


def readme():
    return f"""# EMSE submission package

Built by `make_submission.py` from commit `{git_commit()}`.
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
"""


def check():
    problems, notes = [], []

    man = os.path.join(OUT, "manuscript.tex")
    if not os.path.isdir(OUT):
        print("submission_emse/ does not exist; run without --check first")
        return 1

    text = open(man, encoding="utf-8").read()

    if "CITY" in text or "COUNTRY" in text:
        problems.append("affiliation placeholders CITY/COUNTRY are unfilled")

    pdfs = [f for f in os.listdir(OUT) if f.lower().endswith(".pdf")]
    if not pdfs:
        problems.append("no compiled PDF in the folder; Springer expects one "
                        "you have compiled and checked yourself")

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

    if not os.path.isdir(os.path.join(OUT, "supplementary", "corpora")):
        problems.append("supplementary/corpora is missing")

    # Is the built copy current with the repository?
    src = open(os.path.join(HERE, "manuscript_emse.tex"), encoding="utf-8").read()
    if src != text:
        problems.append("manuscript.tex is stale; rebuild with "
                        "python make_submission.py")

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
    args = ap.parse_args()
    if not args.check:
        build()
    return check()


if __name__ == "__main__":
    sys.exit(main())

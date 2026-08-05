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


def esm_readme():
    return """Online Resource 1

Article : Cross-Corpus Evaluation of Domain-Aware Query Reformulation for
          Regulatory Traceability
Journal : Empirical Software Engineering
Authors : Sumanshu Sohal (corresponding), University of the Cumberlands,
          Williamsburg, KY, USA, and Independent Researcher, Washington, DC,
          USA. sumanshu.95s@outlook.com
          Darshankumar Prajapati, Independent Researcher, Old Bridge, NJ, USA.

CONTENTS

  corpora/diagnostic/            58 requirements, 110 controls, 86 links.
                                 Author-constructed to isolate vocabulary
                                 mismatch.
  corpora/csf_1.1/               106 subcategories, 300 controls, 495 links.
  corpora/hipaa/                 68 requirements, 300 controls, 274 links.
  corpora/privacy_framework/     94 subcategories, 300 controls, 456 links.

  Each corpus directory holds regs.csv, controls.csv and mappings.csv. The
  three NIST corpora also carry provenance.json, written by the builder that
  generated them, recording every source artifact with its SHA-256.

  REPRODUCE.md    how to regenerate every number in the article
  SOURCES.md      provenance and SHA-256 of every NIST source artifact
  LICENSE         MIT, covering the software in the code repository
  LICENSE-DATA    CC BY 4.0, covering these corpora

The three NIST corpora share one 300-control corpus drawn from SP 800-53r5,
so their controls.csv files are byte-identical. They are externally authored
but not independent of one another, which the article treats as its main
limitation.

The underlying NIST artifacts are United States Government works in the
public domain. Our contribution is the extraction and encoding, not the text
or the mappings.

Code, including the evaluation harness and the analysis scripts, is at
https://github.com/sumanshusohal/RAA-Compliance-Mapping
The preregistration of the gated hybrid analysis is at
https://doi.org/10.17605/OSF.IO/NZXRV
"""


def build(out):
    # The compiled PDF is placed here by hand, since nothing in this
    # environment can produce it. Rebuilding used to delete it along with
    # everything else, so a rebuild after any edit silently threw away the
    # one artifact the readiness check insists on.
    keep = {}
    if os.path.isdir(out):
        for f in os.listdir(out):
            if f.lower().endswith(".pdf"):
                with open(os.path.join(out, f), "rb") as fh:
                    keep[f] = fh.read()
        shutil.rmtree(out)
    overleaf = os.path.join(out, "overleaf")
    os.makedirs(overleaf)
    os.makedirs(os.path.join(out, "supplementary", "corpora"))

    # overleaf/ belongs to make_emse.py: it writes main.tex and copies the
    # vendored class files. Calling it here rather than copying its output
    # keeps one source of truth. An earlier version of this script wrote its
    # own main.tex and README into overleaf/, which silently reverted the
    # class files and restored a stale README describing the wrong class.
    rc = subprocess.run([sys.executable, os.path.join(HERE, "make_emse.py")],
                        cwd=HERE, capture_output=True, text=True)
    if rc.returncode != 0:
        print(rc.stdout)
        raise SystemExit("make_emse.py failed; package not built")
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

    supp = os.path.join(out, "supplementary")
    with open(os.path.join(supp, "README.txt"), "w", newline="\n",
              encoding="utf-8") as f:
        f.write(esm_readme())

    # Springer wants the supplement as a single archive. Built here so it
    # cannot drift from the corpora it is supposed to contain.
    import zipfile
    esm = os.path.join(out, "ESM_1.zip")
    with zipfile.ZipFile(esm, "w", zipfile.ZIP_DEFLATED) as z:
        for dp, _, fs in os.walk(supp):
            for f in sorted(fs):
                full = os.path.join(dp, f)
                z.write(full, os.path.relpath(full, supp))

    with open(os.path.join(out, "README.md"), "w", newline="\n",
              encoding="utf-8") as f:
        f.write(package_readme())

    for name, blob in keep.items():
        with open(os.path.join(out, name), "wb") as fh:
            fh.write(blob)
        print(f"  preserved {name}")

    print(f"built {out}\n  from commit {git_commit()[:8]}")


def overleaf_readme():
    return """# Overleaf build folder

Upload this whole folder to Overleaf and compile `main.tex` with **pdfLaTeX**.
Nothing else is needed; the class files are here.

| file | why it is here |
|---|---|
| `main.tex` | the manuscript, `svjour3` with the `smallextended` and `natbib` options |
| `svjour3.cls` | Springer's journal class, required |
| `svglov3.clo` | the class's global option file, loaded by `svjour3.cls` |
| `spbasic.bst` | Springer author-year BibTeX style. **Not used**, see below |

No figure files: the paper has no figures, only `tabular` environments.

`spbasic.bst` is included only for completeness. The bibliography is an inline
`thebibliography` environment with all 56 entries written out, so BibTeX never
runs.

When uploading to Editorial Manager, upload these files **flat**. EMSE asks
that the LaTeX source bundle not use subfolders.

## What the header follows

Checked against `usrguid3.pdf`, the SVJour3 user's guide v3.2: the format
option first in `\\documentclass` with `natbib` added for author-year
citation; `\\author` entries separated by `\\and`; `\\institute` repeating each
author name with `\\at` before their address; `\\titlerunning` and
`\\authorrunning` before `\\maketitle`; `\\keywords` at the end of but inside
the `abstract`; `\\subclass`, `\\PACS` and `\\CRclass` omitted;
acknowledgements as an environment, closing before the bibliography.

`\\date{Received: date / Accepted: date}` is left as the template supplies it.
The template notes that the editor enters the real dates.

## What to look at in the PDF

- the twelve tables, especially the seven-column unified comparison and the
  factorial, which has a `\\cmidrule`-spanned header;
- the `algorithm` float;
- page breaks around the wider tables;
- that `\\citep` renders as author-year rather than question marks, which
  would mean `natbib` is not active;
- the author block and all three affiliation footnotes on page 1.

## Do not edit this copy

`main.tex` is generated from `manuscript3_revised.tex` by `make_emse.py` in
the code repository. Edit the source and regenerate; edits made here are lost
on the next build.
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
| `overleaf/` | `main.tex` plus the SVJour3 class files; upload flat and compile with pdfLaTeX |
| `ESM_1.zip` | Online Resource 1, the supplement the article cites |
| `cover_letter.tex` | addressed to the EMSE editors |
| `supplementary/` | the unzipped contents of ESM_1.zip, for inspection |

## Before you upload

Run `python make_submission.py --check` and clear what it reports.

1. **Compile and read the PDF**, then put it in this folder. See
   `overleaf/README.md`. Springer states that files which do not compile will
   most likely fail to submit.
2. **Upload `ESM_1.zip` as Online Resource 1.** The article's data
   availability statement cites it by that name, so it has to be there.
3. **Archive the artifact and cite its DOI.** Tag a release, deposit it on
   Zenodo or a new OSF component, and add that DOI to the data availability
   statement. This is separate from the preregistration
   (doi:10.17605/OSF.IO/NZXRV), which is immutable and covers one analysis.
4. **DOIs on the references.** 40 of 56 carry one. The remainder are mostly
   NeurIPS, ICLR and arXiv items that have no Crossref DOI; `add_dois.py`
   lists them.

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

    pdfs = [os.path.join(dp, f)
            for dp, _, fs in os.walk(out)
            for f in fs if f.lower().endswith(".pdf")]
    if not pdfs:
        problems.append("no compiled PDF anywhere in the package; Springer "
                        "expects one you have compiled and checked yourself")

    for stray in ("cas-model2-names", "cas-sc", "\\bibliographystyle"):
        if stray in text:
            problems.append(f"Elsevier leftover in main.tex: {stray}")

    # overleaf/main.tex is the file that gets uploaded and becomes the paper,
    # so it has to match the SOURCE, not merely match manuscript_emse.tex.
    # Comparing the two generated files was no guarantee: if make_emse.py was
    # not rerun after an edit to manuscript3_revised.tex, both go stale
    # together and the comparison still passes. Convert the source here and
    # compare against that.
    try:
        import make_emse
        fresh = make_emse.convert(
            open(os.path.join(HERE, "manuscript3_revised.tex"),
                 encoding="utf-8").read())
    except Exception as exc:
        problems.append(f"could not re-derive main.tex from the source: {exc}")
        fresh = None
    if fresh is not None and fresh != text:
        a = fresh.split("\n")
        b = text.split("\n")
        where = next((i + 1 for i in range(max(len(a), len(b)))
                      if (a[i] if i < len(a) else None)
                      != (b[i] if i < len(b) else None)), 0)
        problems.append(
            "overleaf/main.tex does not match manuscript3_revised.tex "
            f"(first difference at line {where}); this is the file that gets "
            "uploaded, so rebuild with python make_submission.py")

    if not os.path.isdir(os.path.join(out, "supplementary", "corpora")):
        problems.append("supplementary/corpora is missing")
    if not os.path.exists(os.path.join(out, "ESM_1.zip")):
        problems.append("ESM_1.zip is missing; the article cites Online "
                        "Resource 1")
    if "Online Resource" not in text:
        notes.append("the manuscript does not mention Online Resource 1")

    # Count DOIs inside the bibliography only. Counting the whole file also
    # caught the OSF preregistration DOI in the declarations, so 39
    # references were reported as 40.
    try:
        bib = text[text.index(r"\begin{thebibliography}"):
                   text.index(r"\end{thebibliography}")]
    except ValueError:
        bib = ""
    n_bib = len(re.findall(r"\\bibitem", bib))
    n_doi = len(re.findall(r"doi\.org|\\doi\{", bib, re.I))
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

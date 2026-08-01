#!/usr/bin/env python3
"""Build the NIST Privacy Framework to SP 800-53 benchmark.

Narrow-gap counterpart to the HIPAA corpus: both sides are NIST-authored, so
the regulation-to-implementation vocabulary gap that reformulation targets is
largely absent here. This corpus carries the study's equivalence prediction.

Requirements : 'PF to SP 800-53r5' sheet of NIST's published crosswalk
Ground truth : the SP 800-53 control references in that sheet
Controls     : the shared SP 800-53 base-control corpus (OSCAL, pinned tag)

The PF sheet's layout differs from the CSF sheet in the same workbook: its
header sits on the third row, the function label is merged across the first
two columns, and control references appear in column 4 rather than 3. It also
uses control enhancements (CM-8(4)), which the CSF sheet does not.

Run from the repo root or from this directory:
    python pf_benchmark/build_pf_benchmark.py
"""
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from corpus_common import (  # noqa: E402
    OSCAL_SHA256, OSCAL_TAG, OSCAL_URL, download, load_oscal_controls,
    norm_code, sha256, split_codes, write_corpus, write_provenance,
)

XLSX_URL = ("https://csrc.nist.gov/files/pubs/sp/800/53/r5/upd1/final/docs/"
            "csf-pf-to-sp800-53r5-mappings.xlsx")
XLSX_SHA256 = "8b09e8c5bd11dcfa494d2f5beec2d32d98df934285c0214712d020abefc162f1"
SHEET = "PF to SP 800-53r5"

# Column positions in the PF sheet (header on row index 2, data from row 3).
COL_SUBCATEGORY = 3
COL_CONTROLS = 4

FUNCTION_NAMES = {
    "ID": "Identify-P", "GV": "Govern-P", "CT": "Control-P",
    "CM": "Communicate-P", "PR": "Protect-P",
}


def main():
    outdir = _HERE
    print("Fetching the shared SP 800-53 control catalog:")
    catalog_path = download(OSCAL_URL,
                            os.path.join(outdir, "sp800-53-catalog.json"),
                            expected_sha256=OSCAL_SHA256)
    controls, codes, code_to_index, catalog_version = load_oscal_controls(
        catalog_path)

    print("\nFetching the NIST CSF/PF crosswalk workbook:")
    xlsx_path = download(XLSX_URL,
                         os.path.join(outdir, "csf-pf-to-sp800-53r5.xlsx"),
                         expected_sha256=XLSX_SHA256)

    sheet = pd.read_excel(xlsx_path, sheet_name=SHEET, header=None)
    rows = sheet.iloc[3:]

    regs, mappings, audit = [], [], []
    n_seen = n_unlinked = n_unresolved = n_unparsed = 0

    for _, row in rows.iterrows():
        sub = str(row[COL_SUBCATEGORY]).strip()
        if not sub or sub == "nan" or ":" not in sub:
            continue
        n_seen += 1

        wanted = set()
        for token in split_codes(row[COL_CONTROLS]):
            code = norm_code(token)
            if code is None:
                # "all -1 controls" and similar blanket references cannot
                # serve as retrieval ground truth, exactly as in the CSF build.
                n_unparsed += 1
                continue
            wanted.add(code)

        rows_idx = []
        for code in sorted(wanted):
            idx = code_to_index.get(code)
            if idx is None:
                n_unresolved += 1
                continue
            rows_idx.append(idx)
        if not rows_idx:
            n_unlinked += 1
            continue

        ident = sub.split(":", 1)[0].strip()
        function = ident.split(".")[0]
        rid = len(regs)
        regs.append((rid, f"PF-{FUNCTION_NAMES.get(function, function)}", sub))
        audit.append((rid, ident, function))
        mappings.extend((rid, idx) for idx in rows_idx)

    print(f"\n  subcategories seen             : {n_seen}")
    print(f"  dropped, no usable link        : {n_unlinked}")
    print(f"  blanket refs skipped           : {n_unparsed}")
    print(f"  unresolved control references  : {n_unresolved}")

    print("\nWriting corpus:")
    write_corpus(outdir, regs, controls, mappings, control_codes=codes)

    pd.DataFrame(audit, columns=["id", "pf_identifier", "function"]).to_csv(
        os.path.join(outdir, "reg_codes.csv"), index=False)

    write_provenance(outdir, sources=[
        {"artifact": "SP 800-53 control catalog (OSCAL)",
         "url": OSCAL_URL, "release_tag": OSCAL_TAG,
         "catalog_version": catalog_version, "sha256": sha256(catalog_path)},
        {"artifact": "NIST Privacy Framework to SP 800-53r5 crosswalk",
         "url": XLSX_URL, "sheet": SHEET, "sha256": sha256(xlsx_path)},
    ], notes={
        "subcategories_seen": n_seen,
        "dropped_no_link": n_unlinked,
        "blanket_references_skipped": n_unparsed,
        "unresolved_control_references": n_unresolved,
        "enhancement_handling": "collapsed to base control",
        "independence_check": (
            "PF subcategory identifiers and text have zero overlap with the "
            "CSF corpus, so the exploratory/confirmatory split is clean."),
    })


if __name__ == "__main__":
    main()

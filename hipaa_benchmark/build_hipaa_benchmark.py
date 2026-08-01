#!/usr/bin/env python3
"""Build the HIPAA Security Rule to SP 800-53 benchmark from NIST sources.

This is the study's only wide-gap, regulation-to-control corpus: the source
side is statutory language (45 CFR Part 164 Subpart C, as reproduced by NIST
in SP 800-66 Rev 2) and the target side is implementation-level control text.

Requirements  : 'standard' and 'imp_spec' elements of CPRT SP800_66_2_0_0
Ground truth  : each element's externalRelationships whose OLIR target is
                SP-800-66-Rev-2-to-SP-800-53-Rev-5.1.1
Controls      : the shared SP 800-53 base-control corpus (OSCAL, pinned tag)

Both sides are public domain. Note that CPRT's documented /export endpoint is
broken (404 json, 500 excel), so the /graph endpoints are used instead.

Run from the repo root or from this directory:
    python hipaa_benchmark/build_hipaa_benchmark.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from corpus_common import (  # noqa: E402
    OSCAL_RELEASE, OSCAL_SHA256, OSCAL_TAG, OSCAL_URL,
    cprt_element_graph, cprt_version_graph, cprt_walk, download,
    load_oscal_controls, norm_code, sha256, write_corpus, write_provenance,
)

CPRT_VERSION = "SP800_66_2_0_0"
OLIR_TARGET = "SP-800-66-Rev-2-to-SP-800-53-Rev-5.1.1"
REQ_TYPES = ("standard", "imp_spec")

# The five Security Rule safeguard sections, used as the stratification key.
SECTION_NAMES = {
    "164.308": "Administrative",
    "164.310": "Physical",
    "164.312": "Technical",
    "164.314": "Organizational",
    "164.316": "Documentation",
}


def collect_requirements(section_id, snapshot_dir):
    """Pull one safeguard section and return its requirement elements."""
    tree = cprt_element_graph(CPRT_VERSION, section_id, snapshot_dir)["elements"]
    found = []

    def visit(node):
        if node.get("elementTypeIdentifier") in REQ_TYPES:
            found.append(node)

    cprt_walk(tree, visit)
    return found


def linked_controls(node):
    """Normalized SP 800-53 control codes linked to a requirement element.

    Enhancements collapse to their base control: CPRT writes zero-padded
    forms such as CP-02(08), and the control corpus holds base controls only.
    """
    out = set()
    for rel in node.get("externalRelationships") or []:
        if OLIR_TARGET not in (rel.get("olirName") or ""):
            continue
        code = norm_code(rel.get("elementIdentifier"))
        if code:
            out.add(code)
    return out


def main():
    outdir = _HERE
    print("Fetching the shared SP 800-53 control catalog:")
    catalog_path = download(OSCAL_URL,
                            os.path.join(outdir, "sp800-53-catalog.json"),
                            expected_sha256=OSCAL_SHA256)
    controls, codes, code_to_index, catalog_version = load_oscal_controls(
        catalog_path)

    print(f"\nPulling HIPAA requirements from CPRT {CPRT_VERSION}:")
    snapshot_dir = os.path.join(outdir, "cprt_snapshots")
    roots = cprt_version_graph(CPRT_VERSION, snapshot_dir)["elements"]
    root_ids = [r["elementIdentifier"] for r in roots]
    print(f"  {len(root_ids)} safeguard sections: {', '.join(root_ids)}")

    regs, mappings, audit = [], [], []
    n_seen = n_unlinked = n_unresolved = 0

    for section in root_ids:
        for node in collect_requirements(section, snapshot_dir):
            n_seen += 1
            ident = node["elementIdentifier"]
            wanted = linked_controls(node)
            if not wanted:
                # Some standards carry only CSF references, no 800-53 links.
                # They cannot serve as retrieval ground truth.
                n_unlinked += 1
                continue

            rows = []
            for code in sorted(wanted):
                idx = code_to_index.get(code)
                if idx is None:
                    n_unresolved += 1
                    continue
                rows.append(idx)
            if not rows:
                n_unlinked += 1
                continue

            rid = len(regs)
            title = (node.get("title") or "").strip()
            body = " ".join((node.get("text") or "").split())
            regs.append((rid, f"HIPAA-{SECTION_NAMES.get(section, section)}",
                         f"{ident} {title}: {body}".strip()))
            audit.append((rid, ident, node["elementTypeIdentifier"], section))
            mappings.extend((rid, idx) for idx in rows)

    print(f"\n  requirement elements seen      : {n_seen}")
    print(f"  dropped, no usable 800-53 link : {n_unlinked}")
    print(f"  unresolved control references  : {n_unresolved}")

    print("\nWriting corpus:")
    write_corpus(outdir, regs, controls, mappings, control_codes=codes)

    import pandas as pd
    pd.DataFrame(audit, columns=["id", "hipaa_identifier", "element_type",
                                 "section"]).to_csv(
        os.path.join(outdir, "reg_codes.csv"), index=False)

    write_provenance(outdir, sources=[
        {"artifact": "SP 800-53 control catalog (OSCAL)",
         "url": OSCAL_URL, "release_tag": OSCAL_TAG,
         "catalog_version": catalog_version, "sha256": sha256(catalog_path)},
        {"artifact": "HIPAA Security Rule requirements and 800-53 mapping",
         "source": "NIST CPRT", "framework_version": CPRT_VERSION,
         "olir": OLIR_TARGET,
         "endpoint": "json/nudp/framework/version/{v}/element/{id}/graph",
         "snapshots": sorted(
             (f, sha256(os.path.join(snapshot_dir, f)))
             for f in os.listdir(snapshot_dir))},
    ], notes={
        "requirement_types": list(REQ_TYPES),
        "requirements_seen": n_seen,
        "requirements_dropped_no_link": n_unlinked,
        "unresolved_control_references": n_unresolved,
        "enhancement_handling": "collapsed to base control",
        "catalog_release_note": (
            f"The OLIR targets 800-53 Rev 5.1.1 while the catalog is "
            f"{OSCAL_RELEASE}. The releases differ by one control (SA-24) and "
            f"every ground-truth link resolves in both, so the shared "
            f"{OSCAL_RELEASE} catalog is used for cross-corpus comparability."),
    })


if __name__ == "__main__":
    main()

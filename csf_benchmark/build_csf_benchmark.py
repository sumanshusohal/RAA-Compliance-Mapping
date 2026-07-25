#!/usr/bin/env python3
"""Build a real-world compliance-mapping benchmark from NIST-published data.

Requirements side : NIST CSF 1.1 subcategories (108 items)
Controls side     : NIST SP 800-53 Rev 5 base controls (OSCAL catalog)
Ground truth      : NIST's official CSF-to-800-53r5 mapping spreadsheet

Inputs (download into this directory before running; see SOURCES.md):
  csf-to-80053r5-mappings.xlsx
    https://csrc.nist.gov/files/pubs/sp/800/53/r5/upd1/final/docs/csf-pf-to-sp800-53r5-mappings.xlsx
    SHA-256 8b09e8c5bd11dcfa494d2f5beec2d32d98df934285c0214712d020abefc162f1
  sp800-53r5-catalog.json  (OSCAL catalog)
    https://raw.githubusercontent.com/usnistgov/oscal-content/main/nist.gov/SP800-53/rev5/json/NIST_SP-800-53_rev5_catalog.json
    SHA-256 01f37cf90ea99d92242c936cbfbdebcc338eef1f71454e2acac36cc56e9bc062

Outputs (RAA harness format): csf_regs.csv, csf_controls.csv,
csf_control_codes.csv (control-id -> SP 800-53 code, for auditing), csf_mappings.csv.
"""
import hashlib
import json
import re
import sys

import pandas as pd

XLSX = "csf-to-80053r5-mappings.xlsx"
CATALOG = "sp800-53r5-catalog.json"
OUTDIR = "."


EXPECTED_SHA256 = {
    XLSX: "8b09e8c5bd11dcfa494d2f5beec2d32d98df934285c0214712d020abefc162f1",
    CATALOG: "01f37cf90ea99d92242c936cbfbdebcc338eef1f71454e2acac36cc56e9bc062",
}


def verify_checksums():
    for path, expected in EXPECTED_SHA256.items():
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        actual = h.hexdigest()
        status = "OK" if actual == expected else "MISMATCH"
        print(f"  {status}  {path}  {actual}")
        if actual != expected:
            print(f"    expected {expected}")


def norm_code(code):
    """Normalize a control code like 'AC-01' / 'AC-1' -> 'AC-1'. None if not a code."""
    m = re.match(r"^([A-Z]{2})-0*(\d+)$", code.strip().replace(" ", "").upper())
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}"


def resolve_params(text, params_by_id):
    def sub(m):
        pid = m.group(1).strip()
        p = params_by_id.get(pid)
        if p is None:
            return "[organization-defined value]"
        if "label" in p:
            return f"[Assignment: {p['label']}]"
        sel = p.get("select")
        if sel:
            choices = "; ".join(sel.get("choice", []))
            return f"[Selection: {choices}]"
        return "[organization-defined value]"
    pat = r"\{\{\s*insert:\s*param,\s*([^}\s]+)\s*\}\}"
    for _ in range(4):
        if not re.search(pat, text):
            break
        text = re.sub(pat, sub, text)
    return text


def collect_prose(part, params_by_id, depth=0, max_depth=2):
    """Collect statement prose from an OSCAL part, up to max_depth levels."""
    out = []
    prose = part.get("prose")
    if prose:
        out.append(resolve_params(prose, params_by_id))
    if depth < max_depth:
        for sub in part.get("parts", []):
            if sub.get("name") == "item":
                out.extend(collect_prose(sub, params_by_id, depth + 1, max_depth))
    return out


def control_text(ctrl):
    params_by_id = {p["id"]: p for p in ctrl.get("params", [])}
    title = ctrl.get("title", "")
    statement_prose = []
    for part in ctrl.get("parts", []):
        if part.get("name") == "statement":
            statement_prose = collect_prose(part, params_by_id)
            break
    text = title + ". " + " ".join(statement_prose)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_withdrawn(ctrl):
    for prop in ctrl.get("props", []):
        if prop.get("name") == "status" and prop.get("value") == "withdrawn":
            return True
    return False


def main():
    print("Verifying source checksums:")
    verify_checksums()
    # ---- Controls from OSCAL catalog (base controls only) ----
    with open(CATALOG, encoding="utf-8") as f:
        catalog = json.load(f)["catalog"]

    controls = []       # (ctrl_code, family_title, text)
    code_to_idx = {}
    for group in catalog.get("groups", []):
        family = group.get("title", "general")
        for ctrl in group.get("controls", []):
            if is_withdrawn(ctrl):
                continue
            code = None
            for prop in ctrl.get("props", []):
                if prop.get("name") == "label":
                    code = prop["value"].strip()
                    break
            if code is None:
                code = ctrl["id"].upper()
            text = control_text(ctrl)
            if len(text) < 20:
                continue
            code_norm = norm_code(code)
            if code_norm is None:
                continue
            code_to_idx[code_norm] = len(controls)
            controls.append((code_norm, family, text))

    print(f"Base controls extracted: {len(controls)}")

    # ---- Requirements + ground truth from NIST mapping spreadsheet ----
    df = pd.read_excel(XLSX, sheet_name="CSF to SP 800-53r5", header=1)
    df.columns = ["function", "category", "subcategory", "controls"]
    df["function"] = df["function"].ffill()

    regs = []
    mappings = []
    n_links_total = 0
    n_links_matched = 0
    unmatched = set()
    for _, row in df.iterrows():
        sub = str(row["subcategory"]).strip()
        if not sub or sub == "nan":
            continue
        func = str(row["function"]).strip()
        fw = "CSF-" + func.split("(")[-1].rstrip(")") if "(" in func else "CSF"

        raw = str(row["controls"])
        if raw == "nan":
            continue
        row_links = []
        for code in re.split(r"[,;]", raw):
            code_norm = norm_code(code)
            if code_norm is None:
                if code.strip():
                    unmatched.add(code.strip()[:40])
                continue
            n_links_total += 1
            idx = code_to_idx.get(code_norm)
            if idx is None:
                unmatched.add(code_norm)
                continue
            n_links_matched += 1
            row_links.append(idx)

        if not row_links:
            # NIST maps this subcategory to "all controls" or nothing concrete;
            # unusable as retrieval ground truth.
            continue
        rid = len(regs)
        regs.append((rid, fw, sub))
        mappings.extend((rid, idx) for idx in row_links)

    print(f"Requirements: {len(regs)}")
    print(f"GT links: {n_links_matched}/{n_links_total} matched to catalog")
    if unmatched:
        print(f"Unmatched control codes ({len(unmatched)}): {sorted(unmatched)[:20]}")

    # ---- Write CSVs ----
    pd.DataFrame(regs, columns=["id", "framework", "text"]).to_csv(
        f"{OUTDIR}/csf_regs.csv", index=False)
    pd.DataFrame(
        [(i, t, fam) for i, (code, fam, t) in enumerate(controls)],
        columns=["id", "text", "family"]).to_csv(f"{OUTDIR}/csf_controls.csv", index=False)
    pd.DataFrame(
        [(code,) for code, _, _ in controls], columns=["code"]).to_csv(
        f"{OUTDIR}/csf_control_codes.csv", index=False)
    pd.DataFrame(mappings, columns=["regulation_id", "control_id"]).to_csv(
        f"{OUTDIR}/csf_mappings.csv", index=False)

    multi = sum(1 for r in set(m[0] for m in mappings)
                if sum(1 for x in mappings if x[0] == r) >= 2)
    print(f"Requirements with >=2 mapped controls: {multi}")
    print("Wrote csf_regs.csv, csf_controls.csv, csf_mappings.csv")


if __name__ == "__main__":
    main()

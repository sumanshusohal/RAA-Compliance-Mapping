#!/usr/bin/env python3
"""Shared plumbing for the corpus builders.

Every benchmark in this study is regenerated from a pinned public-domain
source. This module supplies the four things all builders need:

  1. download + SHA-256 verification that HARD FAILS (exits non-zero) on a
     mismatch, so a silently changed upstream file can never produce a corpus;
  2. CPRT API access, since the documented /export endpoint is broken
     (404 for json, 500 for excel) and the /graph endpoints must be used;
  3. one control-code normalizer shared by every builder, because the sources
     disagree on form (CM-8, CP-02(08), CM-8(4));
  4. corpus writing plus a provenance record naming every source, its hash,
     and the catalog release the mapping targets.

Builders run either from the repo root or from their own directory.
"""
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

CPRT_BASE = "https://csrc.nist.gov/extensions/nudp/services/"
USER_AGENT = "RAA-corpus-builder/1.0 (research; github.com/sumanshusohal/RAA-Compliance-Mapping)"
TIMEOUT = 180
RETRIES = 3


# ---------------------------------------------------------------- hashing --

def sha256(path):
    """SHA-256 of a file, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def die(msg):
    """Abort the build. Corpora must never be produced from unverified input."""
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


# -------------------------------------------------------------- downloads --

def download(url, dest, expected_sha256=None, force=False):
    """Fetch url to dest and verify its hash. Exits non-zero on mismatch.

    Unlike the previous builder, a hash mismatch is fatal rather than a
    printed warning: a changed upstream file silently rewriting the ground
    truth is exactly the failure this study cannot tolerate.
    """
    dest = str(dest)
    if force or not _exists(dest):
        last = None
        for attempt in range(1, RETRIES + 1):
            try:
                r = requests.get(url, headers={"User-Agent": USER_AGENT},
                                 timeout=TIMEOUT, allow_redirects=True)
                if r.status_code == 200:
                    with open(dest, "wb") as f:
                        f.write(r.content)
                    break
                last = f"HTTP {r.status_code}"
            except requests.RequestException as e:
                last = str(e)
            print(f"  attempt {attempt}/{RETRIES} failed ({last}); retrying")
            time.sleep(2 * attempt)
        else:
            die(f"could not download {url}: {last}")

    actual = sha256(dest)
    if expected_sha256 is None:
        print(f"  {dest}  sha256 {actual}  (unpinned)")
    elif actual != expected_sha256:
        die(f"checksum mismatch for {dest}\n"
            f"  expected {expected_sha256}\n"
            f"  actual   {actual}\n"
            f"  The upstream artifact changed. Re-pin deliberately after "
            f"reviewing the diff; do not overwrite the expectation blindly.")
    else:
        print(f"  {dest}  sha256 OK")
    return dest


def _exists(path):
    try:
        with open(path, "rb"):
            return True
    except OSError:
        return False


# ------------------------------------------------------------------ CPRT --

def cprt_get(path, snapshot_dir=None):
    """GET a CPRT service path, returning the parsed 'response' object.

    If snapshot_dir is given, the RAW response body is written there and
    hashed. CPRT is a live service with no version pinning on its graph
    endpoints, so without a stored snapshot a rebuild could silently produce
    a different corpus. The snapshot plus its SHA-256 is what makes the
    corpus reproducible; a cached snapshot is reused in preference to
    refetching.
    """
    if snapshot_dir is not None:
        os.makedirs(snapshot_dir, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", path)
        cached = os.path.join(snapshot_dir, f"{safe}.json")
        if os.path.exists(cached):
            with open(cached, encoding="utf-8") as f:
                data = json.load(f)
            print(f"    [snapshot] {safe[:60]}  sha256 {sha256(cached)[:16]}")
            return data.get("response", data)

    url = CPRT_BASE + path
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT,
                                           "Accept": "application/json"},
                             timeout=TIMEOUT)
            if r.status_code == 200:
                if snapshot_dir is not None:
                    with open(cached, "wb") as f:
                        f.write(r.content)
                    print(f"    [fetched] {safe[:58]}  "
                          f"sha256 {sha256(cached)[:16]}")
                data = r.json()
                return data.get("response", data)
            last = f"HTTP {r.status_code}"
        except (requests.RequestException, ValueError) as e:
            last = str(e)
        print(f"  attempt {attempt}/{RETRIES} failed ({last}); retrying")
        time.sleep(2 * attempt)
    die(f"CPRT request failed: {url}")


def cprt_version_graph(version_id, snapshot_dir=None):
    """Root elements of a CPRT framework version."""
    return cprt_get(f"json/nudp/framework/version/{version_id}/graph",
                    snapshot_dir)


def cprt_element_graph(version_id, element_id, snapshot_dir=None):
    """Full nested subtree under one root element."""
    return cprt_get(
        f"json/nudp/framework/version/{version_id}/element/{element_id}/graph",
        snapshot_dir)


def cprt_walk(nodes, visit):
    """Depth-first walk over a CPRT element tree, calling visit(node)."""
    for node in nodes:
        visit(node)
        kids = node.get("elements")
        if isinstance(kids, list):
            cprt_walk(kids, visit)


# ------------------------------------------------------- code normalizing --

# Sources disagree on control-code form. The CSF sheet writes CM-8, the PF
# sheet writes CM-8(4), CPRT writes CP-02(08). All three must land on the
# same key as the OSCAL catalog, which uses the unpadded base form.
_CODE_RE = re.compile(r"^([A-Z]{2})-0*(\d+)(?:\s*\(\s*0*(\d+)\s*\))?$")


def norm_code(code, keep_enhancement=False):
    """Normalize a control reference to catalog form, or None if unusable.

    'AC-1', 'AC-01', 'ac-1'            -> 'AC-1'
    'CP-02(08)', 'CM-8(4)'             -> 'CM-8'      (base control)
                                       -> 'CM-8(4)'   if keep_enhancement
    'all -1 controls', '', 'nan'       -> None

    Enhancements collapse to their base control by default because the
    control corpus contains base controls only; dropping them instead would
    silently discard roughly a tenth of the PF ground truth.
    """
    if code is None:
        return None
    s = str(code).strip().upper()
    if not s or s == "NAN":
        return None
    s = s.replace(" ", "")
    m = _CODE_RE.match(s)
    if not m:
        return None
    family, num, enh = m.group(1), int(m.group(2)), m.group(3)
    base = f"{family}-{num}"
    if enh and keep_enhancement:
        return f"{base}({int(enh)})"
    return base


def split_codes(raw):
    """Split a delimited control-reference cell into raw tokens."""
    if raw is None:
        return []
    s = str(raw)
    if s.strip().lower() in ("", "nan"):
        return []
    return [t.strip() for t in re.split(r"[,;]", s) if t.strip()]


# ------------------------------------------------------ OSCAL catalog ----

# Pinned to an immutable release tag, not to main. The previous pin used a
# main-branch URL, whose hash only matched because NIST had not yet published
# a newer catalog; the tag cannot drift.
OSCAL_TAG = "v1.5.0"
OSCAL_URL = (f"https://raw.githubusercontent.com/usnistgov/oscal-content/"
             f"{OSCAL_TAG}/nist.gov/SP800-53/rev5/json/"
             f"NIST_SP-800-53_rev5_catalog.json")
OSCAL_SHA256 = "01f37cf90ea99d92242c936cbfbdebcc338eef1f71454e2acac36cc56e9bc062"
OSCAL_RELEASE = "5.2.0"


def _resolve_params(text, params_by_id):
    """Substitute organization-defined parameters with their labels."""
    def sub(m):
        p = params_by_id.get(m.group(1).strip())
        if p is None:
            return "[organization-defined value]"
        if "label" in p:
            return f"[Assignment: {p['label']}]"
        sel = p.get("select")
        if sel:
            return f"[Selection: {'; '.join(sel.get('choice', []))}]"
        return "[organization-defined value]"

    pat = r"\{\{\s*insert:\s*param,\s*([^}\s]+)\s*\}\}"
    for _ in range(4):
        if not re.search(pat, text):
            break
        text = re.sub(pat, sub, text)
    return text


def _collect_prose(part, params_by_id, depth=0, max_depth=2):
    out = []
    prose = part.get("prose")
    if prose:
        out.append(_resolve_params(prose, params_by_id))
    if depth < max_depth:
        for sub in part.get("parts", []):
            if sub.get("name") == "item":
                out.extend(_collect_prose(sub, params_by_id, depth + 1, max_depth))
    return out


def _control_text(ctrl):
    params_by_id = {p["id"]: p for p in ctrl.get("params", [])}
    statement = []
    for part in ctrl.get("parts", []):
        if part.get("name") == "statement":
            statement = _collect_prose(part, params_by_id)
            break
    text = ctrl.get("title", "") + ". " + " ".join(statement)
    return re.sub(r"\s+", " ", text).strip()


def load_oscal_controls(path):
    """Read the OSCAL catalog into the shared control corpus.

    Returns (controls, code_to_index) where controls is a list of
    (id, text, family) and code_to_index maps a normalized control code
    such as 'AC-1' to its row id. Base, non-withdrawn controls only.
    """
    with open(path, encoding="utf-8") as f:
        catalog = json.load(f)["catalog"]

    controls, codes, code_to_index = [], [], {}
    for group in catalog.get("groups", []):
        family = group.get("title", "general")
        for ctrl in group.get("controls", []):
            if any(p.get("name") == "status" and p.get("value") == "withdrawn"
                   for p in ctrl.get("props", [])):
                continue
            label = None
            for p in ctrl.get("props", []):
                if p.get("name") == "label":
                    label = p["value"].strip()
                    break
            code = norm_code(label or ctrl["id"])
            if code is None:
                continue
            text = _control_text(ctrl)
            if len(text) < 20:
                continue
            code_to_index[code] = len(controls)
            codes.append(code)
            controls.append((len(controls), text, family))

    version = catalog.get("metadata", {}).get("version", "?")
    print(f"  catalog {version}: {len(controls)} non-withdrawn base controls")
    return controls, codes, code_to_index, version


# --------------------------------------------------------------- outputs --

def write_corpus(outdir, regs, controls, mappings, control_codes=None,
                 prefix=""):
    """Write the standard CSV triple used by raa_agent.py.

    regs     : list of (id, framework, text)
    controls : list of (id, text, family)
    mappings : list of (regulation_id, control_id)
    prefix   : filename prefix, e.g. 'csf_' to keep the existing published
               artifact names stable for corpora that already shipped.
    """
    outdir = str(outdir).rstrip("/\\")
    pd.DataFrame(regs, columns=["id", "framework", "text"]).to_csv(
        f"{outdir}/{prefix}regs.csv", index=False)
    pd.DataFrame(controls, columns=["id", "text", "family"]).to_csv(
        f"{outdir}/{prefix}controls.csv", index=False)
    pd.DataFrame(sorted(set(mappings)),
                 columns=["regulation_id", "control_id"]).to_csv(
        f"{outdir}/{prefix}mappings.csv", index=False)
    if control_codes is not None:
        pd.DataFrame([(c,) for c in control_codes], columns=["code"]).to_csv(
            f"{outdir}/{prefix}control_codes.csv", index=False)

    n_multi = len({r for r, _ in mappings}
                  & {r for r, _ in mappings if
                     sum(1 for x, _ in mappings if x == r) >= 2})
    print(f"  wrote {len(regs)} requirements, {len(controls)} controls, "
          f"{len(set(mappings))} links ({n_multi} requirements multi-mapped)")


def write_provenance(outdir, sources, notes=None):
    """Record every source, its hash, and the release its mapping targets.

    The paper cites this file; it is what makes the corpus auditable without
    re-running the build.
    """
    rec = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "notes": notes or {},
    }
    path = f"{str(outdir).rstrip('/')}/provenance.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    print(f"  wrote {path}")
    return path

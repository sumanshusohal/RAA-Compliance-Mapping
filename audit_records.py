#!/usr/bin/env python3
"""Check every result record against the working tree. Run before any push.

A record is trustworthy only if two things hold:

  * it was produced from a clean tree, so `git.dirty` is false, AND the
    commit it names actually contained the hashed content. Those are two
    different claims: a clean flag says nothing about whether the named
    commit predates the code. Both are checked; pass --no-history to skip
    the second, which reads every blob out of git.
  * every file it hashed still hashes the same, so the code and inputs behind
    the numbers have not moved underneath them.

Both have failed here before. Records were committed carrying dirty:true and
pointing at a commit that did not contain the code; and analysis modules were
edited after their records were written, leaving stale hashes that a reader
checking provenance would hit immediately.

One file is deliberately exempt from the "regenerate it" advice:
confirmatory_stats.py is hashed in the OSF registration
(doi:10.17605/OSF.IO/NZXRV) and must stay byte-identical to the registered
version. If it ever shows as changed, restore it rather than re-recording.

Exit status is non-zero when anything fails, so this can gate a push.

Usage:
    python audit_records.py
    python audit_records.py --quiet     # only failures
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTERED = {
    "confirmatory_stats.py":
        "79ec294a77c12c79baf0df6187315affa521d8585f520b273050b67f27e11040",
}


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def show_at(commit, name):
    """Bytes of `name` as of `commit`, or None if absent there."""
    r = subprocess.run(["git", "show", f"{commit}:{name}"], cwd=HERE,
                       capture_output=True)
    return r.stdout if r.returncode == 0 else None


def audit(record_path, historical=True):
    """Return a list of problem strings; empty means the record is sound."""
    with open(record_path, encoding="utf-8") as f:
        record = json.load(f)
    if not isinstance(record, dict) or "git" not in record:
        return None
    problems = []
    commit = record["git"].get("commit")
    if record["git"].get("dirty") is not False:
        problems.append("produced from a dirty tree")
    if not commit:
        problems.append("no commit recorded")

    for field in ("spec_hashes", "input_hashes"):
        for name, want in (record.get(field) or {}).items():
            path = os.path.join(HERE, name)
            if not os.path.exists(path):
                problems.append(f"{name}: missing from the tree")
            elif sha256(path) != want:
                problems.append(f"{name}: hash moved since the record")

            # The working-tree check above says the file has not changed
            # since. It does NOT say the commit the record names actually
            # contained that content, which is the claim a reader relies on:
            # a record can point at a commit that predates its own code.
            # That failure has occurred here before.
            if not (historical and commit):
                continue
            blob = show_at(commit, name.replace(os.sep, "/"))
            if blob is None:
                problems.append(f"{name}: absent from commit {commit[:8]}")
            elif hashlib.sha256(
                    blob.replace(b"\r\n", b"\n")).hexdigest() != want:
                problems.append(
                    f"{name}: content at commit {commit[:8]} does not match "
                    f"the recorded hash")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true", help="only show failures")
    ap.add_argument("--no-history", action="store_true",
                    help="skip the check that each recorded commit actually "
                         "contained the hashed content (faster)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(HERE, "results_v3", "**", "*.json"),
                             recursive=True))
    rows, failing = [], 0
    for path in paths:
        try:
            problems = audit(path, historical=not args.no_history)
        except (ValueError, UnicodeDecodeError):
            continue
        if problems is None:
            continue
        rel = os.path.relpath(path, HERE)
        rows.append((rel, problems))
        failing += bool(problems)

    width = max((len(r) for r, _ in rows), default=0)
    for rel, problems in rows:
        if problems:
            print(f"FAIL {rel}")
            for p in problems:
                print(f"       {p}")
        elif not args.quiet:
            print(f"ok   {rel:<{width}}")

    print(f"\n{len(rows)} records, {failing} failing")

    # Separately: the registered artifact must not have moved at all.
    for name, want in REGISTERED.items():
        path = os.path.join(HERE, name)
        if os.path.exists(path) and sha256(path) != want:
            print(f"\nWARNING {name} differs from the OSF-registered version.")
            print("Restore it from tag hybrid-analysis-spec-v1 rather than")
            print("re-recording; the registration hashes this file.")
            failing += 1

    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())

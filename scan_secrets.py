#!/usr/bin/env python3
"""Scan the working tree and full git history for credentials.

This repository is public and ships as the replication package, so a
credential committed once is a credential published forever, even if a later
commit removes it. Run before every push:

    python scan_secrets.py            # working tree + full history
    python scan_secrets.py --tree     # working tree only (fast, pre-commit)

Exits non-zero if anything matches.
"""
import re
import subprocess
import sys

# Patterns are deliberately specific. A generic "high entropy string" rule
# fires constantly on SHA-256 hashes, which this repo is full of by design.
PATTERNS = [
    ("Anthropic API key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("Anthropic OAuth token", re.compile(r"sk-ant-oat[0-9]{2}-[A-Za-z0-9_\-]{20,}")),
    ("OpenAI API key", re.compile(r"\bsk-[A-Za-z0-9]{32,}\b")),
    ("AWS access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("Private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("Assigned API key literal",
     re.compile(r"""api_key\s*=\s*["'][^"'$\{][^"']{12,}["']""")),
    ("Exported key literal",
     re.compile(r"""ANTHROPIC_(?:API_KEY|AUTH_TOKEN)\s*=\s*["']?[A-Za-z0-9_\-]{12,}""")),
]

SKIP_SUFFIXES = (".png", ".jpg", ".jpeg", ".pdf", ".zip", ".xlsx", ".docx")


def scan_text(label, text, hits):
    for name, pat in PATTERNS:
        for m in pat.finditer(text):
            frag = m.group(0)
            redacted = frag[:10] + "..." if len(frag) > 12 else frag
            hits.append((label, name, redacted))


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          errors="replace")


def scan_tree(hits):
    files = git("ls-files").stdout.split("\n")
    n = 0
    for path in filter(None, files):
        if path.lower().endswith(SKIP_SUFFIXES):
            continue
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                scan_text(f"tree:{path}", f.read(), hits)
            n += 1
        except OSError:
            continue
    return n


def scan_history(hits):
    revs = git("rev-list", "--all").stdout.split()
    for rev in revs:
        blob = git("show", rev).stdout
        scan_text(f"commit:{rev[:9]}", blob, hits)
    return len(revs)


def main():
    tree_only = "--tree" in sys.argv
    hits = []

    n_files = scan_tree(hits)
    print(f"scanned {n_files} tracked files in the working tree")

    if not tree_only:
        n_revs = scan_history(hits)
        print(f"scanned {n_revs} commits across all refs")

    if hits:
        print(f"\nFAIL: {len(hits)} potential credential(s) found\n")
        for where, what, frag in hits:
            print(f"  {what:<26} {frag:<16} {where}")
        print("\nA credential in git history stays published even after "
              "removal. Rotate the key first, then rewrite history.")
        return 1

    print("\nPASS: no credentials found in the working tree or history.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

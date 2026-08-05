#!/usr/bin/env python3
"""Look up DOIs for the bibliography via Crossref and insert the verified ones.

Springer asks that available DOIs be given as full links. Fifty-six entries is
too many to do by hand reliably, and it is exactly the kind of task where a
wrong answer is worse than no answer: a plausible-looking DOI that points at
the wrong paper is a citation error a reader will trust.

So nothing is inserted unless three things agree with the Crossref record:

  * the first author's surname,
  * the publication year, within one year to allow for online-first, and
  * the title, at a token overlap of 0.75 or better after normalisation.

Anything short of that is reported as unmatched and left alone for a human.
--report shows what would change without touching the file.

Usage:
    python add_dois.py --report
    python add_dois.py
"""
import argparse
import io
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "manuscript3_revised.tex")
MAILTO = "sumanshu.95s@outlook.com"
TITLE_THRESHOLD = 0.75

STOP = {"a", "an", "the", "of", "for", "and", "in", "on", "to", "with", "at"}


def norm_tokens(text):
    text = re.sub(r"\\[a-zA-Z]+\s*", " ", text)
    text = re.sub(r"[^A-Za-z0-9 ]", " ", text.lower())
    return {t for t in text.split() if t and t not in STOP}


def overlap(a, b):
    ta, tb = norm_tokens(a), norm_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def crossref(query):
    url = ("https://api.crossref.org/works?rows=3"
           "&select=DOI,title,author,issued&query.bibliographic="
           + re.sub(r"\s+", "+", query.strip()))
    try:
        out = subprocess.run(
            ["curl", "-sS", "--max-time", "25", "-A",
             f"RAA-refs/1.0 (mailto:{MAILTO})", url],
            capture_output=True, text=True, check=True).stdout
        return json.loads(out)["message"]["items"]
    except Exception:
        return []


def parse_entries(tex):
    """(key, label, raw_body) for each bibitem."""
    out = []
    for m in re.finditer(r"\\bibitem(?:\[([^\]]*)\])?\{([^}]*)\}", tex):
        start = m.end()
        nxt = tex.find("\\bibitem", start)
        end = tex.find("\\end{thebibliography}", start)
        stop = min(x for x in (nxt, end) if x != -1)
        out.append((m.group(2), m.group(1) or "", tex[start:stop].strip(),
                    m.start(), stop))
    return out


def split_entry(body):
    """Return (first-author surname, year, title) from a bibitem body."""
    m = re.search(r"(\d{4})\.\s*(.+?)\.\s", body, re.S)
    if not m:
        return None, None, None
    year = int(m.group(1))
    title = re.sub(r"\s+", " ", m.group(2)).strip()
    surname = re.split(r"[,\s]", body.strip())[0]
    surname = re.sub(r"[^A-Za-z\\'-]", "", surname)
    return surname, year, title


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", action="store_true",
                    help="show matches without writing")
    args = ap.parse_args()

    tex = io.open(TEX, encoding="utf-8").read()
    entries = parse_entries(tex)
    print("%d bibitems\n" % len(entries))

    found, missed = {}, []
    for key, label, body, _, _ in entries:
        if "doi.org" in body:
            continue
        surname, year, title = split_entry(body)
        if not title:
            missed.append((key, "could not parse title"))
            continue
        items = crossref("%s %s %s" % (title, surname or "", year or ""))
        time.sleep(0.4)
        best = None
        for it in items:
            t = (it.get("title") or [""])[0]
            sim = overlap(title, t)
            auth = it.get("author") or [{}]
            fam = (auth[0].get("family") or "").lower()
            yr = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
            ok_a = bool(surname) and surname.lower()[:5] in fam
            ok_y = yr is not None and year is not None and abs(yr - year) <= 1
            if sim >= TITLE_THRESHOLD and ok_a and ok_y:
                if best is None or sim > best[1]:
                    best = (it["DOI"], sim, t)
        if best:
            found[key] = best[0]
            print("  ok    %-18s %.2f  %s" % (key, best[1], best[0]))
        else:
            missed.append((key, title[:60]))
            print("  MISS  %-18s %s" % (key, title[:60]))

    print("\n%d matched, %d unmatched" % (len(found), len(missed)))
    if args.report or not found:
        if missed:
            print("\nunmatched, leave to a human:")
            for k, why in missed:
                print("  %-18s %s" % (k, why))
        return 0

    # Insert as a full link at the end of each matched entry.
    out, last = [], 0
    for key, label, body, start, stop in parse_entries(tex):
        if key not in found:
            continue
        doi = found[key]
        insert = " \\url{https://doi.org/%s}" % doi
        out.append(tex[last:stop].rstrip())
        out.append(insert + "\n")
        last = stop
    out.append(tex[last:])
    io.open(TEX, "w", encoding="utf-8", newline="\n").write("".join(out))
    print("\ninserted %d DOI links" % len(found))
    return 0


if __name__ == "__main__":
    sys.exit(main())

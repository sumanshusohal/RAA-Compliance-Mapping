#!/usr/bin/env python3
"""Generate the manuscript's result tables from the committed records.

Three tables carry the paper's quantitative claims and were previously typed
by hand from console output. That is how the manuscript came to state a
reformulation effect of +0.140 while the records said +0.121, and how a count
of 9 requirements appeared in the abstract and the body disagreed.

This emits the table bodies from the JSON records and, with --check, verifies
that what is in the manuscript matches. --check exits non-zero on a mismatch,
so it can gate a push alongside audit_records.py.

    tab:multi   reformulation across four corpora, primary one-pass protocol
    tab:lsi     the three-cell protocol/fitting factorial
    tab:hybrid  the preregistered gated hybrid

Usage:
    python make_tables.py                 # print the generated bodies
    python make_tables.py --check         # compare against the manuscript
    python make_tables.py --write         # splice them into the manuscript
"""
import argparse
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "manuscript3_revised.tex")

ORDER = [("diagnostic", "Diagnostic (ours)"), ("nist", "CSF~1.1"),
         ("hipaa", "HIPAA"), ("pf", "Privacy Fw.")]


def load(*parts):
    with open(os.path.join(HERE, *parts), encoding="utf-8") as f:
        return json.load(f)


def num(x, places=3):
    """Format a signed effect the way the manuscript does: +.121, -.029, +-.000."""
    s = "%.*f" % (places, abs(x))
    s = s.lstrip("0")
    if abs(x) < 5e-4:
        return r"$\pm" + s + "$"
    return "$" + ("+" if x > 0 else "-") + s + "$"


def plain(x, places=3):
    return ("%.*f" % (places, x)).lstrip("0")


def table_multi():
    r = load("results_v3", "shared", "reform_vs_multi_onepass.json")
    res = r["results_by_lsi_fit"]["inductive"]
    rows = []
    for key, label in ORDER:
        c = res[key]
        e = c["exact_sign_test"]
        rows.append(
            "%s & %d & %s & %s & %s & $[%s, %s]$ & %d/%d/%d \\\\" % (
                label, c["n"], plain(c["multi_top1"]), plain(c["reform_top1"]),
                num(c["mean_difference"]),
                ("+" if c["ci_low"] > 0 else "-") + plain(abs(c["ci_low"])),
                ("+" if c["ci_high"] > 0 else "-") + plain(abs(c["ci_high"])),
                e["wins"], e["losses"], e["ties"]))
    return rows[:1] + ["\\midrule"] + rows[1:]


def table_lsi():
    a = load("results_v3", "shared", "holdout_controls_only.json")["results"]
    b = load("results_v3", "shared", "holdout_train_cal.json")["results"]
    c = load("results_v3", "shared",
             "reform_vs_multi_onepass.json")["results_by_lsi_fit"]["inductive"]
    rows = []
    for key, label in ORDER:
        va = a[key]["mean_difference"]
        vb = b[key]["mean_difference"]
        vc = c[key]["mean_difference"]
        rows.append("%-11s & %s & %s & %s & %s & %s \\\\" % (
            label.replace(" (ours)", ""), num(va), num(vb), num(vc),
            num(vb - va), "$" + plain(vc - va) + "$"))
    return rows


def table_hybrid():
    r = load("results_v3", "hybrid", "hybrid_gated.json")["results"]
    per = r["per_corpus"]
    prim = r["primary"]

    def row(key, label):
        c = per[key]
        return "%s & %d & %s & %s & %s & %s \\\\" % (
            label, c["n"], plain(c["top1"]["semantic"]),
            plain(c["top1"]["hybrid_gated"]),
            num(c["mean_difference_vs_semantic"]),
            plain(c["gate_fired_rate"], 2))

    rows = [row(k, lab) for k, lab in ORDER[1:]]
    rows.append("\\midrule")
    rows.append("Pooled & %d & --- & --- & %s & --- \\\\" % (
        prim["n"], num(prim["mean_difference"])))
    rows.append("\\midrule")
    rows.append("\\multicolumn{6}{@{}l}{\\emph{Engineered corpus, excluded "
                "from the pooled estimate}} \\\\")
    rows.append(row("diagnostic", "Diagnostic"))
    return rows


def body_in_manuscript(tex, label):
    """Return the data rows of the tabular carrying \\label{label}.

    Everything up to and including the first \\midrule is header: the column
    titles, and on tab:lsi a spanning \\multicolumn with its \\cmidrule.
    Selecting rows by "contains an ampersand" swept those in as data.
    """
    i = tex.index("\\label{%s}" % label)
    j = tex.index("\\begin{tabular}", i)
    k = tex.index("\\end{tabular}", j)
    _, _, body = tex[j:k].partition("\\midrule")
    rows = []
    for line in body.split("\n"):
        line = line.strip()
        if not line or line.startswith(("\\midrule", "\\bottomrule",
                                        "\\cmidrule")):
            continue
        if "\\multicolumn" in line:
            continue
        if "&" in line:
            rows.append(re.sub(r"\s+", " ", line))
    return rows


def norm(rows):
    """Data rows only, whitespace-normalised, for comparison.

    Drops the rules and spanning labels, which are layout rather than data
    and which body_in_manuscript() already strips from the other side.
    """
    out = []
    for r in rows:
        r = re.sub(r"\s+", " ", r).strip()
        if not r or r.startswith(("\\midrule", "\\bottomrule", "\\cmidrule")):
            continue
        if "\\multicolumn" in r:
            continue
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    generated = {"tab:multi": table_multi(), "tab:lsi": table_lsi()}
    hybrid = table_hybrid()
    if hybrid:
        generated["tab:hybrid"] = hybrid

    if not args.check:
        for label, rows in generated.items():
            print("%% %s" % label)
            for r in rows:
                print(r)
            print()
        return 0

    tex = io.open(TEX, encoding="utf-8").read()
    failing = 0
    for label, rows in generated.items():
        want, got = norm(rows), norm(body_in_manuscript(tex, label))
        if want == got:
            print("ok   %s (%d rows)" % (label, len(want)))
            continue
        failing += 1
        print("FAIL %s" % label)
        for i in range(max(len(want), len(got))):
            w = want[i] if i < len(want) else "<missing>"
            g = got[i] if i < len(got) else "<missing>"
            if w != g:
                print("       record    : %s" % w)
                print("       manuscript: %s" % g)
    print("\n%d tables checked, %d failing" % (len(generated), failing))
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())

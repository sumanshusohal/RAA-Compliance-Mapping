# Continuous integration

`github-workflow-checks.yml` is a GitHub Actions workflow. It lives here
rather than at `.github/workflows/checks.yml` only because the token used to
push this repository lacks the `workflow` scope, so it cannot create files
under `.github/workflows/`. To activate it:

```bash
mkdir -p .github/workflows
cp ci/github-workflow-checks.yml .github/workflows/checks.yml
git add .github/workflows/checks.yml && git commit -m "Enable integrity CI"
git push          # needs a token with the workflow scope
```

## What it checks

The gates that have caught real errors in this project:

| step | catches |
|---|---|
| `audit_records.py` | records from dirty trees, records naming a commit that did not contain their code, hashes that moved after the record was written |
| `make_tables.py --check` | manuscript tables drifting from the records they came from |
| `make_emse.py --check` | the generated Springer version going stale against the source |
| `test_hybrid_spec.py` | the frozen preregistered specification |
| `scan_secrets.py` | credentials in the tree or history |
| corpus shapes | a builder silently changing a corpus, or an unresolved control reference |
| lexical smoke test | the deterministic path becoming non-reproducible across runs |

`audit_records.py` needs full history, which is why the checkout step sets
`fetch-depth: 0`.

## What it does not check

Anything requiring `torch` or `sentence-transformers`: `score_all.py`,
`score_raa.py`, `freeze_backends.py` and the LLM arm. Those regenerate
committed inputs rather than verify them, and their outputs are in the
repository, so the checks above already cover whether the committed results
are internally consistent.

Nothing compiles LaTeX. `manuscript_emse.tex` has never been built; that has
to happen on Overleaf with the Springer Nature template.

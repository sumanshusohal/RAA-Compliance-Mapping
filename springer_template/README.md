# Springer SVJour3 class files

Vendored from Springer's SVJour3 distribution (LaTeX class version 3.1,
global option file svglov3.clo) so the submission package builds without
chasing a download.

EMSE's own FAQ points authors at an Overleaf SVJour3 template, which is why
the manuscript targets this class rather than the newer `sn-jnl`.

| file | role |
|---|---|
| `svjour3.cls` | the document class |
| `svglov3.clo` | global option file, loaded by the class; sets the 11.9 cm text block under `smallextended` |
| `spbasic.bst` | Springer author-year BibTeX style. Unused: the bibliography is an inline `thebibliography` |

These are Springer's files, redistributed here only to make the submission
package self-contained. They are not covered by this repository's MIT licence.

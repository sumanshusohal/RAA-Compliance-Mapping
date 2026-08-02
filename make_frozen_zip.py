#!/usr/bin/env python3
"""Build a byte-reproducible archive of the frozen analysis inputs.

The OSF registration cites this archive by SHA-256. A hash is only useful if
anyone can rebuild the archive and get the same one, and an ordinary `zip`
cannot: it records file modification times, and its entry order depends on how
the filesystem enumerates directories. Two archives of identical content would
then hash differently and the registration would look falsified.

Three things are pinned here:

  * entry order  - paths sorted, so enumeration order cannot leak in
  * timestamps   - every entry fixed at 1980-01-01, the ZIP epoch
  * permissions  - fixed mode, so a umask difference cannot change bytes

Compression level is fixed too, since zlib output depends on it.

    python make_frozen_zip.py           # build and print the hash
    python make_frozen_zip.py --check   # rebuild and verify determinism

The archive is not tracked in git. It is derived from frozen_backends/, whose
contents are tracked and individually hashed in manifest.json.
"""
import argparse
import hashlib
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "frozen_backends")
OUT = os.path.join(HERE, "frozen_backends.zip")
EPOCH = (1980, 1, 1, 0, 0, 0)


def entries():
    paths = []
    for root, dirs, files in os.walk(SRC):
        dirs.sort()
        for f in sorted(files):
            full = os.path.join(root, f)
            paths.append(os.path.relpath(full, HERE).replace("\\", "/"))
    return sorted(paths)


def build(out):
    paths = entries()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in paths:
            zi = zipfile.ZipInfo(p, date_time=EPOCH)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            with open(os.path.join(HERE, p), "rb") as f:
                z.writestr(zi, f.read())
    return paths


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="build twice and confirm the hashes agree")
    args = ap.parse_args()

    if not os.path.isdir(SRC):
        raise SystemExit("frozen_backends/ missing. Run freeze_backends.py.")

    paths = build(OUT)
    h = sha256(OUT)
    print(f"entries : {len(paths)}")
    print(f"size    : {os.path.getsize(OUT) / 1e6:.2f} MB")
    print(f"sha256  : {h}")

    if args.check:
        tmp = OUT + ".check"
        build(tmp)
        h2 = sha256(tmp)
        os.remove(tmp)
        print(f"rebuild : {'IDENTICAL' if h == h2 else 'DIFFERS'}")
        if h != h2:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

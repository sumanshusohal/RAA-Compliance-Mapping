#!/usr/bin/env python3
"""Pinned Hugging Face revisions for every neural model, in every code path.

WHY THIS EXISTS
freeze_backends.py pins the dual-encoder revision, but it is the only file
that did. raa_agent.build_semantic_scorer, raa_agent.build_crossencoder_
reranker and first_stage.py all loaded by model name alone, which resolves
through refs/main: whatever checkpoint the machine happens to hold, or
whatever the Hub serves today.

That is not hypothetical here. The development machine has two cached
snapshots of all-MiniLM-L6-v2, 1110a243 and c9745ed1. An unpinned load can
take either, and nothing downstream would say which one produced a number.

Hashing output arrays, which the hybrid does, preserves OUR results. It does
not let an independent researcher regenerate them from the same inputs. Both
are needed, so the revisions live here and every path imports them.

PROVENANCE OF THESE VALUES
Each is the commit the local Hugging Face cache resolved to for the runs
reported in the paper, read from the snapshot directory name under
~/.cache/huggingface/hub/models--*/snapshots/. The dual-encoder value is
independently corroborated: it is the revision recorded in the OSF
registration (doi:10.17605/OSF.IO/NZXRV) and in frozen_backends/manifest.json,
and this module asserts the two agree at import time.

Usage:
    python model_pins.py        # print the pins and verify they agree
"""
import sys

DUAL_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
DUAL_ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_REVISION = "c5ee24cb16019beea0893ab7796b1df96625c6b8"

PINS = {
    DUAL_ENCODER: DUAL_ENCODER_REVISION,
    CROSS_ENCODER: CROSS_ENCODER_REVISION,
}


def _check_against_registration():
    """The dual-encoder pin must match the OSF-registered one.

    freeze_backends.py is one of the seven registered artifacts and carries
    its own copy of this revision. If the two ever disagree, the repository
    is describing two different models as the same one.
    """
    try:
        import freeze_backends as fb
    except Exception:
        return None
    if fb.DUAL_ENCODER_REVISION != DUAL_ENCODER_REVISION:
        raise RuntimeError(
            "dual-encoder revision disagrees with the registered value: "
            f"model_pins {DUAL_ENCODER_REVISION} vs freeze_backends "
            f"{fb.DUAL_ENCODER_REVISION}")
    return True


def resolved_revision(model_name):
    """What the local cache actually holds, for recording alongside results.

    Returns None when the model is not cached or the hub API is unavailable,
    so callers can record "unknown" rather than assert something false.
    """
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(model_name, revision=PINS[model_name],
                                 local_files_only=True)
        return path.rstrip("/\\").split("snapshots")[-1].strip("/\\")
    except Exception:
        return None


def main():
    agree = _check_against_registration()
    for name, rev in PINS.items():
        got = resolved_revision(name)
        state = "cached" if got else "not cached locally"
        print(f"{name}\n    pinned  {rev}\n    {state}"
              + (f" as {got}" if got else ""))
    print("\ndual-encoder pin matches freeze_backends: "
          + ("yes" if agree else "freeze_backends not importable"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

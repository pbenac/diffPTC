#!/usr/bin/env python3
"""LLM-generated. Compare two diffPTC output directories.

Text/CSV files must be byte-for-byte identical (the bit-identical requirement).
PNG files are compared byte-for-byte but reported as informational only (plot
metadata and rendering can differ without any numerical change).

Usage:
    python llm_tools/compare_outputs.py REF_DIR CAND_DIR
"""
import filecmp
import os
import sys

import numpy as np

STRICT_EXT = {".csv", ".txt", ".dat", ".fits"}
INFO_EXT = {".png", ".pdf"}


def _csv_maxdiff(a, b):
    try:
        x = np.genfromtxt(a, delimiter=",", skip_header=1)
        y = np.genfromtxt(b, delimiter=",", skip_header=1)
        if x.shape != y.shape:
            return f"shape {x.shape} vs {y.shape}"
        with np.errstate(invalid="ignore"):
            return f"max |diff| = {np.nanmax(np.abs(x - y)):.3e}"
    except Exception as exc:  # noqa: BLE001
        return f"(could not parse: {exc})"


def compare_dirs(ref, cand):
    """Return (rows, ok). rows = list of (relpath, status, detail)."""
    ref_files, cand_files = {}, {}
    for root, _dirs, files in os.walk(ref):
        for f in files:
            p = os.path.join(root, f)
            ref_files[os.path.relpath(p, ref)] = p
    for root, _dirs, files in os.walk(cand):
        for f in files:
            p = os.path.join(root, f)
            cand_files[os.path.relpath(p, cand)] = p

    rows, ok = [], True
    for rel in sorted(set(ref_files) | set(cand_files)):
        ext = os.path.splitext(rel)[1].lower()
        strict = ext in STRICT_EXT or ext not in INFO_EXT
        if rel not in cand_files:
            rows.append((rel, "MISSING in candidate", ""))
            ok &= not strict
            continue
        if rel not in ref_files:
            rows.append((rel, "EXTRA in candidate", ""))
            continue
        same = filecmp.cmp(ref_files[rel], cand_files[rel], shallow=False)
        if same:
            rows.append((rel, "identical", ""))
        else:
            detail = _csv_maxdiff(ref_files[rel], cand_files[rel]) if ext == ".csv" else ""
            if strict:
                rows.append((rel, "DIFFERENT", detail))
                ok = False
            else:
                rows.append((rel, "differs (informational)", detail))
    return rows, ok


def print_report(rows, ok):
    w = max([len(r[0]) for r in rows] + [10])
    for rel, status, detail in rows:
        print(f"  {rel:<{w}}  {status}  {detail}")
    print("\nRESULT:", "PASS (all text/CSV outputs bit-identical)" if ok else "FAIL")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    r, good = compare_dirs(sys.argv[1], sys.argv[2])
    print_report(r, good)
    sys.exit(0 if good else 1)

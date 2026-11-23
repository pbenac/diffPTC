#!/usr/bin/env python3
"""LLM-generated. Synthetic test for dark_noise_measured.py.

    python llm_tools/test_dark_noise_measured.py            # run from the repo root

Writes six low-noise synthetic dark ramps (0.5 DN per read, where digitization matters)
and checks:
 1. per-read measured noise and reset noise are bit-identical to dark_noise.py
    (noq_dn and reset_e), and the "reference only" subtracted value equals
    dark_noise.py's reported read noise (sread_dn) bit-for-bit.
 2. Measured CDS noise is std(F2-F1) with digitization left in: it matches
    sqrt(2*rn^2 + 2/12) DN and is larger than the digitization-subtracted value.
 3. The script runs end to end and writes its log and summary plot.
Writes under 1 MB of temporary FITS files, deleted at the end.
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
import make_synthetic_ramps as msr  # noqa: E402
import dark_noise as dn  # noqa: E402
import dark_noise_measured as dnm  # noqa: E402

GAIN = 2.0
RN_E = 1.0                      # -> 0.5 DN per read
NUMBERS = (2001, 2002, 2003, 2004, 2005, 2006)
FAIL = []


def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what)
    if not ok:
        FAIL.append(what)


def same(a, b):
    return (np.isnan(a) and np.isnan(b)) or a == b


def main():
    tmp = tempfile.mkdtemp(prefix="diffptc_dnm_")
    try:
        data = os.path.join(tmp, "data")
        msr.write_ramps(data, numbers=NUMBERS, nf=4, flux_e_per_read=0.0, gain=GAIN, rn_e=RN_E)
        roi = (4, 60, 4, 60)
        args = ("IFS", "260101", data, list(NUMBERS), GAIN, *roi, 16)

        old = dn.read_noise_for_mode(*args)
        new = dnm.measured_noise_for_mode(*args)

        check(same(new["read_dn"], old["noq_dn"]), "per-read measured == dark_noise noq_dn (bit-identical)")
        check(same(new["reset_e"], old["reset_e"]), "reset_e == dark_noise reset_e (bit-identical)")
        check(same(new["read_noq_dn"], old["sread_dn"]), "reference subtracted == dark_noise sread_dn")

        expect = np.sqrt(2 * (RN_E / GAIN) ** 2 + 2.0 / 12.0)
        print(f"     measured CDS {new['cds_dn']:.4f} DN, expected ~{expect:.4f}; "
              f"subtracted CDS {new['cds_noq_dn']:.4f} DN")
        check(abs(new["cds_dn"] / expect - 1) < 0.04, "measured CDS matches sqrt(2 rn^2 + 2/12) within 4%")
        check(new["cds_dn"] > new["cds_noq_dn"], "measured CDS > digitization-subtracted CDS")
        check(abs(new["cds_e"] - new["cds_dn"] * GAIN) < 1e-12, "CDS e- = DN * gain")

        cfg = msr.synthetic_yaml(data, os.path.join(tmp, "out"), numbers=NUMBERS)
        cfg["dataset"]["run_subdir"] = False
        cfg["dark_roi"] = dict(zip(("x_start", "x_end", "y_start", "y_end"), roi))
        cfg["dark_analysis"] = {"channel_width": 16}
        cfg["fast"]["gain"] = GAIN
        cfg["fast"]["dark"] = {"ramps": list(NUMBERS)}
        ypath = os.path.join(tmp, "dnm.yaml")
        with open(ypath, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        p = subprocess.run([sys.executable, os.path.join(REPO, "dark_noise_measured.py"), ypath],
                           capture_output=True, text=True, cwd=tmp)
        print(p.stdout[-1500:], p.stderr[-800:])
        out = os.path.join(tmp, "out")
        check(p.returncode == 0, "dark_noise_measured.py exits 0")
        check(os.path.exists(os.path.join(out, "synthetic_darks_noise_measured.txt")), "log written")
        check(os.path.exists(os.path.join(out, "synthetic_darks_noise_measured_summary.png")), "plot written")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nALL PASS" if not FAIL else f"\n{len(FAIL)} FAILED")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

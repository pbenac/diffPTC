#!/usr/bin/env python3
"""LLM-generated. Synthetic test for gain_select.py.

    python llm_tools/test_gain_select.py            # run from the repo root

Builds two synthetic IMG "days" with different illumination and modes, runs gain.py on
each day's dataset YAML, then runs gain_select.py on a selection that takes one mode
from each day plus a read-limit override. Checks:
 1. Each selected PTC CSV (signal, variance) and per-pair gain CSV is byte-identical to
    gain.py's output for the same dataset, mode and read limits.
 2. The summary reports MATCH for the expected gains taken from gain.py's own results.
 3. A per-mode PNG is written for every entry, plus the combined figure.
 4. An entry with a wrong expect_gain is reported as DIFFERENT.
Writes about 5 MB of temporary FITS files, deleted at the end.
"""
import os
import shutil
import subprocess
import sys
import tempfile

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import make_synthetic_ramps as msr  # noqa: E402


def run(script, ypath, cwd):
    p = subprocess.run([sys.executable, os.path.join(REPO, script), ypath],
                       capture_output=True, text=True, cwd=cwd)
    if p.returncode != 0:
        print(p.stdout, p.stderr)
        raise SystemExit(f"{script} failed")
    return p.stdout


def dataset(tmp, date, flux, numbers, modes):
    """Write one synthetic day's ramps and a gain.py YAML with the given modes."""
    data = os.path.join(tmp, f"data_{date}")
    msr.write_ramps(data, head="IMG", date=date, numbers=numbers, seed=int(date),
                    flux_e_per_read=flux)
    cfg = msr.synthetic_yaml(data, os.path.join(tmp, f"out_{date}"),
                             numbers=numbers, head="IMG", date=date)
    cfg["dataset"]["name"] = f"day{date}"
    cfg["dataset"]["modes"] = list(modes)
    del cfg["fast"]
    for key, (name, last) in modes.items():
        cfg[key] = {"name": name, "frametime": 1.0, "detrend": False,
                    "analysis": {"last_read": last},
                    "illuminated": {"ramps": list(numbers), "pairing": "adjacent"}}
    path = os.path.join(tmp, f"day{date}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path, cfg


def read(p):
    with open(p, "rb") as f:
        return f.read()


def gain_from_csv(p):
    import numpy as np
    S, V = np.loadtxt(p, delimiter=",", skiprows=1, unpack=True)
    sys.path.insert(0, REPO)
    import gain
    return float(gain.fit_gain(S, V)["gain"])


def main():
    tmp = tempfile.mkdtemp(prefix="diffptc_select_")
    ok = True
    try:
        # Day A: brighter light, modes "fa" (reads 2:11) and "sa" (2:8).
        ya, ca = dataset(tmp, "260828", 400.0, (1001, 1002, 1003, 1004),
                         {"fa": ("FastA", 11), "sa": ("SlowA", 8)})
        # Day B: dimmer light, mode "fb" (2:11).
        yb, cb = dataset(tmp, "260824", 250.0, (2001, 2002, 2003),
                         {"fb": ("FastB", 11)})
        run("gain.py", ya, tmp)
        run("gain.py", yb, tmp)

        # Reference for an override: day A mode "fa" refit with last_read 6 by gain.py.
        ca6 = yaml.safe_load(open(ya))
        ca6["dataset"]["name"] = "day260828_r6"
        ca6["dataset"]["modes"] = ["fa"]
        ca6["fa"]["analysis"]["last_read"] = 6
        ya6 = os.path.join(tmp, "day260828_r6.yaml")
        yaml.safe_dump(ca6, open(ya6, "w"), sort_keys=False)
        run("gain.py", ya6, tmp)

        outa, outb = ca["dataset"]["outpath"], cb["dataset"]["outpath"]
        ref = {  # label -> (gain.py ptc csv, gain.py pair csv)
            "FastA": (f"{outa}/day260828_ptc_FastA.csv", f"{outa}/day260828_pair_gains_FastA.csv"),
            "FastB": (f"{outb}/day260824_ptc_FastB.csv", f"{outb}/day260824_pair_gains_FastB.csv"),
            "SlowA": (f"{outa}/day260828_ptc_SlowA.csv", f"{outa}/day260828_pair_gains_SlowA.csv"),
            "FastA_r6": (f"{outa}/day260828_r6_ptc_FastA.csv",
                         f"{outa}/day260828_r6_pair_gains_FastA.csv"),
        }
        expect = {k: round(gain_from_csv(v[0]), 3) for k, v in ref.items()}

        sel_out = os.path.join(tmp, "selected")
        sel = {"name": "sel", "outpath": sel_out, "entries": [
            {"dataset": os.path.basename(ya), "mode": "fa", "label": "FastA",
             "expect_gain": expect["FastA"]},
            {"dataset": os.path.basename(yb), "mode": "fb", "label": "FastB",
             "expect_gain": expect["FastB"]},
            {"dataset": os.path.basename(ya), "mode": "sa", "label": "SlowA",
             "expect_gain": expect["SlowA"]},
        ]}
        sp = os.path.join(tmp, "sel.yaml")
        yaml.safe_dump(sel, open(sp, "w"), sort_keys=False)
        run("gain_select.py", sp, REPO)  # dataset paths resolve via the selection's folder

        for label in ("FastA", "FastB", "SlowA"):
            same = (read(f"{sel_out}/sel_ptc_{label}.csv") == read(ref[label][0]) and
                    read(f"{sel_out}/sel_pair_gains_{label}.csv") == read(ref[label][1]))
            print(f"[{'PASS' if same else 'FAIL'}] {label}: CSVs byte-identical to gain.py")
            ok &= same
            png = os.path.exists(f"{sel_out}/sel_{label}_ptc.png")
            print(f"[{'PASS' if png else 'FAIL'}] {label}: per-mode PNG written")
            ok &= png
        combo = os.path.exists(f"{sel_out}/selgain_compare.png")
        print(f"[{'PASS' if combo else 'FAIL'}] combined figure written")
        ok &= combo
        summ = open(f"{sel_out}/sel_summary.txt").read()
        m = summ.count("MATCH") == 3 and "DIFFERENT" not in summ
        print(f"[{'PASS' if m else 'FAIL'}] summary reports MATCH for all 3 expected gains")
        ok &= m

        # Read-limit override and a wrong expectation.
        sel2 = {"name": "sel2", "outpath": sel_out, "entries": [
            {"dataset": ya, "mode": "fa", "label": "FastA_r6", "last_read": 6,
             "expect_gain": expect["FastA_r6"]},
            {"dataset": ya, "mode": "fa", "label": "FastA_wrong", "expect_gain": 1.0},
        ]}
        sp2 = os.path.join(tmp, "sel2.yaml")
        yaml.safe_dump(sel2, open(sp2, "w"), sort_keys=False)
        run("gain_select.py", sp2, tmp)
        same = read(f"{sel_out}/sel2_ptc_FastA_r6.csv") == read(ref["FastA_r6"][0])
        print(f"[{'PASS' if same else 'FAIL'}] last_read override matches gain.py with last_read 6")
        ok &= same
        summ2 = open(f"{sel_out}/sel2_summary.txt").read()
        d = "MATCH" in summ2 and "DIFFERENT" in summ2
        print(f"[{'PASS' if d else 'FAIL'}] wrong expect_gain reported as DIFFERENT")
        ok &= d
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("OVERALL", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

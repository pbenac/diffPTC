#!/usr/bin/env python3
"""LLM-generated. Synthetic test for per-dataset output folders (run_dirs.py).

    python llm_tools/test_run_dirs.py            # run from the repo root

Checks, with gain.py and dark_noise.py on synthetic ramps:
 1. The first gain.py run writes into <outpath>/<name>/, with config_gain.yaml (a byte copy
    of the YAML) and run_info_gain.txt.
 2. dark_noise.py on the same dataset joins that folder.
 3. A second gain.py run goes to <name>_1/ and leaves <name>/ untouched; a second
    dark_noise.py run joins <name>_1/; a third gain.py run goes to <name>_2/.
 4. With run_subdir: false, outputs are written flat into outpath, with no run files, and
    every CSV and text output is byte-identical to gain.py on main.
 5. Results in the run folder are byte-identical to the flat run.
Writes about 1 MB of temporary FITS files, deleted at the end.
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

NUMS = (1001, 1002, 1003, 1004)


def run(code, script, ypath, cwd):
    p = subprocess.run([sys.executable, os.path.join(code, script), ypath],
                       capture_output=True, text=True, cwd=cwd)
    if p.returncode:
        print(p.stdout[-1500:], p.stderr[-1500:])
    return p.returncode == 0


def snapshot(d):
    return {n: open(os.path.join(d, n), "rb").read() for n in sorted(os.listdir(d))}


def main():
    tmp = tempfile.mkdtemp(prefix="diffptc_rundirs_")
    ok = True

    def check(cond, text):
        nonlocal ok
        print(f"[{'PASS' if cond else 'FAIL'}] {text}")
        ok &= bool(cond)

    try:
        data = os.path.join(tmp, "data")
        msr.write_ramps(data, head="IMG", date="260118", numbers=NUMS)
        out = os.path.join(tmp, "results")
        cfg = msr.synthetic_yaml(data, out, numbers=NUMS, head="IMG", date="260118")
        cfg["dark_roi"] = dict(cfg["roi"])
        cfg["dark_analysis"] = {"channel_width": 32}
        cfg["fast"]["gain"] = 2.0
        cfg["fast"]["dark"] = {"ramps": list(NUMS)}
        ypath = os.path.join(tmp, "ds.yaml")
        with open(ypath, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        d0, d1, d2 = (os.path.join(out, n) for n in ("synthetic", "synthetic_1", "synthetic_2"))

        # 1.
        check(run(REPO, "gain.py", ypath, tmp), "gain.py run 1")
        f0 = set(os.listdir(d0)) if os.path.isdir(d0) else set()
        check({"syntheticresults.txt", "synthetic_ptc_Fast.csv", "synthetic_pair_gains_Fast.csv",
               "syntheticgain_compare.png", "config_gain.yaml", "run_info_gain.txt"} == f0,
              f"run 1 in results/synthetic/: {sorted(f0)}")
        check(open(os.path.join(d0, "config_gain.yaml"), "rb").read() == open(ypath, "rb").read(),
              "config_gain.yaml is a byte copy of the YAML")
        info = open(os.path.join(d0, "run_info_gain.txt")).read()
        check(all(k in info for k in ("command:", "git commit:", "git branch:", "started:")),
              "run_info_gain.txt has command, time and git details")
        check(not any(n.endswith((".csv", ".txt", ".png")) for n in os.listdir(out)
                      if os.path.isfile(os.path.join(out, n))), "nothing written flat into results/")

        # 2.
        check(run(REPO, "dark_noise.py", ypath, tmp), "dark_noise.py run 1")
        f0b = set(os.listdir(d0))
        check({"config_dark_noise.yaml", "run_info_dark_noise.txt", "synthetic_darks_noise.txt"} <= f0b
              and not os.path.exists(d1), "dark_noise.py joins results/synthetic/")
        before = snapshot(d0)

        # 3.
        check(run(REPO, "gain.py", ypath, tmp), "gain.py run 2")
        check(os.path.exists(os.path.join(d1, "run_info_gain.txt")), "gain.py run 2 goes to synthetic_1/")
        check(snapshot(d0) == before, "results/synthetic/ untouched by run 2")
        check(run(REPO, "dark_noise.py", ypath, tmp), "dark_noise.py run 2")
        check(os.path.exists(os.path.join(d1, "run_info_dark_noise.txt")) and not os.path.exists(d2),
              "dark_noise.py run 2 joins synthetic_1/")
        check(run(REPO, "gain.py", ypath, tmp), "gain.py run 3")
        check(os.path.exists(os.path.join(d2, "run_info_gain.txt")), "gain.py run 3 goes to synthetic_2/")

        # 4. run_subdir: false, compared with main.
        ref = os.path.join(tmp, "ref")
        os.makedirs(ref)
        names = subprocess.run(["git", "ls-tree", "--name-only", "main"], cwd=REPO,
                               capture_output=True, text=True, check=True).stdout.split()
        for n in names:
            if n.endswith(".py"):
                src = subprocess.run(["git", "show", f"main:{n}"], cwd=REPO,
                                     capture_output=True, check=True).stdout
                open(os.path.join(ref, n), "wb").write(src)
        flat, mainout = os.path.join(tmp, "flat"), os.path.join(tmp, "main_out")
        for code, o, name in ((REPO, flat, "flat.yaml"), (ref, mainout, "main.yaml")):
            c = yaml.safe_load(open(ypath))
            c["dataset"]["outpath"] = o
            c["dataset"]["run_subdir"] = False
            yp = os.path.join(tmp, name)
            yaml.safe_dump(c, open(yp, "w"), sort_keys=False)
            check(run(code, "gain.py", yp, tmp), f"gain.py with run_subdir: false ({'branch' if code == REPO else 'main'})")
        sf, sm = snapshot(flat), snapshot(mainout)
        check(not any(n.startswith(("config_", "run_info_")) for n in sf) and
              not any(os.path.isdir(os.path.join(flat, n)) for n in sf),
              "run_subdir: false writes flat, with no run files")
        text = [n for n in sm if n.endswith((".csv", ".txt"))]
        check(text and all(sf.get(n) == sm[n] for n in text),
              f"run_subdir: false: {len(text)} CSV/text outputs byte-identical to main")

        # 5.
        s0 = snapshot(d0)
        check(all(s0[n] == sm[n] for n in text), "run-folder results byte-identical to the flat run")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("OVERALL", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

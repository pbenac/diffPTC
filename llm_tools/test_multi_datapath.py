#!/usr/bin/env python3
"""LLM-generated. Synthetic test for 'datapath' given as a list of folders (data_paths.py).

    python llm_tools/test_multi_datapath.py            # run from the repo root

Writes six synthetic IMG ramps and runs gain.py, dark_noise.py and dark_current.py:
 1. With a single datapath string, every text and CSV output is byte-identical to the
    same scripts as they are on main (git show main:<file>), so nothing changes for
    existing YAMLs.
 2. With the ramps split over two folders like a night past midnight (nov18/si260118_*,
    nov19/si260119_*, and one nov19/si260118_* file), and datapath given as a list, every
    CSV is byte-identical to the single-folder run, and the text outputs differ only in
    the line naming the data directory.
 3. Two files with the same ramp number in the listed folders stop gain.py with an error
    naming both.
 4. A ramp missing from every listed folder is reported as "files not ready", as before.
Writes about 2 MB of temporary FITS files, deleted at the end.
"""
import glob
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

SCRIPTS = ("gain.py", "dark_noise.py", "dark_current.py")
NUMS = (1001, 1002, 1003, 1004, 1005, 1006)


def run(code_dir, script, ypath, cwd):
    return subprocess.run([sys.executable, os.path.join(code_dir, script), ypath],
                          capture_output=True, text=True, cwd=cwd)


def config(datapath, outpath, ramps=NUMS):
    c = msr.synthetic_yaml(datapath, outpath, numbers=ramps, head="IMG", date="260118")
    c["dark_roi"] = dict(c["roi"])
    c["dataset"]["run_subdir"] = False  # flat, to compare with main
    c["dark_analysis"] = {"channel_width": 32}
    c["fast"]["gain"] = 2.0
    c["fast"]["illuminated"]["pairing"] = "adjacent"
    c["fast"]["dark"] = {"ramps": list(ramps)}
    return c


def write_yaml(path, cfg):
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def outputs(d):
    return {os.path.basename(p): open(p, "rb").read()
            for p in sorted(glob.glob(os.path.join(d, "*.txt")) + glob.glob(os.path.join(d, "*.csv")))}


def main():
    tmp = tempfile.mkdtemp(prefix="diffptc_multipath_")
    ok = True

    def check(cond, text):
        nonlocal ok
        print(f"[{'PASS' if cond else 'FAIL'}] {text}")
        ok &= bool(cond)

    try:
        # Reference code: every top-level .py file as it is on main.
        ref = os.path.join(tmp, "ref")
        os.makedirs(ref)
        names = subprocess.run(["git", "ls-tree", "--name-only", "main"], cwd=REPO,
                               capture_output=True, text=True, check=True).stdout.split()
        for n in names:
            if n.endswith(".py"):
                src = subprocess.run(["git", "show", f"main:{n}"], cwd=REPO,
                                     capture_output=True, check=True).stdout
                open(os.path.join(ref, n), "wb").write(src)

        one = os.path.join(tmp, "one")
        msr.write_ramps(one, head="IMG", date="260118", numbers=NUMS)

        # 1. Single string: candidate == main.
        y_ref = write_yaml(os.path.join(tmp, "ref.yaml"), config(one, os.path.join(tmp, "out_ref")))
        y_one = write_yaml(os.path.join(tmp, "one.yaml"), config(one, os.path.join(tmp, "out_one")))
        for s in SCRIPTS:
            a, b = run(ref, s, y_ref, tmp), run(REPO, s, y_one, tmp)
            if a.returncode or b.returncode:
                print(a.stderr[-2000:], b.stderr[-2000:])
            check(a.returncode == 0 and b.returncode == 0, f"{s} runs (main and branch)")
        o_ref, o_one = outputs(os.path.join(tmp, "out_ref")), outputs(os.path.join(tmp, "out_one"))
        check(o_ref.keys() == o_one.keys() and len(o_ref) >= 5, f"same {len(o_ref)} output files")
        check(o_ref == o_one, "single datapath: every txt/csv byte-identical to main")

        # 2. Split over two folders, like a night that runs past midnight.
        n18, n19 = os.path.join(tmp, "nov18"), os.path.join(tmp, "nov19")
        os.makedirs(n18)
        os.makedirs(n19)
        for n in NUMS:
            src = os.path.join(one, f"si260118_{n:05d}.fits")
            if n <= 1003:
                dst = os.path.join(n18, os.path.basename(src))
            elif n == 1004:   # started before midnight, written after
                dst = os.path.join(n19, os.path.basename(src))
            else:
                dst = os.path.join(n19, f"si260119_{n:05d}.fits")
            shutil.copy(src, dst)
        open(os.path.join(n19, "si260119_01005_qramp.fits"), "w").write("not a ramp")
        y_two = write_yaml(os.path.join(tmp, "two.yaml"),
                           config([n18, n19], os.path.join(tmp, "out_two")))
        for s in SCRIPTS:
            r = run(REPO, s, y_two, tmp)
            if r.returncode:
                print(r.stderr[-2000:])
            check(r.returncode == 0, f"{s} runs with datapath as a list")
        o_two = outputs(os.path.join(tmp, "out_two"))
        csvs = [k for k in o_one if k.endswith(".csv")]
        check(csvs and all(o_two.get(k) == o_one[k] for k in csvs),
              f"split folders: all {len(csvs)} CSVs byte-identical to the single-folder run")
        same_txt = True
        for k in (k for k in o_one if k.endswith(".txt")):
            la = o_one[k].decode().splitlines()
            lb = o_two.get(k, b"").decode().splitlines()
            diff = [(x, y) for x, y in zip(la, lb) if x != y]
            if len(la) != len(lb) or any("data directory" not in x for x, _ in diff):
                same_txt = False
                print("   differs:", k, diff[:3])
        check(same_txt, "split folders: text outputs differ only in the data-directory line")

        # 3. Duplicate ramp number across folders -> error naming both.
        shutil.copy(os.path.join(n18, "si260118_01002.fits"), os.path.join(n19, "si260119_01002.fits"))
        r = run(REPO, "gain.py", y_two, tmp)
        msg = r.stdout + r.stderr
        check("more than one file matches" in msg and "nov18/si260118_01002.fits" in msg
              and "nov19/si260119_01002.fits" in msg, "duplicate ramp number is an error naming both files")
        os.remove(os.path.join(n19, "si260119_01002.fits"))

        # 4. Missing ramp -> 'files not ready', as with a single folder.
        os.remove(os.path.join(n19, "si260119_01006.fits"))
        r = run(REPO, "gain.py", y_two, tmp)
        check(r.returncode == 0 and "1005/1006: files not ready; skipped" in r.stdout,
              "missing ramp reported as 'files not ready'")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("OVERALL", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

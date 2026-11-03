#!/usr/bin/env python3
"""LLM-generated. Run the same dataset YAML through two checkouts and compare outputs.

    python llm_tools/run_regression.py --ref ../diffPTC_ref --cand ../diffPTC_cand \
        --yaml datasets/CD5/260622_IFS.yaml --scripts gain.py [dark_noise.py ...]

To compare a differently named candidate script, pair each one with a reference script:

    python llm_tools/run_regression.py --ref ../diffPTC_ref --cand . \
        --yaml datasets/260901_IMG_Illuminated_Defaults.yaml \
        --scripts gain.py --cand-scripts gain_new.py

Each checkout runs `python <script> <yaml-copy>` with its own directory as cwd, so it imports
its own modules. The YAML copy has dataset.outpath redirected to a scratch folder, so your real
results folder is never touched. Data paths are used read-only exactly as in the YAML.
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_outputs import compare_dirs, print_report  # noqa: E402

STUBS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stubs")


def run_one(checkout, script, yaml_path, outdir, stub):
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    cfg["dataset"]["outpath"] = outdir + os.sep
    os.makedirs(outdir, exist_ok=True)
    ycopy = os.path.join(outdir, "_config_copy.yaml")
    with open(ycopy, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    env = dict(os.environ)
    if stub:
        env["PYTHONPATH"] = STUBS + os.pathsep + env.get("PYTHONPATH", "")
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, script, ycopy], cwd=checkout, env=env,
                       capture_output=True, text=True)
    wall = time.perf_counter() - t0
    os.remove(ycopy)
    return p.returncode, wall, p.stdout, p.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--scripts", nargs="+", default=["gain.py"])
    ap.add_argument("--cand-scripts", nargs="+", default=None,
                    help="candidate script names, paired in order with --scripts (default: same names)")
    ap.add_argument("--workdir", default=None, help="scratch folder (default: temp dir)")
    ap.add_argument("--stub-gui-io", action="store_true")
    args = ap.parse_args()

    cand_scripts = args.cand_scripts or args.scripts
    if len(cand_scripts) != len(args.scripts):
        ap.error("--cand-scripts must list one script for each of --scripts")

    work = args.workdir or tempfile.mkdtemp(prefix="diffptc_regress_")
    yaml_path = os.path.abspath(args.yaml)
    all_ok = True
    for script, cand_script in zip(args.scripts, cand_scripts):
        print(f"\n=== {script} ===" if script == cand_script else f"\n=== {script} vs {cand_script} ===")
        outs = {}
        for label, checkout, name in (("ref", args.ref, script), ("cand", args.cand, cand_script)):
            out = os.path.join(work, script.replace(".py", ""), label)
            rc, wall, so, se = run_one(os.path.abspath(checkout), name, yaml_path, out, args.stub_gui_io)
            outs[label] = out
            print(f"  {label:<4} exit={rc}  wall={wall:7.2f} s")
            if rc != 0:
                all_ok = False
                print(se[-1500:])
        rows, ok = compare_dirs(outs["ref"], outs["cand"])
        print_report(rows, ok)
        all_ok &= ok
    print("\nOVERALL:", "PASS" if all_ok else "FAIL", f"(scratch outputs in {work})")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

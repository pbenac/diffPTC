#!/usr/bin/env python3
"""LLM-generated. Profile one diffPTC analysis script on a dataset YAML.

Usage (from the repo root):
    python llm_tools/profile_run.py gain.py datasets/CD5/260622_IFS.yaml [--top 30] [--sort cumulative|tottime]

Writes <yaml-stem>_<script-stem>.prof next to the current directory and prints the top entries.
Plots are produced as normal (set dataset.outpath in the YAML to a scratch folder if you do not
want to overwrite results).
"""
import argparse
import cProfile
import io
import os
import pstats
import runpy
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script", help="e.g. gain.py")
    ap.add_argument("yaml", help="dataset YAML")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--sort", default="cumulative", choices=["cumulative", "tottime"])
    ap.add_argument("--stub-gui-io", action="store_true",
                    help="use llm_tools/stubs/gui_io.py if the real gui_io is not importable")
    args = ap.parse_args()

    script = os.path.abspath(args.script)
    repo = os.path.dirname(script)
    sys.path.insert(0, repo)
    if args.stub_gui_io:
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stubs"))
    sys.argv = [script, args.yaml]

    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit:
        pass
    finally:
        prof.disable()
    wall = time.perf_counter() - t0

    stem = f"{os.path.splitext(os.path.basename(args.yaml))[0]}_{os.path.splitext(os.path.basename(script))[0]}"
    prof.dump_stats(stem + ".prof")

    s = io.StringIO()
    pstats.Stats(prof, stream=s).sort_stats(args.sort).print_stats(args.top)
    print(f"\nwall time: {wall:.1f} s   (profile saved to {stem}.prof)\n")
    print(s.getvalue())


if __name__ == "__main__":
    main()

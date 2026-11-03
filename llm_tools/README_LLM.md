# llm_tools: LLM-generated regression and profiling helpers

Everything in this folder was written by Claude (an LLM), not by the repo owner.
Lines Claude wrote in the repo owner's scripts are marked inline with `# LLM-generated`.

Requirement while this folder exists: LLM-modified analysis code must reproduce the
original outputs **bit-for-bit** (byte-identical text/CSV outputs).

## Tools

| File | Purpose |
|---|---|
| `profile_run.py` | Run one of the analysis scripts under cProfile on a dataset YAML and print the hot spots. Run this on the remote server on a real dataset. |
| `compare_outputs.py` | Compare two output directories. Text/CSV must match byte-for-byte; PNG differences are reported but informational. |
| `run_regression.py` | Run the same YAML through a reference checkout and a candidate checkout, then compare all outputs. Use on the remote server. |
| `make_synthetic_ramps.py` | Write fake FITS ramps with a known gain/read noise (local testing without real data). |
| `test_synthetic.py` | End-to-end local self-test of the harness using synthetic ramps. No real data needed. |
| `stubs/gui_io.py` | Stand-in for the `gui_io` module, which is imported by the scripts and tracked at the repo root as `gui_io.py`. Used only if the real one is not importable (`--stub-gui-io`). |

## Typical remote workflow

```bash
# 0. See the current state before changing anything
cd ~/diffPTC
git worktree list && git status --short --branch

# 1. Pin the reference to the commit you are comparing against. Use --detach, never a branch
#    name: a branch can only be open in one folder at a time.
git fetch origin
git worktree add --detach ../diffPTC_ref origin/main
# If ../diffPTC_ref already exists, move it instead:
# git -C ../diffPTC_ref checkout --detach origin/main

# 2. Put this folder on the candidate branch
git switch llm/<topic>

# 3. Profile on a real dataset (find what actually costs time)
python llm_tools/profile_run.py gain.py datasets/CD5/260622_IFS.yaml --top 25

# 4. Regression-check the candidate against the reference
python llm_tools/run_regression.py --ref ../diffPTC_ref --cand . \
    --yaml datasets/CD5/260622_IFS.yaml --scripts gain.py --workdir ~/regress_check
grep -c "failed" ~/regress_check/gain/ref/*results.txt
```

`run_regression.py` and `profile_run.py` rewrite `dataset.outpath` in a temporary copy of the YAML
so your real `results/` folder is never touched. `run_regression.py` prints PASS/FAIL per output
file and per-script wall times.

Reading the results:
- PASS needs the `grep` to print `0`. `gain.py` logs fit errors and exits normally, so two runs
  where every fit failed would still compare as identical.
- Run order does not affect the PASS/FAIL comparison, but it can affect timings. The reference runs
  first, so it may pay for uncached disk reads. To compare speed, repeat the run with the order
  swapped (`--ref . --cand ../diffPTC_ref`). Delete the `--workdir` folder between runs if the
  outputs may differ.

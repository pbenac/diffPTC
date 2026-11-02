# llm_tools: LLM-generated regression and profiling helpers

Everything in this folder was written by Claude (an LLM), not by the repo owner.
It adds new files only; it does not modify any existing diffPTC code.
Lives on branch `llm/regression-tools`; merge or discard at your discretion.

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
| `stubs/gui_io.py` | Stand-in for the `gui_io` module, which is imported by the scripts but is not tracked in the repo. Used only if the real one is not importable (`--stub-gui-io`). |

## Typical remote workflow

```bash
# 1. Two checkouts: reference (main) and candidate (the LLM branch)
git worktree add ../diffPTC_ref main
git worktree add ../diffPTC_cand llm/<some-speedup-branch>

# 2. Profile the reference on a real dataset (find what actually costs time)
python llm_tools/profile_run.py gain.py datasets/CD5/260622_IFS.yaml --top 25

# 3. Regression-check the candidate against the reference
python llm_tools/run_regression.py --ref ../diffPTC_ref --cand ../diffPTC_cand \
    --yaml datasets/CD5/260622_IFS.yaml --scripts gain.py
```

`run_regression.py` rewrites `dataset.outpath` in a temporary copy of the YAML so your real
`results/` folder is never touched. It prints PASS/FAIL per output file and per-script wall times.

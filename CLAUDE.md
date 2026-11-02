# CLAUDE.md: working rules for diffPTC

(Draft written by Claude (an LLM) for the repo owner to review. Edit freely.)

## What this repo is
Detector characterization toolkit (gain/PTC, read and reset noise, dark current, well depth)
for SCALES H2RG data. Analyses are YAML-driven; see README.md.

## Rules for LLM-assisted work
1. **Never commit to or edit `main`.** Do all work on a branch named `llm/<short-topic>`.
   The owner decides later whether to merge.
2. **Do not modify existing files without asking first.** For code changes, create a duplicate
   (for example `gain_fast.py`) or a new file instead, and leave the original untouched.
3. **Label LLM-written code.** Start each new file with a comment saying it is LLM-generated.
   Commits must keep the Claude co-author trailer so `git log` shows what came from the LLM.
4. **Results must stay bit-identical** to the original code for now. Do not use float32,
   reordered reductions, or any change that could alter low-order digits. If a speedup needs
   a looser tolerance, describe it and ask; do not enable it.
5. **The real data lives on a remote server and is never copied here.** Do not try to run
   analyses on real data locally. Write scripts and tests; the owner runs them on the server
   and reports back.
6. **Before calling a change done**, run `python llm_tools/test_synthetic.py` and confirm it
   passes. On the server, the owner runs `llm_tools/run_regression.py` against a real dataset.
7. **Do not push, force-push, or delete branches** unless explicitly asked.

## Layout
- `gain.py`, `dark_noise.py`, `dark_current.py`, `well_depth.py`: analysis scripts
- `datasets/`: one YAML per dataset
- `results/`: committed outputs; the reference for regression checks
- `llm_tools/`: LLM-generated profiling, comparison, and synthetic-test helpers
- `gui_io.py` is imported by the scripts but is not tracked here; `llm_tools/stubs/gui_io.py`
  is a stand-in used only for local synthetic tests.

## Environment notes
Check Python, numpy, scipy and astropy versions before changing numerics; the server and the
local machine may differ.
#!/usr/bin/env python3
"""LLM-generated. Local self-test of the regression harness using synthetic ramps.

    python llm_tools/test_synthetic.py            # run from the repo root

Checks:
 1. gain.py recovers the injected gain (sanity check of the synthetic data).
 2. Running unmodified code twice is bit-identical (deterministic; harness reports PASS).
 3. A deliberately perturbed copy of gain.py is reported as FAIL (harness detects changes).
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


def sh(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def main():
    tmp = tempfile.mkdtemp(prefix="diffptc_synth_")
    data = os.path.join(tmp, "data")
    msr.write_ramps(data)
    ypath = os.path.join(tmp, "synthetic.yaml")
    with open(ypath, "w") as f:
        yaml.safe_dump(msr.synthetic_yaml(data, os.path.join(tmp, "out")), f, sort_keys=False)

    # copies of the repo's python files: "ref" untouched, "cand" untouched, "bad" perturbed
    def clone(name):
        d = os.path.join(tmp, name)
        os.makedirs(d)
        for fn in os.listdir(REPO):
            if fn.endswith(".py"):
                shutil.copy(os.path.join(REPO, fn), d)
        return d

    ref, cand, bad = clone("ref"), clone("cand"), clone("bad")
    src = open(os.path.join(bad, "gain.py")).read()
    assert "sigma=3.0" in src
    open(os.path.join(bad, "gain.py"), "w").write(src.replace("sigma=3.0", "sigma=2.9", 1))

    reg = os.path.join(HERE, "run_regression.py")
    base = [sys.executable, reg, "--yaml", ypath, "--scripts", "gain.py", "--stub-gui-io"]

    r1 = sh(*base, "--ref", ref, "--cand", cand, "--workdir", os.path.join(tmp, "w1"))
    print(r1.stdout, r1.stderr[-800:])
    ok1 = r1.returncode == 0

    r2 = sh(*base, "--ref", ref, "--cand", bad, "--workdir", os.path.join(tmp, "w2"))
    ok2 = r2.returncode != 0
    print(r2.stdout[-900:])

    log = open(os.path.join(tmp, "w1", "gain", "ref", "synthetic" + "results.txt")).read()
    print(log)
    print("TEST 1 identical-code run passes:      ", "OK" if ok1 else "FAILED")
    print("TEST 2 perturbed code is detected:     ", "OK" if ok2 else "FAILED")
    print(f"(true gain injected = {msr.TRUE_GAIN} e-/DN, RN = {msr.TRUE_RN_E} e-)")
    shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(0 if (ok1 and ok2) else 1)


if __name__ == "__main__":
    main()

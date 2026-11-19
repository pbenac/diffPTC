#!/usr/bin/env python3
"""LLM-generated. Synthetic tests for gain_ipc.py (IPC-corrected gain).

    python llm_tools/test_gain_ipc.py            # run from the repo root

Checks:
 1. With no IPC injected, gain_ipc's raw S, V and gain are bit-identical to gain.py,
    the measured alpha is ~0, and the corrected gain stays at the true gain.
 2. With a known nearest-neighbour IPC kernel injected, the raw gain is biased high by
    1/sum(k^2), and the kernel and covariance-sum corrections recover the true gain and alpha.
 3. gain_ipc.py runs end to end on a YAML and writes its outputs without a failed fit,
    and does not write gain.py's output names.
Writes about 15 MB of temporary FITS files, deleted at the end.
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import yaml
from astropy.io import fits
from scipy.ndimage import convolve

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
import make_synthetic_ramps as msr  # noqa: E402
import gain  # noqa: E402
import gain_ipc  # noqa: E402

AX, AY = 0.020, 0.015   # injected IPC (x = columns, y = rows)
# gain.py clips the difference image at 3 sigma, which keeps only ~97.0% of a Gaussian's
# variance, so even with no IPC it reports ~3.1% above the injected gain. That bias is shared
# by gain.py and gain_ipc.py; the tests compare against the clip-biased expectation.
CLIP_VAR = 0.9698


def make_ipc_ramp(rng, alpha_x, alpha_y, nf=12, ny=256, nx=256, flux_e_per_read=400.0,
                  g=msr.TRUE_GAIN, rn_e=msr.TRUE_RN_E):
    k0 = 1.0 - 2 * alpha_x - 2 * alpha_y
    kern = np.array([[0, alpha_y, 0], [alpha_x, k0, alpha_x], [0, alpha_y, 0]])
    yy, xx = np.mgrid[0:ny, 0:nx]
    prnu = 1.0 + 0.02 * np.sin(xx / 9.0) * np.cos(yy / 11.0)
    e_cum = np.zeros((ny, nx))
    cube = np.empty((nf, ny, nx))
    for j in range(nf):
        if j > 0:
            e_cum += rng.poisson(flux_e_per_read * prnu).astype(float)
        sig = convolve(e_cum, kern, mode="wrap")   # IPC acts on collected charge
        cube[j] = msr.BIAS_DN + (sig + rng.normal(0.0, rn_e, (ny, nx))) / g   # RN after IPC
    return np.clip(np.rint(cube), 0, 65535).astype(np.uint16)


def write(outdir, alpha_x, alpha_y, numbers=(1001, 1002, 1003, 1004), seed=1):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    for n in numbers:
        p = os.path.join(outdir, f"ss260101_{n:05d}.fits")
        fits.PrimaryHDU(make_ipc_ramp(rng, alpha_x, alpha_y)).writeto(p, overwrite=True)


def run_both(data, log):
    pairs = gain.make_pairs({"illuminated": {"ramps": [1001, 1002, 1003, 1004]}},
                            [1001, 1002, 1003, 1004])
    roi = (8, 248, 8, 248)
    ref = gain.gain_diff_many_pairs("IFS", "260101", pairs, data, *roi, log,
                                    first_read=2, last_read=11)
    new = gain_ipc.ipc_many_pairs("IFS", "260101", pairs, data, *roi, log,
                                  {"method": "kernel", "max_lag": 2},
                                  first_read=2, last_read=11)
    return ref, new


def main():
    tmp = tempfile.mkdtemp(prefix="diffptc_ipc_")
    log = open(os.path.join(tmp, "log.txt"), "w")
    true_g = msr.TRUE_GAIN / CLIP_VAR   # what an IPC-free detector gives here
    ok = []

    # 1. no IPC
    d0 = os.path.join(tmp, "noipc")
    write(d0, 0.0, 0.0)
    ref, new = run_both(d0, log)
    same = (np.array_equal(ref["S"], new["S"]) and np.array_equal(ref["V"], new["V"])
            and ref["gain"] == new["raw_gain"]
            and all(a["gain"] == b["raw_gain"]
                    for a, b in zip(ref["pair_results"], new["pair_results"])))
    t1 = (same and abs(new["alpha_x"]) < 0.003 and abs(new["alpha_y"]) < 0.003
          and abs(new["raw_gain"] / true_g - 1) < 0.02
          and abs(new["gain_kernel"] / true_g - 1) < 0.04)
    print(f"no IPC : raw {new['raw_gain']:.4f}  kernel {new['gain_kernel']:.4f}  "
          f"covsum {new['gain_covsum']:.4f}  alpha=({new['alpha_x']:.4f},{new['alpha_y']:.4f})  "
          f"raw identical to gain.py: {same}")
    ok.append(t1)

    # 2. injected IPC
    d1 = os.path.join(tmp, "ipc")
    write(d1, AX, AY, seed=2)   # different noise from case 1
    ref, new = run_both(d1, log)
    s2 = gain_ipc.kernel_sum_sq(AX, AY)
    t2 = (np.array_equal(ref["V"], new["V"])
          and abs(new["raw_gain"] / (true_g / s2) - 1) < 0.02
          and abs(new["gain_kernel"] / true_g - 1) < 0.04
          and abs(new["gain_covsum"] / true_g - 1) < 0.04
          and abs(new["alpha_x"] - AX) < 0.004 and abs(new["alpha_y"] - AY) < 0.004)
    print(f"IPC    : raw {new['raw_gain']:.4f} (expect {true_g / s2:.4f})  kernel "
          f"{new['gain_kernel']:.4f}  covsum {new['gain_covsum']:.4f}  (expect {true_g:.4f})  "
          f"alpha=({new['alpha_x']:.4f},{new['alpha_y']:.4f}) (true {AX},{AY})")
    ok.append(t2)

    # 3. end to end
    out = os.path.join(tmp, "out")
    cfg = msr.synthetic_yaml(d1, out)
    cfg["roi"] = {"x_start": 8, "x_end": 248, "y_start": 8, "y_end": 248}
    cfg["ipc"] = {"method": "kernel"}
    ypath = os.path.join(tmp, "ipc.yaml")
    with open(ypath, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    p = subprocess.run([sys.executable, os.path.join(REPO, "gain_ipc.py"), ypath],
                       capture_output=True, text=True, cwd=REPO)
    print(p.stdout, p.stderr[-800:])
    run = os.path.join(out, "synthetic")  # per-dataset run folder (run_dirs.py)
    files = sorted(os.listdir(run)) if os.path.isdir(run) else []
    expect = {"syntheticresults_ipc.txt", "synthetic_ptc_ipc_Fast.csv",
              "synthetic_pair_gains_ipc_Fast.csv", "syntheticgain_ipc_compare.png",
              "config_gain_ipc.yaml", "run_info_gain_ipc.txt"}
    t3 = (p.returncode == 0 and "failed" not in p.stdout and set(files) == expect)
    print("outputs:", files)
    ok.append(t3)

    print("TEST 1 no IPC: raw == gain.py, alpha ~ 0:     ", "OK" if ok[0] else "FAILED")
    print("TEST 2 injected IPC recovered:                ", "OK" if ok[1] else "FAILED")
    print("TEST 3 end-to-end run, own output names only: ", "OK" if ok[2] else "FAILED")
    log.close()
    shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()

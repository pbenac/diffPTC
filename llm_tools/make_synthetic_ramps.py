#!/usr/bin/env python3
"""LLM-generated. Write synthetic detector ramps with a known gain and read noise.

Files are named like the real data (ss<date>_<NNNNN>.fits for IFS, si... for IMG) so the
diffPTC scripts can read them unchanged. Fixed seed => deterministic.
"""
import os

import numpy as np
from astropy.io import fits

TRUE_GAIN = 2.0       # e-/DN
TRUE_RN_E = 10.0      # e- read noise per read
BIAS_DN = 1000.0


def make_ramp(rng, nf=12, ny=64, nx=64, flux_e_per_read=400.0, gain=TRUE_GAIN, rn_e=TRUE_RN_E):
    yy, xx = np.mgrid[0:ny, 0:nx]
    prnu = 1.0 + 0.02 * np.sin(xx / 9.0) * np.cos(yy / 11.0)   # smooth flat-field structure
    e_cum = np.zeros((ny, nx))
    cube = np.empty((nf, ny, nx))
    for j in range(nf):
        if j > 0:
            e_cum += rng.poisson(flux_e_per_read * prnu).astype(float)
        cube[j] = BIAS_DN + (e_cum + rng.normal(0.0, rn_e, (ny, nx))) / gain
    return np.clip(np.rint(cube), 0, 65535).astype(np.uint16)


def write_ramps(outdir, head="IFS", date="260101", numbers=(1001, 1002, 1003, 1004), seed=12345, **kw):
    os.makedirs(outdir, exist_ok=True)
    prefix = "ss" if head == "IFS" else "si"
    rng = np.random.default_rng(seed)
    paths = []
    for n in numbers:
        p = os.path.join(outdir, f"{prefix}{date}_{int(n):05d}.fits")
        fits.PrimaryHDU(make_ramp(rng, **kw)).writeto(p, overwrite=True)  # uint16 -> BZERO=32768
        paths.append(p)
    return paths


def synthetic_yaml(datapath, outpath, numbers=(1001, 1002, 1003, 1004), head="IFS", date="260101"):
    return {
        "dataset": {"name": "synthetic", "descr": "LLM synthetic ramps", "datapath": datapath,
                    "outpath": outpath, "head": head, "date": date, "modes": ["fast"],
                    "quicklook": False},
        "roi": {"x_start": 4, "x_end": 60, "y_start": 4, "y_end": 60},
        "analysis": {"first_read": 2, "last_read": 11},
        "fast": {"name": "Fast", "frametime": 1.0, "detrend": False,
                 "illuminated": {"ramps": list(numbers)}},
    }

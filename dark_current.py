#!/usr/bin/env python3
"""
Dark current per readout mode from dark ramps.

Refactored to match gain.py style:
  python dark_current.py config.yaml

Expected YAML additions, per mode:

fast06:
  gain: 43.944
  dark:
    ramps: [4433, 4434, 4435]

or:

fast06:
  gain: 43.944
  dark:
    ramp_range: [4433, 4450]   # inclusive start, exclusive stop, like range()

Optional:

dark_current:
  primary_mode: slow52
  histogram_exclude_first: true
"""
import os
import sys
import warnings

import numpy as np
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import yaml


def P(log, *a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.write(s + "\n")
    log.flush()


def obs_filename(head, date, n):
    pref = "ss" if head == "IFS" else "si"
    return f"{pref}{date}_{n:05d}.fits"


def reg_cube(head, date, n, datapath, X0, X1, Y0, Y1):
    fn = os.path.join(datapath, obs_filename(head, date, n))
    with fits.open(fn, memmap=True, do_not_scale_image_data=True) as h:
        bz = h[0].header.get("BZERO", 0)
        return np.asarray(h[0].data[:, Y0:Y1, X0:X1]).astype(np.float64) + bz


def pix_slope(cube):
    """Per-pixel OLS slope over reads 1..N, dropping reset frame index 0."""
    y = cube[1:]
    n = y.shape[0]
    x = np.arange(n)
    xc = (x - x.mean())[:, None, None]
    return (xc * (y - y.mean(0))).sum(0) / (xc * xc).sum()


def parse_ramps(mode_cfg):
    dark_cfg = mode_cfg.get("dark", mode_cfg.get("darks", {}))
    if "ramps" in dark_cfg:
        return list(dark_cfg["ramps"])
    if "ramp_range" in dark_cfg:
        start, stop = dark_cfg["ramp_range"]
        return list(range(int(start), int(stop)))
    if "range" in dark_cfg:
        start, stop = dark_cfg["range"]
        return list(range(int(start), int(stop)))
    if "start" in dark_cfg and "stop" in dark_cfg:
        return list(range(int(dark_cfg["start"]), int(dark_cfg["stop"])))
    return []


def get_gain(mode_cfg):
    for key in ("gain", "conversion_gain", "gain_e_per_dn"):
        if key in mode_cfg:
            return float(mode_cfg[key])
    return None


def analyze_mode(head, date, datapath, ramps, gain, itime, X0, X1, Y0, Y1):
    vals = []
    profiles = []
    nf_last = None

    for n in ramps:
        c = reg_cube(head, date, n, datapath, X0, X1, Y0, Y1)
        nf = c.shape[0]
        nf_last = nf
        dmap = pix_slope(c) * gain / itime
        vals.append(sigma_clipped_stats(dmap, sigma=3, maxiters=5)[1])
        med = np.median(c.reshape(nf, -1), axis=1)
        profiles.append(med - med[1])

    vals = np.array(vals)
    dark = np.median(vals)
    err = np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else float("nan")

    L = min(len(p) for p in profiles)
    profile_arr = np.array([p[:L] for p in profiles])
    profile = (np.arange(L) * itime, profile_arr.mean(0))

    return dict(dark=dark, err=err, g=gain, it=itime, nramp=len(ramps), ttot=(nf_last - 1) * itime, profile=profile)


def make_plots(summary, config, head, date, datapath, outpath, setname, X0, X1, Y0, Y1):
    if not summary:
        return

    dark_current_cfg = config.get("dark_current", {})
    primary_mode_key = dark_current_cfg.get("primary_mode")

    if primary_mode_key and primary_mode_key in config:
        primary_name = config[primary_mode_key]["name"]
    else:
        primary_name = next(iter(summary))
        primary_mode_key = next((m for m in config["dataset"]["modes"] if config[m]["name"] == primary_name), None)

    names = list(summary)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    for name in names:
        t, p = summary[name]["profile"]
        ax[0].plot(t, p, "o-", ms=3, label=name)
        if name == primary_name and len(t) > 2:
            a = np.polyfit(t[1:], p[1:], 1)
            ax[0].plot(t, a[0] * t + a[1], "--", lw=1)

    ax[0].set_xlabel("time since reset [s]")
    ax[0].set_ylabel("region median signal [DN]")
    ax[0].set_title("mean dark ramp (rel. read 1)")
    ax[0].legend()

    d = [summary[n]["dark"] for n in names]
    e = [summary[n]["err"] for n in names]
    ax[1].bar(names, d, yerr=e, capsize=5)
    for i, n in enumerate(names):
        ax[1].text(i, d[i], f"{d[i]:.1f}", ha="center", va="bottom")
    ax[1].axhline(0, color="0.6", lw=.8)
    ax[1].set_ylabel("dark current [e-/s]")
    ax[1].set_title("dark current")

    if primary_mode_key:
        mode_cfg = config[primary_mode_key]
        ramps = parse_ramps(mode_cfg)
        gain = get_gain(mode_cfg)
        itime = float(mode_cfg["frametime"])
        exclude_first = bool(dark_current_cfg.get("histogram_exclude_first", True))
        hist_ramps = ramps[1:] if exclude_first and len(ramps) > 1 else ramps
        maps = [pix_slope(reg_cube(head, date, n, datapath, X0, X1, Y0, Y1)) * gain / itime for n in hist_ramps]
        dmap = np.median(maps, axis=0)
        m, _, s = sigma_clipped_stats(dmap, sigma=3)
        ax[2].hist(dmap.ravel(), bins=120, range=(m - 5 * s, m + 5 * s))
        ax[2].axvline(m, color="k", ls="--", label=f"median {m:.1f} e-/s")
        ax[2].set_xlabel("per-pixel dark current [e-/s]")
        ax[2].set_ylabel("pixels")
        ax[2].set_title(f"{primary_name} per-pixel dark")
        ax[2].legend()
    else:
        ax[2].axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(outpath, f"{setname}_dark_current.png"), dpi=110)
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("Error: No yaml configuration file provided")
        sys.exit(1)

    yamlfile = sys.argv[1]
    with open(yamlfile) as f:
        config = yaml.safe_load(f)

    dataset = config["dataset"]
    head = dataset["head"]
    date = dataset["date"]
    datapath = dataset["datapath"]
    outpath = dataset["outpath"]
    setname = dataset["name"]
    modes = dataset["modes"]

    os.makedirs(outpath, exist_ok=True)

    X0 = config["dark_roi"]["x_start"]
    X1 = config["dark_roi"]["x_end"]
    Y0 = config["dark_roi"]["y_start"]
    Y1 = config["dark_roi"]["y_end"]

    logname = f"{setname}_dark_current.txt"
    summary = {}

    with open(os.path.join(outpath, logname), "w") as log:
        P(log, f"=== Dark current by readout mode (region x[{X0}:{X1}] y[{Y0}:{Y1}]) ===\n")
        P(log, f"{'mode':9s} {'ITIME':>7s} {'ramps':>6s} {'t_tot(s)':>8s} {'dark(e-/s)':>12s} {'+/-':>7s}")

        for mode in modes:
            mode_cfg = config[mode]
            name = mode_cfg["name"]
            itime = float(mode_cfg["frametime"])
            ramps = parse_ramps(mode_cfg)
            gain = get_gain(mode_cfg)

            if not ramps:
                P(log, f"{name:9s} -- no dark ramps configured; skipped --")
                continue
            if gain is None:
                P(log, f"{name:9s} -- no gain configured; skipped --")
                continue

            r = analyze_mode(head, date, datapath, ramps, gain, itime, X0, X1, Y0, Y1)
            summary[name] = r
            P(log, f"{name:9s} {itime:7.3f} {len(ramps):6d} {r['ttot']:8.1f} {r['dark']:12.2f} {r['err']:7.2f}")

        if summary:
            primary_mode_key = config.get("dark_current", {}).get("primary_mode")
            if primary_mode_key and primary_mode_key in config:
                primary_name = config[primary_mode_key]["name"]
            else:
                primary_name = next(iter(summary))
            if primary_name in summary:
                sl = summary[primary_name]
                P(log, "\nNotes:")
                P(log, f"  Primary dark current ({primary_name}): {sl['dark']:.2f} +/- {sl['err']:.2f} e-/s")

    make_plots(summary, config, head, date, datapath, outpath, setname, X0, X1, Y0, Y1)
    print("DONE")


if __name__ == "__main__":
    main()

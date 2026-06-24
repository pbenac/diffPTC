#!/usr/bin/env python3
"""
Detector gain via the two-image DIFFERENCE (mean-variance) method.

This version is YAML-driven and supports either the original two-ramp config

  illuminated:
    n1: 4433
    n2: 4434

or the newer ramp-list style

  illuminated:
    ramps: [4433, 4434, 4435, 4436]

or

  illuminated:
    ramp_range: [4433, 4437]   # inclusive start, exclusive stop, like range()

When more than two illuminated ramps are supplied, pair selection is controlled by

  illuminated:
    pairing: all_pairs          # default for 3+ ramps

Supported pairing modes are: all_pairs, adjacent, first_two.  You can also pass
explicit pairs:

  illuminated:
    pairs:
      - [4433, 4434]
      - [4435, 4436]
"""
import os
import sys
import warnings
from itertools import combinations

import numpy as np
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import yaml
import scipy


def P(log, *a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.write(s + "\n")
    log.flush()


def file_ready(head, date, n, datapath):
    obsnum_string = f"{int(n):05d}"

    if head == "IFS":
        pref = "ss"
    else:
        pref = "si"

    filename = pref + date + "_" + obsnum_string + ".fits"
    fn = os.path.join(datapath, filename)
    if not os.path.exists(fn):
        return False
    try:
        with fits.open(fn) as h:
            nf = h[0].header["NAXIS3"]
            n2 = h[0].header["NAXIS2"]
            n1 = h[0].header["NAXIS1"]
            expect = nf * n1 * n2 * 2
        return os.path.getsize(fn) >= expect
    except Exception:
        return False


def load_region(head, date, n, datapath, X0, X1, Y0, Y1):
    obsnum_string = f"{int(n):05d}"

    if head == "IFS":
        pref = "ss"
    else:
        pref = "si"

    filename = pref + date + "_" + obsnum_string + ".fits"
    fn = os.path.join(datapath, filename)
    h = fits.open(fn, memmap=True, do_not_scale_image_data=True)
    bz = h[0].header.get("BZERO", 0)
    reg = np.asarray(h[0].data[:, Y0:Y1, X0:X1]).astype(np.float64) + bz
    h.close()
    return reg


def row_detrend(img):
    """Remove vertical row-dependent structure by subtracting each row median."""
    return img - np.median(img, axis=1, keepdims=True)


def parse_ramps(mode_cfg, section="illuminated"):
    """Return a list of ramp numbers from a mode config.

    Preferred syntax:
        illuminated:
          ramps: [4433, 4434, 4435]

    Also supported:
        illuminated:
          ramp_range: [4433, 4436]   # Python-style stop-exclusive range

    Backward compatible with:
        illuminated:
          n1: 4433
          n2: 4434
    """
    ramp_cfg = mode_cfg.get(section, {})

    if "ramps" in ramp_cfg:
        return [int(n) for n in ramp_cfg["ramps"]]

    if "ramp_range" in ramp_cfg:
        start, stop = ramp_cfg["ramp_range"]
        return list(range(int(start), int(stop)))

    if "range" in ramp_cfg:
        start, stop = ramp_cfg["range"]
        return list(range(int(start), int(stop)))

    if "start" in ramp_cfg and "stop" in ramp_cfg:
        return list(range(int(ramp_cfg["start"]), int(ramp_cfg["stop"])))

    # Backward compatibility with your existing gain.py YAML.
    if "n1" in ramp_cfg and "n2" in ramp_cfg:
        return [int(ramp_cfg["n1"]), int(ramp_cfg["n2"])]

    return []


def make_pairs(mode_cfg, ramps, section="illuminated"):
    """Return ramp-number pairs according to the mode config."""
    ramp_cfg = mode_cfg.get(section, {})

    if "pairs" in ramp_cfg:
        return [(int(a), int(b)) for a, b in ramp_cfg["pairs"]]

    if len(ramps) < 2:
        return []

    pairing = ramp_cfg.get("pairing", mode_cfg.get("pairing", None))
    if pairing is None:
        pairing = "first_two" if len(ramps) == 2 else "all_pairs"

    pairing = str(pairing).lower()

    if pairing in ("first_two", "first", "first_pair"):
        return [(ramps[0], ramps[1])]

    if pairing in ("adjacent", "consecutive"):
        return list(zip(ramps[:-1], ramps[1:]))

    if pairing in ("all_pairs", "all", "combinations"):
        return list(combinations(ramps, 2))

    raise ValueError(
        f"Unknown pairing mode {pairing!r}. "
        "Use all_pairs, adjacent, first_two, or explicit illuminated.pairs."
    )


def fit_gain(S, V):
    """Fit Var_single = slope*S + intercept and return gain quantities."""
    S = np.asarray(S)
    V = np.asarray(V)

    if len(S) == 0:
        raise ValueError("No signal/variance points available for fit.")

    smax = S.max()
    good = (S > 0.05 * smax) & (S < 0.90 * smax)

    if good.sum() < 2:
        raise ValueError(
            f"Need at least 2 good PTC points for linear fit; got {good.sum()}."
        )

    c, cov = np.polyfit(S[good], V[good], 1, cov=True)
    slope, inter = c
    gerr = np.sqrt(cov[0, 0]) / slope**2
    gain = 1.0 / slope
    rn_dn = np.sqrt(max(inter, 0))
    rn_e = rn_dn * gain

    return dict(
        S=S,
        V=V,
        good=good,
        slope=slope,
        inter=inter,
        gain=gain,
        gain_err=gerr,
        rn_dn=rn_dn,
        rn_e=rn_e,
        smax=smax,
    )


def gain_diff(head, date, n1, n2, datapath, X0, X1, Y0, Y1, log,
              detrend=False, fixed_xmax=None, _quiet=False):
    A = load_region(head, date, n1, datapath, X0, X1, Y0, Y1)
    B = load_region(head, date, n2, datapath, X0, X1, Y0, Y1)
    nf = min(A.shape[0], B.shape[0])
    A1, B1 = A[1], B[1]
    mA1, mB1 = np.median(A1), np.median(B1)

    S, V = [], []
    nbad = 0
    for j in range(2, nf):
        # Guard against corrupt/truncated frames.
        if np.median(A[j]) < 0.5 * mA1 or np.median(B[j]) < 0.5 * mB1:
            nbad += 1
            continue
        da = A[j] - A1
        db = B[j] - B1
        d = db - da
        if detrend:
            d = row_detrend(d)
        _, _, sd = sigma_clipped_stats(d, sigma=3.0, maxiters=5)
        S.append(0.5 * (np.median(da) + np.median(db)))
        V.append(sd**2 / 2.0)

    if nbad and not _quiet:
        P(log, f"   [{n1}/{n2}: skipped {nbad} corrupt/zero frames]")

    r = fit_gain(np.array(S), np.array(V))
    r["nf"] = nf
    r["pair"] = (n1, n2)
    return r


def gain_diff_many_pairs(head, date, pairs, datapath, X0, X1, Y0, Y1, log,
                         detrend=False, fixed_xmax=None):
    """Run gain_diff for each pair and fit once to all pair/read points combined."""
    pair_results = []
    all_S = []
    all_V = []
    nf_values = []

    for n1, n2 in pairs:
        if not (file_ready(head, date, n1, datapath) and file_ready(head, date, n2, datapath)):
            P(log, f"   [{n1}/{n2}: files not ready; skipped]")
            continue

        r = gain_diff(
            head, date, n1, n2, datapath, X0, X1, Y0, Y1, log,
            detrend=detrend, fixed_xmax=fixed_xmax,
        )
        pair_results.append(r)
        all_S.append(r["S"])
        all_V.append(r["V"])
        nf_values.append(r["nf"])

    if not pair_results:
        raise ValueError("No valid illuminated ramp pairs were available.")

    combined = fit_gain(np.concatenate(all_S), np.concatenate(all_V))
    combined["nf"] = int(min(nf_values))
    combined["pairs"] = pairs
    combined["pair_results"] = pair_results
    return combined


def gain_diff_forceIntercept(head, date, n1, n2, datapath, X0, X1, Y0, Y1,
                             intercept, log, detrend=False, fixed_xmax=None,
                             _quiet=False):
    A = load_region(head, date, n1, datapath, X0, X1, Y0, Y1)
    B = load_region(head, date, n2, datapath, X0, X1, Y0, Y1)
    nf = min(A.shape[0], B.shape[0])
    A1, B1 = A[1], B[1]
    mA1, mB1 = np.median(A1), np.median(B1)

    S, V = [], []
    nbad = 0
    for j in range(2, nf):
        if np.median(A[j]) < 0.5 * mA1 or np.median(B[j]) < 0.5 * mB1:
            nbad += 1
            continue
        da = A[j] - A1
        db = B[j] - B1
        d = db - da
        if detrend:
            d = row_detrend(d)
        _, _, sd = sigma_clipped_stats(d, sigma=3.0, maxiters=5)
        S.append(0.5 * (np.median(da) + np.median(db)))
        V.append(sd**2 / 2.0)

    if nbad and not _quiet:
        P(log, f"   [{n1}/{n2}: skipped {nbad} corrupt/zero frames]")

    S = np.array(S)
    V = np.array(V)
    smax = S.max()
    good = (S > 0.05 * smax) & (S < 0.90 * smax)

    y_intercept = intercept

    def linear_model(x, m):
        return m * x + y_intercept

    popt, cov = scipy.optimize.curve_fit(linear_model, S[good], V[good])
    inter = intercept
    slope = popt[0]
    gerr = np.sqrt(cov[0, 0]) / slope**2
    gain = 1.0 / slope
    rn_dn = np.sqrt(max(inter, 0))
    rn_e = rn_dn * gain
    return dict(S=S, V=V, good=good, slope=slope, inter=inter,
                gain=gain, gain_err=gerr, rn_dn=rn_dn, rn_e=rn_e,
                nf=nf, smax=smax, pair=(n1, n2))


def save_pair_summary(outpath, setname, mode_name, pair_results):
    rows = []
    for r in pair_results:
        n1, n2 = r["pair"]
        rows.append([
            n1,
            n2,
            r["gain"],
            r["gain_err"],
            r["rn_e"],
            r["smax"],
            r["nf"],
        ])

    if rows:
        np.savetxt(
            os.path.join(outpath, f"{setname}_pair_gains_{mode_name}.csv"),
            np.asarray(rows),
            header="n1,n2,gain_e_per_DN,gain_err,RN_e,max_signal_DN,nframes",
            delimiter=",",
            comments="",
        )


def plot_results(outpath, setname, results):
    if not results:
        return

    col = {"Slow5.2": "C0", "Fast1.0": "C1", "Fast0.6": "C2"}
    names = list(results.keys())
    fig = plt.figure(figsize=(15, 9))

    # Row 1: per-mode PTC in DN, own axes.
    for i, name in enumerate(names):
        r = results[name]
        c = col.get(name, "k")
        ax = fig.add_subplot(2, 3, i + 1)
        ax.plot(r["S"], r["V"], ".", color="0.6", ms=4)
        ax.plot(r["S"][r["good"]], r["V"][r["good"]], "o", color=c, ms=4)
        xx = np.linspace(0, r["smax"], 100)
        ax.plot(xx, r["slope"] * xx + r["inter"], "-", color=c, lw=1.5)
        ax.set_xlabel("signal S [DN]")
        ax.set_ylabel("variance/image [DN^2]")
        npairs = len(r.get("pair_results", []))
        pair_text = f", {npairs} pairs" if npairs > 1 else ""
        ax.set_title(
            f"{name}: g = {r['gain']:.2f} ± {r['gain_err']:.2f} e-/DN{pair_text}\n"
            f"RN = {r['rn_e']:.0f} e-, ITIME={r['itime']}s"
        )

    # Row 2 left: electron-space collapse.
    axc = fig.add_subplot(2, 3, 4)
    for name in names:
        r = results[name]
        c = col.get(name, "k")
        Se = r["S"] * r["gain"]
        Ve = r["V"] * r["gain"]**2
        axc.plot(Se, Ve, ".", color=c, ms=4, label=name)
    lim = max(r["smax"] * results[n]["gain"] for n, r in results.items())
    axc.plot([0, lim], [0, lim], "k--", lw=1, label="Poisson  var=signal")
    axc.set_xlabel("signal [e-]")
    axc.set_ylabel("variance [e-^2]")
    axc.set_title("electron-space collapse\n(all modes -> Poisson line)")
    axc.legend()

    # Row 2 mid: gain bar chart.
    axb = fig.add_subplot(2, 3, 5)
    g = [results[n]["gain"] for n in names]
    ge = [results[n]["gain_err"] for n in names]
    axb.bar(names, g, yerr=ge, capsize=5, color=[col.get(n, "k") for n in names])
    for i, n in enumerate(names):
        axb.text(i, g[i], f"{g[i]:.2f}", ha="center", va="bottom")
    axb.set_ylabel("gain [e-/DN]")
    axb.set_title("Gain by readout mode")

    # Row 2 right: total electrons accumulated.
    axe = fig.add_subplot(2, 3, 6)
    etot = [results[n]["smax"] * results[n]["gain"] for n in names]
    axe.bar(names, etot, color=[col.get(n, "k") for n in names])
    for i, n in enumerate(names):
        axe.text(i, etot[i], f"{etot[i] / 1e3:.0f}k", ha="center", va="bottom")
    axe.set_ylabel("max accumulated [e-]")
    axe.set_title("total e- per ramp\n(should match: same illumination)")

    fig.tight_layout()
    fig.savefig(os.path.join(outpath, f"{setname}gain_compare.png"), dpi=110)
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("Error: No yaml configuration file provided")
        sys.exit(1)

    yamlfile = sys.argv[1]
    with open(yamlfile) as f:
        config = yaml.safe_load(f)

    head = config["dataset"]["head"]
    date = config["dataset"]["date"]
    datapath = config["dataset"]["datapath"]
    outpath = config["dataset"]["outpath"]
    setname = config["dataset"]["name"]
    modes = config["dataset"]["modes"]

    os.makedirs(outpath, exist_ok=True)

    X0 = config["roi"]["x_start"]
    X1 = config["roi"]["x_end"]
    Y0 = config["roi"]["y_start"]
    Y1 = config["roi"]["y_end"]

    logname = f"{setname}results.txt"

    with open(os.path.join(outpath, logname), "w") as log:
        results = {}

        P(log, "=== gain by readout mode (two-image difference method) ===")
        P(log, f"Region x[{X0}:{X1}] y[{Y0}:{Y1}]\n")
        P(log, f"{'mode':9s} {'ITIME':>8s} {'pairs':>6s} {'reads':>6s} {'maxDN':>7s} "
               f"{'gain(e-/DN)':>12s} {'err':>6s} {'RN(e-)':>7s}")

        for mode in modes:
            mode_cfg = config[mode]
            name = mode_cfg["name"]
            itime = mode_cfg["frametime"]
            detrend = mode_cfg.get("detrend", False)
            fixed_xmax = mode_cfg.get("fixed_xmax", None)

            ramps = parse_ramps(mode_cfg, section="illuminated")
            pairs = make_pairs(mode_cfg, ramps, section="illuminated")

            if not ramps:
                P(log, f"{name:9s} -- no illuminated ramps configured; skipped --")
                continue
            if not pairs:
                P(log, f"{name:9s} -- need at least 2 illuminated ramps; skipped --")
                continue

            P(log, f"   {name}: illuminated ramps={ramps}; pairs={pairs}")

            try:
                r = gain_diff_many_pairs(
                    head, date, pairs, datapath, X0, X1, Y0, Y1, log,
                    detrend=detrend, fixed_xmax=fixed_xmax,
                )
            except Exception as exc:
                P(log, f"{name:9s} -- gain fit failed: {exc}; skipped --")
                continue

            r["itime"] = itime
            results[name] = r

            P(log, f"{name:9s} {itime:8.4f} {len(r['pair_results']):6d} {r['nf']-2:6d} {r['smax']:7.0f} "
                   f"{r['gain']:12.3f} {r['gain_err']:6.3f} {r['rn_e']:7.1f}"
                   + ("   [row-detrended]" if detrend else ""))

            if detrend and len(pairs) == 1:
                n1, n2 = pairs[0]
                r0 = gain_diff(
                    head, date, n1, n2, datapath, X0, X1, Y0, Y1, log,
                    detrend=False, _quiet=True, fixed_xmax=fixed_xmax,
                )
                P(log, f"          (without sinusoid removal: gain={r0['gain']:.2f}±{r0['gain_err']:.2f}, "
                       f"RN={r0['rn_e']:.0f} e-  -> detrend tightens error ~{r0['gain_err']/r['gain_err']:.0f}x)")

            np.savetxt(
                os.path.join(outpath, f"{setname}_ptc_{name}.csv"),
                np.column_stack([r["S"], r["V"]]),
                header="signal_DN,var_single_DN2",
                delimiter=",",
                comments="",
            )
            save_pair_summary(outpath, setname, name, r["pair_results"])

        plot_results(outpath, setname, results)


if __name__ == "__main__":
    main()

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

Optional PTC read selection can be set globally:

  analysis:
    first_read: 2
    last_read: 8

or overridden for an individual mode:

  fast06:
    first_read: 2
    last_read: 6

Read numbers here are zero-based FITS-cube indices, matching the existing code.
The reference image remains read index 1.  first_read/last_read only control
which later reads become PTC points; last_read is inclusive.  If omitted, the
current default behavior is preserved: use reads 2 through the end of the ramp.
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
import gui_io


def P(log, *a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.write(s + "\n")
    log.flush()



def as_optional_float(value):
    """Convert YAML values like null/'None'/'' to None, otherwise float."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null"):
        return None
    return float(value)


def as_optional_int(value):
    """Convert YAML values like null/'None'/'' to None, otherwise int."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null"):
        return None
    return int(value)


def selected_read_bounds(nf, first_read=None, last_read=None):
    """Return inclusive PTC read bounds, preserving the historical defaults.

    The reference read is always array index 1, so PTC points cannot start
    before index 2.  last_read is inclusive.
    """
    first = 2 if first_read is None else int(first_read)
    last = nf - 1 if last_read is None else int(last_read)

    if first < 2:
        raise ValueError(
            f"first_read={first} is invalid: the PTC reference is read index 1, "
            "so first_read must be >= 2."
        )
    if last < first:
        raise ValueError(
            f"last_read={last} must be >= first_read={first}."
        )
    if first >= nf:
        raise ValueError(
            f"first_read={first} is outside a ramp with {nf} reads."
        )

    last = min(last, nf - 1)
    return first, last


def prefix_for_head(head):
    return "ss" if head == "IFS" else "si"


def data_directory(datapath, quicklook=False):
    """Return the directory containing FITS files for normal or quicklook products."""
    return os.path.join(datapath, "ql_redux") if quicklook else datapath


def obs_filename(head, date, n, quicklook=False):
    """Build detector FITS filename.

    Normal:
        ss260625_21740.fits

    Quicklook:
        ss260625_21740_reads.fits
    """
    suffix = "_reads" if quicklook else ""
    return f"{prefix_for_head(head)}{date}_{int(n):05d}{suffix}.fits"


def fits_path(datapath, head, date, n, quicklook=False):
    return os.path.join(data_directory(datapath, quicklook), obs_filename(head, date, n, quicklook))


def file_ready(head, date, n, datapath, quicklook=False, gui=False):
    if gui:
        return gui_io.gui_acquisition_ready(n, min_reads=3)

    fn = fits_path(datapath, head, date, n, quicklook=quicklook)
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


def load_region(head, date, n, datapath, X0, X1, Y0, Y1, quicklook=False, gui=False,  # LLM-generated: split to add last_read
                last_read=None):  # LLM-generated: last read the caller needs
    # LLM-generated docstring
    """Return (reads 0..last_read of the ROI, total reads in the ramp).

    Array indices match the full cube, and nf is always the full read count. At least
    reads 0 and 1 are kept: read 1 is the reference, and it is used before last_read
    is validated. GUI mode still loads every read.
    """
    if gui:
        cube, _header, _source = gui_io.load_gui_cube(n, roi=(X0, X1, Y0, Y1))
        return cube, cube.shape[0]  # LLM-generated: also return the read count

    fn = fits_path(datapath, head, date, n, quicklook=quicklook)
    h = fits.open(fn, memmap=True, do_not_scale_image_data=True)
    bz = h[0].header.get("BZERO", 0)
    data = h[0].data  # LLM-generated
    nf = data.shape[0]  # LLM-generated: full read count; loads no pixels
    stop = None if last_read is None else max(int(last_read) + 1, 2)  # LLM-generated: keep reads 0 and 1
    reg = np.asarray(data[:stop, Y0:Y1, X0:X1]).astype(np.float64) + bz  # LLM-generated: load only reads before stop
    h.close()
    return reg, nf  # LLM-generated: also return the read count


def row_detrend(img):
    """Remove vertical row-dependent structure by subtracting each row median."""
    return img - np.median(img, axis=1, keepdims=True)


def parse_ramps(mode_cfg, section="illuminated", gui=False):
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
    if gui:
        return gui_io.gui_identifiers(mode_cfg, section=section)

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


def make_pairs(mode_cfg, ramps, section="illuminated", gui=False):
    """Return ramp-number pairs according to the mode config."""
    ramp_cfg = mode_cfg.get(section, {})

    if "pairs" in ramp_cfg:
        if gui:
            return [(str(a), str(b)) for a, b in ramp_cfg["pairs"]]
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


def fit_gain(S, V, fixed_xmax=None):
    """Fit Var_single = slope*S + intercept and return gain quantities."""
    S = np.asarray(S)
    V = np.asarray(V)

    if len(S) == 0:
        raise ValueError("No signal/variance points available for fit.")

    if fixed_xmax is None:
        smax = S.max()
    else:
        smax = fixed_xmax

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
              detrend=False, fixed_xmax=None, quicklook=False, gui=False,
              first_read=None, last_read=None, _quiet=False):
    A, nf_a = load_region(head, date, n1, datapath, X0, X1, Y0, Y1, quicklook=quicklook, gui=gui,  # LLM-generated
                          last_read=last_read)  # LLM-generated
    B, nf_b = load_region(head, date, n2, datapath, X0, X1, Y0, Y1, quicklook=quicklook, gui=gui,  # LLM-generated
                          last_read=last_read)  # LLM-generated
    nf = min(nf_a, nf_b)  # LLM-generated: full read counts, not the loaded array sizes
    A1, B1 = A[1], B[1]
    mA1, mB1 = np.median(A1), np.median(B1)

    read_first, read_last = selected_read_bounds(nf, first_read, last_read)

    S, V = [], []
    read_indices = []
    nbad = 0
    for j in range(read_first, read_last + 1):
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
        read_indices.append(j)

    if nbad and not _quiet:
        P(log, f"   [{n1}/{n2}: skipped {nbad} corrupt/zero frames]")

    r = fit_gain(np.array(S), np.array(V), fixed_xmax=fixed_xmax)
    r["nf"] = nf
    r["pair"] = (n1, n2)
    r["read_indices"] = np.asarray(read_indices, dtype=int)
    r["first_read"] = read_first
    r["last_read"] = read_last
    r["nreads_used"] = len(read_indices)
    return r


def gain_diff_many_pairs(head, date, pairs, datapath, X0, X1, Y0, Y1, log,
                         detrend=False, fixed_xmax=None, quicklook=False, gui=False,
                         first_read=None, last_read=None):
    """Run gain_diff for each pair and fit once to all pair/read points combined."""
    pair_results = []
    all_S = []
    all_V = []
    nf_values = []

    for n1, n2 in pairs:
        if not (file_ready(head, date, n1, datapath, quicklook=quicklook, gui=gui) and file_ready(head, date, n2, datapath, quicklook=quicklook, gui=gui)):
            P(log, f"   [{n1}/{n2}: files not ready; skipped]")
            continue

        r = gain_diff(
            head, date, n1, n2, datapath, X0, X1, Y0, Y1, log,
            detrend=detrend, fixed_xmax=fixed_xmax, quicklook=quicklook, gui=gui,
            first_read=first_read, last_read=last_read,
        )
        pair_results.append(r)
        all_S.append(r["S"])
        all_V.append(r["V"])
        nf_values.append(r["nf"])

    if not pair_results:
        raise ValueError("No valid illuminated ramp pairs were available.")

    combined = fit_gain(np.concatenate(all_S), np.concatenate(all_V), fixed_xmax=fixed_xmax)
    combined["nf"] = int(min(nf_values))
    combined["pairs"] = pairs
    combined["pair_results"] = pair_results
    combined["first_read"] = min(r["first_read"] for r in pair_results)
    combined["last_read"] = min(r["last_read"] for r in pair_results)
    combined["nreads_used"] = int(sum(r["nreads_used"] for r in pair_results))
    return combined


def gain_diff_forceIntercept(head, date, n1, n2, datapath, X0, X1, Y0, Y1,
                             intercept, log, detrend=False, fixed_xmax=None,
                             quicklook=False, gui=False, first_read=None,
                             last_read=None, _quiet=False):
    A, nf_a = load_region(head, date, n1, datapath, X0, X1, Y0, Y1, quicklook=quicklook, gui=gui,  # LLM-generated
                          last_read=last_read)  # LLM-generated
    B, nf_b = load_region(head, date, n2, datapath, X0, X1, Y0, Y1, quicklook=quicklook, gui=gui,  # LLM-generated
                          last_read=last_read)  # LLM-generated
    nf = min(nf_a, nf_b)  # LLM-generated: full read counts, not the loaded array sizes
    A1, B1 = A[1], B[1]
    mA1, mB1 = np.median(A1), np.median(B1)

    read_first, read_last = selected_read_bounds(nf, first_read, last_read)

    S, V = [], []
    read_indices = []
    nbad = 0
    for j in range(read_first, read_last + 1):
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
        read_indices.append(j)

    if nbad and not _quiet:
        P(log, f"   [{n1}/{n2}: skipped {nbad} corrupt/zero frames]")

    S = np.array(S)
    V = np.array(V)
    smax = S.max() if fixed_xmax is None else fixed_xmax
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
                nf=nf, smax=smax, pair=(n1, n2),
                read_indices=np.asarray(read_indices, dtype=int),
                first_read=read_first, last_read=read_last,
                nreads_used=len(read_indices))


def save_pair_summary(outpath, setname, mode_name, pair_results):
    """Write per-pair gain results; identifiers may be ramp numbers or GUI timestamps."""
    fn = os.path.join(outpath, f"{setname}_pair_gains_{mode_name}.csv")
    with open(fn, "w") as f:
        f.write("n1,n2,gain_e_per_DN,gain_err,RN_e,max_signal_DN,nframes\n")
        for r in pair_results:
            n1, n2 = r["pair"]
            f.write(
                f"{n1},{n2},{r['gain']},{r['gain_err']},{r['rn_e']},"
                f"{r['smax']},{r['nf']}\n"
            )


def mode_color_map(names):
    """Assign each plotted mode a stable color from Matplotlib's color cycle.

    This avoids depending on exact mode-name strings like 'Slow5.2'. If the YAML
    names change, the first mode still gets C0, the second C1, etc.
    """
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colors:
        colors = [f"C{i}" for i in range(10)]
    return {name: colors[i % len(colors)] for i, name in enumerate(names)}


def plot_results(outpath, setname, results):
    if not results:
        return

    names = list(results.keys())
    col = mode_color_map(names)
    fig = plt.figure(figsize=(15, 9))

    # Row 1: per-mode PTC in DN, own axes.
    for i, name in enumerate(names):
        r = results[name]
        c = col[name]
        ax = fig.add_subplot(2, 3, i + 1)
        ax.plot(r["S"], r["V"], ".", color="0.6", ms=4)
        ax.plot(r["S"][r["good"]], r["V"][r["good"]], "o", color=c, ms=4)
        xx = np.linspace(0, r["smax"], 100)
        ax.plot(xx, r["slope"] * xx + r["inter"], "-", color=c, lw=1.5)
        ax.set_xlabel("signal S [DN]")
        ax.set_ylabel(r"variance/image [DN$^2$]")
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
        c = col[name]
        Se = r["S"] * r["gain"]
        Ve = r["V"] * r["gain"]**2
        axc.plot(Se, Ve, ".", color=c, ms=4, label=name)
    lim = max(r["smax"] * results[n]["gain"] for n, r in results.items())
    axc.plot([0, lim], [0, lim], "k--", lw=1, label="Poisson: var=signal")
    axc.set_xlabel("signal [e-]")
    axc.set_ylabel(r"variance [e-$^2$]")
    axc.set_title("Photon Transfer Curve after Gain Conversion \n(all modes -> Poisson line)")
    axc.legend()

    # Row 2 mid: gain bar chart.
    axb = fig.add_subplot(2, 3, 5)
    g = [results[n]["gain"] for n in names]
    ge = [results[n]["gain_err"] for n in names]
    axb.bar(names, g, yerr=ge, capsize=5, color=[col[n] for n in names])
    for i, n in enumerate(names):
        axb.text(i, g[i], f"{g[i]:.2f}", ha="center", va="bottom")
    axb.set_ylabel("gain [e-/DN]")
    axb.set_title("Gain by readout mode")

    # Row 2 right: total electrons accumulated.
    axe = fig.add_subplot(2, 3, 6)
    etot = [results[n]["smax"] * results[n]["gain"] for n in names]
    axe.bar(names, etot, color=[col[n] for n in names])
    for i, n in enumerate(names):
        axe.text(i, etot[i], f"{etot[i] / 1e3:.0f}k", ha="center", va="bottom")
    axe.set_ylabel("max accumulated [e-]")
    axe.set_title("Total e- per ramp\n(should match: same illumination)")

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

    gui = gui_io.is_gui_mode(config)
    head = config["dataset"].get("head", "GUI")
    date = config["dataset"].get("date", "")
    datapath = config["dataset"].get("datapath", "")
    quicklook = False if gui else bool(config["dataset"].get("quicklook", False))
    outpath = config["dataset"]["outpath"]
    setname = config["dataset"]["name"]
    modes = config["dataset"]["modes"]
    analysis_cfg = config.get("analysis", {})

    os.makedirs(outpath, exist_ok=True)

    X0 = config["roi"]["x_start"]
    X1 = config["roi"]["x_end"]
    Y0 = config["roi"]["y_start"]
    Y1 = config["roi"]["y_end"]

    logname = f"{setname}results.txt"

    with open(os.path.join(outpath, logname), "w") as log:
        results = {}

        P(log, "=== gain by readout mode (two-image difference method) ===")
        P(log, f"Region x[{X0}:{X1}] y[{Y0}:{Y1}]")
        if gui:
            P(log, f"GUI Live Mode=True; data root={gui_io.GUI_BASE_PATH}\n")
        else:
            P(log, f"quicklook={quicklook}; data directory={data_directory(datapath, quicklook)}\n")
        P(log, f"{'mode':9s} {'ITIME':>8s} {'pairs':>6s} {'reads':>6s} {'maxDN':>7s} "
               f"{'gain(e-/DN)':>12s} {'err':>6s} {'RN(e-)':>7s}")

        for mode in modes:
            mode_cfg = config[mode]
            name = mode_cfg["name"]
            itime = mode_cfg["frametime"]
            detrend = mode_cfg.get("detrend", False)
            fixed_xmax = as_optional_float(mode_cfg.get("fixed_xmax", None))

            # Per-mode analysis.read limits override the global analysis defaults.
            mode_analysis = mode_cfg.get("analysis", {})
            first_read = as_optional_int(
                mode_analysis.get("first_read", analysis_cfg.get("first_read", None))
            )
            last_read = as_optional_int(
                mode_analysis.get("last_read", analysis_cfg.get("last_read", None))
            )

            ramps = parse_ramps(mode_cfg, section="illuminated", gui=gui)
            pairs = make_pairs(mode_cfg, ramps, section="illuminated", gui=gui)

            if not ramps:
                P(log, f"{name:9s} -- no illuminated ramps configured; skipped --")
                continue
            if not pairs:
                P(log, f"{name:9s} -- need at least 2 illuminated ramps; skipped --")
                continue

            read_text = (
                f"reads={first_read if first_read is not None else 2}:"
                f"{last_read if last_read is not None else 'end'} (inclusive)"
            )
            P(log, f"   {name}: illuminated ramps={ramps}; pairs={pairs}; {read_text}")

            try:
                r = gain_diff_many_pairs(
                    head, date, pairs, datapath, X0, X1, Y0, Y1, log,
                    detrend=detrend, fixed_xmax=fixed_xmax, quicklook=quicklook, gui=gui,
                    first_read=first_read, last_read=last_read,
                )
            except Exception as exc:
                P(log, f"{name:9s} -- gain fit failed: {exc}; skipped --")
                continue

            r["itime"] = itime
            results[name] = r

            display_nreads = min(pr["nreads_used"] for pr in r["pair_results"])
            P(log, f"{name:9s} {itime:8.4f} {len(r['pair_results']):6d} {display_nreads:6d} {r['smax']:7.0f} "
                   f"{r['gain']:12.3f} {r['gain_err']:6.3f} {r['rn_e']:7.1f}"
                   + ("   [row-detrended]" if detrend else ""))

            if detrend and len(pairs) == 1:
                n1, n2 = pairs[0]
                r0 = gain_diff(
                    head, date, n1, n2, datapath, X0, X1, Y0, Y1, log,
                    detrend=False, _quiet=True, fixed_xmax=fixed_xmax, quicklook=quicklook, gui=gui,
                    first_read=first_read, last_read=last_read,
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

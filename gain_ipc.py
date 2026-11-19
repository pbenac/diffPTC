#!/usr/bin/env python3
# LLM-generated (Claude). Experimental copy of gain.py with an interpixel capacitance (IPC)
# correction. gain.py itself is untouched; this file imports its helpers so the uncorrected
# ("raw") numbers are computed by exactly the same code.
"""
Detector gain via the two-image difference method, corrected for interpixel capacitance.

Why IPC biases the PTC gain
---------------------------
IPC couples a fraction of each pixel's signal into its neighbours. With a kernel k that sums
to 1 (signal is conserved), the mean signal S is unchanged, but the per-pixel shot-noise
variance is reduced by sum(k**2) < 1, because the noise is smeared into the neighbours:

    Var_pixel = sum(k**2) * S / g_true + RN**2

so the raw PTC gain 1/slope overestimates the true gain: g_raw = g_true / sum(k**2).
The variance that leaves the pixel shows up as covariance with its neighbours
(e.g. C(1,0) ~ 2*alpha*S/g for a nearest-neighbour coupling alpha).

Read noise is added after the coupling (in the readout chain), so it is not spread and the
DN read noise from the PTC intercept is unaffected. RN(e-) is reported as rn_dn * g_corrected.

Correction methods (YAML key ipc.method)
----------------------------------------
kernel (default)
    Measure the neighbour covariances C(1,0) and C(0,1) of each difference image, fit each
    linearly against signal like the variance, and solve the slope ratios for a
    nearest-neighbour kernel (alpha_x, alpha_y; centre 1 - 2*alpha_x - 2*alpha_y).
    g_corr = g_raw * sum(k**2). Fitting slopes vs signal means constant (signal-independent)
    correlated read noise goes into the intercepts and does not bias alpha.
covariance_sum
    Sum the covariances over a (2*max_lag+1)^2 window, which recovers the full shot-noise
    variance for any kernel up to that size (also removes most of the brighter-fatter
    effect). g_corr = 1/slope of that summed PTC. Noisier than 'kernel', and its intercept
    includes any spatially correlated read noise, so RN still comes from the raw intercept.
fixed
    Use a supplied alpha (alpha, or alpha_x and alpha_y) instead of measuring it.

All three estimates are always computed and logged; ipc.method only picks which one is the
headline "gain_ipc" and goes in the plots.

YAML
----
Same config as gain.py, plus an optional section (global, or per mode to override):

  ipc:
    method: kernel          # kernel | covariance_sum | fixed
    max_lag: 2              # window half-width for covariance_sum
    alpha: 0.015            # only for method: fixed (or alpha_x / alpha_y)

Outputs (named so they never overwrite gain.py outputs):
  <name>results_ipc.txt, <name>_ptc_ipc_<mode>.csv, <name>_pair_gains_ipc_<mode>.csv,
  <name>gain_ipc_compare.png
"""
import os
import sys
import warnings

import numpy as np
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.stats import sigma_clip, sigma_clipped_stats
import yaml
import data_paths  # LLM-generated

import gain as G
import gui_io

P = G.P


# ----------------------------------------------------------------------------------------
# Covariance measurement
# ----------------------------------------------------------------------------------------
def half_plane_lags(max_lag):
    """Non-zero lags (dx, dy) covering half the window; C(-l) = C(l) gives the other half."""
    lags = [(dx, 0) for dx in range(1, max_lag + 1)]
    for dy in range(1, max_lag + 1):
        for dx in range(-max_lag, max_lag + 1):
            lags.append((dx, dy))
    return lags


def _overlap(arr, dx, dy):
    """Return (arr[y, x], arr[y+dy, x+dx]) views over the overlapping region."""
    ny, nx = arr.shape
    y0, y1 = max(0, -dy), ny - max(0, dy)
    x0, x1 = max(0, -dx), nx - max(0, dx)
    return arr[y0:y1, x0:x1], arr[y0 + dy:y1 + dy, x0 + dx:x1 + dx]


def lag_covariances(d, lags, sigma=3.0, maxiters=5):
    """Covariance of image d at each lag, using only pixel pairs that both survive clipping.

    Returns (C00, {lag: C}). x is the column (fast) axis, y the row axis.
    """
    mask = np.ma.getmaskarray(sigma_clip(d, sigma=sigma, maxiters=maxiters))
    w = ~mask
    z = np.where(w, d - d[w].mean(), 0.0)
    c00 = float(np.sum(z * z) / w.sum())
    out = {}
    for dx, dy in lags:
        za, zb = _overlap(z, dx, dy)
        wa, wb = _overlap(w, dx, dy)
        n = np.count_nonzero(wa & wb)
        out[(dx, dy)] = float(np.sum(za * zb) / n) if n else np.nan
    return c00, out


# ----------------------------------------------------------------------------------------
# Kernel algebra
# ----------------------------------------------------------------------------------------
def kernel_sum_sq(alpha_x, alpha_y):
    k0 = 1.0 - 2.0 * alpha_x - 2.0 * alpha_y
    return k0**2 + 2.0 * alpha_x**2 + 2.0 * alpha_y**2


def solve_alpha(r10, r01, tol=1e-12, maxiter=200):
    """Nearest-neighbour kernel from covariance ratios r10 = C(1,0)/C(0,0), r01 = C(0,1)/C(0,0).

    For centre k0 = 1 - 2ax - 2ay and neighbours ax (x), ay (y):
        C(1,0)/C(0,0) = 2*ax*k0 / sum(k**2),   C(0,1)/C(0,0) = 2*ay*k0 / sum(k**2)
    Solved by fixed-point iteration (converges quickly for |alpha| << 0.1).
    """
    ax, ay = r10 / 2.0, r01 / 2.0
    for _ in range(maxiter):
        k0 = 1.0 - 2.0 * ax - 2.0 * ay
        s2 = kernel_sum_sq(ax, ay)
        ax_new = r10 * s2 / (2.0 * k0)
        ay_new = r01 * s2 / (2.0 * k0)
        if abs(ax_new - ax) < tol and abs(ay_new - ay) < tol:
            return ax_new, ay_new
        ax, ay = ax_new, ay_new
    raise ValueError(f"IPC alpha did not converge (r10={r10:.4g}, r01={r01:.4g}).")


def _slope(S, Y, good):
    good = good & np.isfinite(Y)
    if good.sum() < 2:
        return np.nan, np.nan
    c = np.polyfit(S[good], Y[good], 1)
    return float(c[0]), float(c[1])


# ----------------------------------------------------------------------------------------
# Per-pair measurement
# ----------------------------------------------------------------------------------------
def ipc_pair_points(head, date, n1, n2, datapath, X0, X1, Y0, Y1, log, lags,
                    detrend=False, quicklook=False, gui=False,
                    first_read=None, last_read=None):
    """Same read loop as gain.gain_diff, plus lag covariances of each difference image.

    S and V are computed with the same statements as gain.gain_diff, so they match it exactly.
    """
    A, nf_a = G.load_region(head, date, n1, datapath, X0, X1, Y0, Y1, quicklook=quicklook,
                            gui=gui, last_read=last_read)
    B, nf_b = G.load_region(head, date, n2, datapath, X0, X1, Y0, Y1, quicklook=quicklook,
                            gui=gui, last_read=last_read)
    nf = min(nf_a, nf_b)
    A1, B1 = A[1], B[1]
    mA1, mB1 = np.median(A1), np.median(B1)
    read_first, read_last = G.selected_read_bounds(nf, first_read, last_read)

    S, V, C00, read_indices = [], [], [], []
    C = {lag: [] for lag in lags}
    nbad = 0
    for j in range(read_first, read_last + 1):
        if np.median(A[j]) < 0.5 * mA1 or np.median(B[j]) < 0.5 * mB1:
            nbad += 1
            continue
        da = A[j] - A1
        db = B[j] - B1
        d = db - da
        if detrend:
            d = G.row_detrend(d)
        _, _, sd = sigma_clipped_stats(d, sigma=3.0, maxiters=5)
        S.append(0.5 * (np.median(da) + np.median(db)))
        V.append(sd**2 / 2.0)
        c00, cl = lag_covariances(d, lags)
        C00.append(c00 / 2.0)
        for lag in lags:
            C[lag].append(cl[lag] / 2.0)
        read_indices.append(j)

    if nbad:
        P(log, f"   [{n1}/{n2}: skipped {nbad} corrupt/zero frames]")

    return dict(S=np.array(S), V=np.array(V), C00=np.array(C00),
                C={lag: np.array(v) for lag, v in C.items()},
                nf=nf, pair=(n1, n2), read_indices=np.asarray(read_indices, dtype=int),
                first_read=read_first, last_read=read_last, nreads_used=len(read_indices))


def ipc_fit(pts, lags, ipc_cfg, fixed_xmax=None):
    """Raw fit (gain.fit_gain) plus the three IPC-corrected gain estimates."""
    S = pts["S"]
    r = G.fit_gain(S, pts["V"], fixed_xmax=fixed_xmax)
    good = r["good"]
    raw_gain = r["gain"]

    # Slopes of each covariance vs signal, normalised to the slope of the same-mask C(0,0).
    b00, _ = _slope(S, pts["C00"], good)
    b10, _ = _slope(S, pts["C"][(1, 0)], good)
    b01, _ = _slope(S, pts["C"][(0, 1)], good)
    b11, _ = _slope(S, pts["C"][(1, 1)], good) if (1, 1) in pts["C"] else (np.nan, np.nan)
    r10, r01, r11 = b10 / b00, b01 / b00, b11 / b00

    try:
        ax, ay = solve_alpha(r10, r01)
        s2_kernel = kernel_sum_sq(ax, ay)
    except ValueError:
        ax = ay = s2_kernel = np.nan
    g_kernel = raw_gain * s2_kernel

    # Summed-covariance PTC: C00 + 2 * sum over half-plane lags.
    Vsum = pts["C00"] + 2.0 * sum(pts["C"][lag] for lag in lags)
    rsum = G.fit_gain(S, Vsum, fixed_xmax=fixed_xmax)
    g_covsum = rsum["gain"]

    # Fixed alpha from config (if given).
    fax = G.as_optional_float(ipc_cfg.get("alpha_x", ipc_cfg.get("alpha", None)))
    fay = G.as_optional_float(ipc_cfg.get("alpha_y", ipc_cfg.get("alpha", None)))
    if fax is not None and fay is not None:
        s2_fixed = kernel_sum_sq(fax, fay)
        g_fixed = raw_gain * s2_fixed
    else:
        s2_fixed = g_fixed = np.nan

    method = str(ipc_cfg.get("method", "kernel")).lower()
    if method == "kernel":
        g_ipc, g_ipc_err, s2 = g_kernel, r["gain_err"] * s2_kernel, s2_kernel
    elif method in ("covariance_sum", "covsum"):
        g_ipc, g_ipc_err, s2 = g_covsum, rsum["gain_err"], g_covsum / raw_gain
    elif method == "fixed":
        if fax is None or fay is None:
            raise ValueError("ipc.method=fixed needs ipc.alpha (or alpha_x and alpha_y).")
        g_ipc, g_ipc_err, s2 = g_fixed, r["gain_err"] * s2_fixed, s2_fixed
    else:
        raise ValueError(f"Unknown ipc.method {method!r}; use kernel, covariance_sum or fixed.")

    r.update(
        raw_gain=raw_gain, raw_gain_err=r["gain_err"], raw_rn_e=r["rn_e"],
        r10=r10, r01=r01, r11=r11, alpha_x=ax, alpha_y=ay,
        sum_k2=s2, gain_kernel=g_kernel, gain_covsum=g_covsum, gain_fixed=g_fixed,
        covsum_fit=rsum, Vsum=Vsum, C00=pts["C00"], C=pts["C"],
        ipc_method=method, gain_ipc=g_ipc, gain_ipc_err=g_ipc_err,
        rn_e_ipc=r["rn_dn"] * g_ipc,
    )
    return r


def ipc_many_pairs(head, date, pairs, datapath, X0, X1, Y0, Y1, log, ipc_cfg,
                   detrend=False, fixed_xmax=None, quicklook=False, gui=False,
                   first_read=None, last_read=None):
    """Per-pair fits plus one combined fit to all pair/read points (as in gain.py)."""
    max_lag = int(ipc_cfg.get("max_lag", 2))
    if max_lag < 1:
        raise ValueError("ipc.max_lag must be >= 1.")
    lags = half_plane_lags(max_lag)

    pair_results, all_pts = [], []
    for n1, n2 in pairs:
        if not (G.file_ready(head, date, n1, datapath, quicklook=quicklook, gui=gui)
                and G.file_ready(head, date, n2, datapath, quicklook=quicklook, gui=gui)):
            P(log, f"   [{n1}/{n2}: files not ready; skipped]")
            continue
        pts = ipc_pair_points(head, date, n1, n2, datapath, X0, X1, Y0, Y1, log, lags,
                              detrend=detrend, quicklook=quicklook, gui=gui,
                              first_read=first_read, last_read=last_read)
        pr = ipc_fit(pts, lags, ipc_cfg, fixed_xmax=fixed_xmax)
        for k in ("nf", "pair", "read_indices", "first_read", "last_read", "nreads_used"):
            pr[k] = pts[k]
        pair_results.append(pr)
        all_pts.append(pts)

    if not pair_results:
        raise ValueError("No valid illuminated ramp pairs were available.")

    combined_pts = dict(
        S=np.concatenate([p["S"] for p in all_pts]),
        V=np.concatenate([p["V"] for p in all_pts]),
        C00=np.concatenate([p["C00"] for p in all_pts]),
        C={lag: np.concatenate([p["C"][lag] for p in all_pts]) for lag in lags},
    )
    combined = ipc_fit(combined_pts, lags, ipc_cfg, fixed_xmax=fixed_xmax)
    combined["nf"] = int(min(r["nf"] for r in pair_results))
    combined["pairs"] = pairs
    combined["pair_results"] = pair_results
    combined["first_read"] = min(r["first_read"] for r in pair_results)
    combined["last_read"] = min(r["last_read"] for r in pair_results)
    combined["nreads_used"] = int(sum(r["nreads_used"] for r in pair_results))
    return combined


# ----------------------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------------------
def save_ptc_csv(outpath, setname, name, r):
    lags = sorted(r["C"].keys(), key=lambda l: (l[1], l[0]))
    cols = [r["S"], r["V"], r["C00"]] + [r["C"][l] for l in lags] + [r["Vsum"]]
    header = ",".join(["signal_DN", "var_single_DN2", "C00_masked_DN2"]
                      + [f"C{dx}_{dy}_DN2" for dx, dy in lags] + ["Vsum_DN2"])
    np.savetxt(os.path.join(outpath, f"{setname}_ptc_ipc_{name}.csv"),
               np.column_stack(cols), header=header, delimiter=",", comments="")


def save_pair_summary(outpath, setname, name, pair_results):
    fn = os.path.join(outpath, f"{setname}_pair_gains_ipc_{name}.csv")
    with open(fn, "w") as f:
        f.write("n1,n2,gain_raw,gain_raw_err,alpha_x,alpha_y,sum_k2,gain_kernel,"
                "gain_covsum,gain_ipc,gain_ipc_err,RN_e_ipc,max_signal_DN,nframes\n")
        for r in pair_results:
            n1, n2 = r["pair"]
            f.write(f"{n1},{n2},{r['raw_gain']},{r['raw_gain_err']},{r['alpha_x']},"
                    f"{r['alpha_y']},{r['sum_k2']},{r['gain_kernel']},{r['gain_covsum']},"
                    f"{r['gain_ipc']},{r['gain_ipc_err']},{r['rn_e_ipc']},{r['smax']},{r['nf']}\n")


def plot_results(outpath, setname, results):
    if not results:
        return
    names = list(results.keys())
    col = G.mode_color_map(names)
    n = len(names)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8), squeeze=False)
    for i, name in enumerate(names):
        r = results[name]
        c = col[name]
        S, good = r["S"], r["good"]
        xx = np.linspace(0, r["smax"], 100)

        ax = axes[0, i]
        ax.plot(S, r["V"], ".", color="0.6", ms=4, label="raw variance")
        ax.plot(xx, r["slope"] * xx + r["inter"], "-", color="0.4", lw=1)
        if np.isfinite(r["sum_k2"]) and r["sum_k2"] > 0:
            Vk = (r["V"] - r["inter"]) / r["sum_k2"] + r["inter"]
            ax.plot(S[good], Vk[good], "o", color=c, ms=4, label="IPC-corrected")
            ax.plot(xx, xx / r["gain_ipc"] + r["inter"], "-", color=c, lw=1.5)
        rs = r["covsum_fit"]
        ax.plot(S, r["Vsum"], "x", color="k", ms=4, alpha=0.5, label="summed covariance")
        ax.set_xlabel("signal S [DN]")
        ax.set_ylabel(r"variance/image [DN$^2$]")
        ax.set_title(f"{name} ({r['ipc_method']}): g_ipc = {r['gain_ipc']:.3f} ± "
                     f"{r['gain_ipc_err']:.3f}\nraw {r['raw_gain']:.3f}, "
                     f"covsum {rs['gain']:.3f} e-/DN")
        ax.legend(fontsize=8)

        ax = axes[1, i]
        with np.errstate(divide="ignore", invalid="ignore"):
            ax.plot(S, r["C"][(1, 0)] / r["C00"], "o", ms=3, label="C(1,0)/C(0,0)  x")
            ax.plot(S, r["C"][(0, 1)] / r["C00"], "s", ms=3, label="C(0,1)/C(0,0)  y")
            if (1, 1) in r["C"]:
                ax.plot(S, r["C"][(1, 1)] / r["C00"], "^", ms=3, label="C(1,1)/C(0,0)")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("signal S [DN]")
        ax.set_ylabel("neighbour correlation")
        ax.set_title(f"alpha_x = {r['alpha_x']:.4f}, alpha_y = {r['alpha_y']:.4f}\n"
                     f"(from slopes vs signal)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outpath, f"{setname}gain_ipc_compare.png"), dpi=110)
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("Error: No yaml configuration file provided")
        sys.exit(1)

    with open(sys.argv[1]) as f:
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
    ipc_global = config.get("ipc", {}) or {}

    os.makedirs(outpath, exist_ok=True)
    X0, X1 = config["roi"]["x_start"], config["roi"]["x_end"]
    Y0, Y1 = config["roi"]["y_start"], config["roi"]["y_end"]

    with open(os.path.join(outpath, f"{setname}results_ipc.txt"), "w") as log:
        results = {}
        P(log, "=== gain by readout mode, IPC-corrected (two-image difference method) ===")
        P(log, f"Region x[{X0}:{X1}] y[{Y0}:{Y1}]")
        if gui:
            P(log, f"GUI Live Mode=True; data root={gui_io.GUI_BASE_PATH}\n")
        else:
            P(log, f"quicklook={quicklook}; data directory={data_paths.describe(datapath, quicklook)}\n")  # LLM-generated
        P(log, f"{'mode':9s} {'ITIME':>8s} {'pairs':>6s} {'reads':>6s} {'maxDN':>7s} "
               f"{'g_raw':>7s} {'alpha_x':>8s} {'alpha_y':>8s} {'g_kern':>7s} {'g_csum':>7s} "
               f"{'g_ipc':>7s} {'err':>6s} {'RN(e-)':>7s} method")

        for mode in modes:
            mode_cfg = config[mode]
            name = mode_cfg["name"]
            itime = mode_cfg["frametime"]
            detrend = mode_cfg.get("detrend", False)
            fixed_xmax = G.as_optional_float(mode_cfg.get("fixed_xmax", None))
            mode_analysis = mode_cfg.get("analysis", {})
            first_read = G.as_optional_int(
                mode_analysis.get("first_read", analysis_cfg.get("first_read", None)))
            last_read = G.as_optional_int(
                mode_analysis.get("last_read", analysis_cfg.get("last_read", None)))
            ipc_cfg = dict(ipc_global)
            ipc_cfg.update(mode_cfg.get("ipc", {}) or {})

            ramps = G.parse_ramps(mode_cfg, section="illuminated", gui=gui)
            pairs = G.make_pairs(mode_cfg, ramps, section="illuminated", gui=gui)
            if not ramps:
                P(log, f"{name:9s} -- no illuminated ramps configured; skipped --")
                continue
            if not pairs:
                P(log, f"{name:9s} -- need at least 2 illuminated ramps; skipped --")
                continue

            read_text = (f"reads={first_read if first_read is not None else 2}:"
                         f"{last_read if last_read is not None else 'end'} (inclusive)")
            P(log, f"   {name}: illuminated ramps={ramps}; pairs={pairs}; {read_text}")

            try:
                r = ipc_many_pairs(head, date, pairs, datapath, X0, X1, Y0, Y1, log, ipc_cfg,
                                   detrend=detrend, fixed_xmax=fixed_xmax, quicklook=quicklook,
                                   gui=gui, first_read=first_read, last_read=last_read)
            except Exception as exc:
                P(log, f"{name:9s} -- gain fit failed: {exc}; skipped --")
                continue

            r["itime"] = itime
            results[name] = r
            nreads = min(pr["nreads_used"] for pr in r["pair_results"])
            P(log, f"{name:9s} {itime:8.4f} {len(r['pair_results']):6d} {nreads:6d} "
                   f"{r['smax']:7.0f} {r['raw_gain']:7.3f} {r['alpha_x']:8.4f} "
                   f"{r['alpha_y']:8.4f} {r['gain_kernel']:7.3f} {r['gain_covsum']:7.3f} "
                   f"{r['gain_ipc']:7.3f} {r['gain_ipc_err']:6.3f} {r['rn_e_ipc']:7.1f} "
                   f"{r['ipc_method']}" + ("   [row-detrended]" if detrend else ""))
            P(log, f"          sum(k^2)={r['sum_k2']:.4f}  C(1,1)/C(0,0) slope ratio="
                   f"{r['r11']:.4f}  raw RN={r['raw_rn_e']:.1f} e-")

            save_ptc_csv(outpath, setname, name, r)
            save_pair_summary(outpath, setname, name, r["pair_results"])

        plot_results(outpath, setname, results)


if __name__ == "__main__":
    main()

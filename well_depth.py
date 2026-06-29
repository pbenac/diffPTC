#!/usr/bin/env python3
"""
Well-depth and ramp-profile analysis for diffPTC-style YAML configs.

Example usage:
    python well_depth.py config.yaml

This script uses the same dataset/mode structure as gain.py.  For each mode in
config["dataset"]["modes"], it loads the configured illuminated ramps, measures
accumulated signal in a chosen ROI as a function of frame number/time, estimates
where the ramp deviates from an initial linear response, and saves ramp plots.

Quicklook support:
    dataset:
      quicklook: true

uses files in datapath/ql_redux/ with names like ss260625_21740_reads.fits.
With quicklook: false or omitted, it uses datapath directly and names like
ss260625_21740.fits.

Optional configuration:
    well_depth:
      roi:                 # optional; falls back to top-level roi
        x_start: 833
        x_end: 1020
        y_start: 1284
        y_end: 1500
      reference_read_index: 1      # zero-indexed frame subtracted from all reads
      slope_start_frame: 2         # one-indexed plotted frame number
      slope_stop_frame: 4          # one-indexed plotted frame number
      deviation_threshold: 0.10    # fractional deviation from initial linear fit
      slope_cutoff: 0.05           # sustained local slope < this*initial_slope -> well depth
      min_signal_fraction: 0.25    # ignore cutoff tests below this fraction of ramp max
      consecutive_points: 5        # require sustained hits before reporting WD
      interpolation_points: 2000
      statistic: median            # median or mean across ROI
      errorbar: sem                # sem, std, or none
      ylim_max_multiplier: 1.08
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
import yaml


def P(log, *args):
    s = " ".join(str(x) for x in args)
    print(s)
    log.write(s + "\n")
    log.flush()


def prefix_for_head(head):
    return "ss" if str(head).upper() == "IFS" else "si"


def truthy(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def effective_datapath(datapath, quicklook=False):
    return os.path.join(datapath, "ql_redux") if quicklook else datapath


def obs_filename(head, date, filenum, quicklook=False):
    suffix = "_reads" if quicklook else ""
    return f"{prefix_for_head(head)}{date}_{int(filenum):05d}{suffix}.fits"


def fits_path(datapath, head, date, filenum, quicklook=False):
    return os.path.join(effective_datapath(datapath, quicklook), obs_filename(head, date, filenum, quicklook))


def parse_ramps(mode_cfg, section="illuminated"):
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

    if "n1" in ramp_cfg and "n2" in ramp_cfg:
        return [int(ramp_cfg["n1"]), int(ramp_cfg["n2"])]

    return []


def get_gain(mode_cfg):
    for key in ("gain", "conversion_gain", "gain_e_per_dn"):
        if key in mode_cfg and mode_cfg[key] is not None:
            return float(mode_cfg[key])
    return None


def get_roi(config):
    wd_cfg = config.get("well_depth", {})
    roi_cfg = wd_cfg.get("roi", config.get("well_depth_roi", config.get("roi")))
    if roi_cfg is None:
        raise KeyError("No ROI configured. Provide top-level roi or well_depth.roi.")
    return (
        int(roi_cfg["x_start"]),
        int(roi_cfg["x_end"]),
        int(roi_cfg["y_start"]),
        int(roi_cfg["y_end"]),
    )


def load_region_and_header(head, date, n, datapath, X0, X1, Y0, Y1, quicklook=False):
    fn = fits_path(datapath, head, date, n, quicklook=quicklook)
    with fits.open(fn, memmap=True, do_not_scale_image_data=True) as h:
        bz = h[0].header.get("BZERO", 0)
        cube = np.asarray(h[0].data[:, Y0:Y1, X0:X1]).astype(np.float64) + bz
        header = dict(h[0].header)
    return cube, header, fn


def region_stat(cube, statistic="median"):
    flat = cube.reshape(cube.shape[0], -1)
    statistic = str(statistic).lower()
    if statistic == "mean":
        return np.mean(flat, axis=1)
    if statistic == "median":
        return np.median(flat, axis=1)
    raise ValueError("well_depth.statistic must be 'median' or 'mean'.")


def frame_time_from_header_or_config(header, mode_cfg):
    for key in ("ITIME", "itime", "FRAMTIME", "FRMTIME"):
        if key in header and header[key] is not None:
            try:
                return float(header[key])
            except Exception:
                pass
    return float(mode_cfg["frametime"])




def first_sustained_hit(mask, n_consecutive):
    """Return first index where mask is True for n_consecutive samples, else None."""
    mask = np.asarray(mask, dtype=bool)
    n_consecutive = max(1, int(n_consecutive))
    if n_consecutive == 1:
        hits = np.where(mask)[0]
        return int(hits[0]) if len(hits) else None
    if len(mask) < n_consecutive:
        return None
    run = np.convolve(mask.astype(int), np.ones(n_consecutive, dtype=int), mode="valid")
    hits = np.where(run >= n_consecutive)[0]
    return int(hits[0]) if len(hits) else None


def local_slopes_per_frame(x, y):
    """Robust local slope estimate dy/dx on an evenly or unevenly spaced grid."""
    return np.gradient(y, x)

def analyze_mode(config, mode_key, X0, X1, Y0, Y1, log):
    dataset = config["dataset"]
    mode_cfg = config[mode_key]
    wd_cfg = config.get("well_depth", {})

    head = dataset["head"]
    date = str(dataset["date"])
    datapath = dataset["datapath"]
    quicklook = truthy(dataset.get("quicklook", False))

    name = mode_cfg["name"]
    ramps = parse_ramps(mode_cfg, section="illuminated")
    if not ramps:
        raise ValueError(f"{name}: no illuminated ramps configured")

    reference_read_index = int(wd_cfg.get("reference_read_index", 1))
    slope_start_frame = int(wd_cfg.get("slope_start_frame", 2))
    slope_stop_frame = int(wd_cfg.get("slope_stop_frame", 4))
    deviation_threshold = float(wd_cfg.get("deviation_threshold", 0.10))
    slope_cutoff = float(wd_cfg.get("slope_cutoff", wd_cfg.get("wd_cutoff", 0.05)))
    min_signal_fraction = float(wd_cfg.get("min_signal_fraction", 0.25))
    consecutive_points = int(wd_cfg.get("consecutive_points", wd_cfg.get("sustained_points", 5)))
    interpolation_points = int(wd_cfg.get("interpolation_points", 2000))
    statistic = wd_cfg.get("statistic", "median")
    errorbar = str(wd_cfg.get("errorbar", "sem")).lower()

    ramp_profiles = []
    headers = []
    loaded_files = []

    for n in ramps:
        cube, header, fn = load_region_and_header(head, date, n, datapath, X0, X1, Y0, Y1, quicklook=quicklook)
        if reference_read_index >= cube.shape[0]:
            raise ValueError(
                f"{name}: reference_read_index={reference_read_index} is outside ramp length {cube.shape[0]}"
            )
        prof = region_stat(cube, statistic=statistic)
        prof = prof - prof[reference_read_index]
        ramp_profiles.append(prof)
        headers.append(header)
        loaded_files.append(fn)

    min_len = min(len(p) for p in ramp_profiles)
    profiles = np.asarray([p[:min_len] for p in ramp_profiles])
    frame_numbers = np.arange(1, min_len + 1, dtype=float)

    signal = np.median(profiles, axis=0)
    scatter = np.std(profiles, axis=0, ddof=1) if len(profiles) > 1 else np.zeros_like(signal)
    if errorbar == "sem" and len(profiles) > 1:
        yerr = scatter / np.sqrt(len(profiles))
    elif errorbar == "std":
        yerr = scatter
    else:
        yerr = None

    itime = frame_time_from_header_or_config(headers[0], mode_cfg)
    time_sec = frame_numbers * itime

    # Fit a simple early linear response using one-indexed frame numbers.
    i0 = max(0, slope_start_frame - 1)
    i1 = min(min_len - 1, slope_stop_frame - 1)
    if i1 <= i0:
        raise ValueError(f"{name}: slope_stop_frame must be after slope_start_frame")

    slope_frame = (signal[i1] - signal[i0]) / (frame_numbers[i1] - frame_numbers[i0])
    slope_dn_s = slope_frame / itime
    intercept = signal[i0] - slope_frame * frame_numbers[i0]
    line = slope_frame * frame_numbers + intercept

    # Directional nonlinearity / rolloff relative to the early linear model.
    # For well-depth purposes, only signal falling BELOW the linear extrapolation
    # should count. Signal above the line is not saturation-like rolloff.
    denom = np.where(np.abs(line) > 1e-12, line, np.nan)
    fractional_rolloff = 1.0 - signal / denom
    deviation = fractional_rolloff  # kept under this legacy name for CSV compatibility

    x_interp = np.linspace(frame_numbers[0], frame_numbers[-1], interpolation_points)
    signal_interp = np.interp(x_interp, frame_numbers, signal)
    line_interp = slope_frame * x_interp + intercept
    denom_interp = np.where(np.abs(line_interp) > 1e-12, line_interp, np.nan)
    rolloff_interp = 1.0 - signal_interp / denom_interp

    # Ignore very low-signal parts of the ramp. Otherwise small point-to-point
    # noise near zero can falsely look like nonlinearity or local flattening.
    max_signal = float(np.nanmax(signal_interp)) if len(signal_interp) else np.nan
    above_min_signal = signal_interp >= min_signal_fraction * max_signal
    after_fit = x_interp >= frame_numbers[i1]

    # Deviation threshold: require sustained positive rolloff below the initial
    # linear model, not a single noisy point and not absolute deviation.
    deviation_mask = (
        after_fit
        & above_min_signal
        & np.isfinite(rolloff_interp)
        & (rolloff_interp > deviation_threshold)
    )
    k = first_sustained_hit(deviation_mask, consecutive_points)
    if k is not None:
        dev_frame = float(x_interp[k])
        dev_time = dev_frame * itime
        dev_dn = float(signal_interp[k])
    else:
        dev_frame = dev_time = dev_dn = np.nan

    # Slope cutoff: require sustained flattening. This prevents one noisy
    # derivative estimate from producing a bogus very-low well-depth value.
    local_slope = local_slopes_per_frame(x_interp, signal_interp)
    if slope_frame >= 0:
        slope_mask = after_fit & above_min_signal & (local_slope < slope_cutoff * slope_frame)
    else:
        slope_mask = after_fit & above_min_signal & (local_slope > slope_cutoff * slope_frame)
    k = first_sustained_hit(slope_mask, consecutive_points)
    if k is not None:
        cutoff_frame = float(x_interp[k])
        cutoff_time = cutoff_frame * itime
        cutoff_dn = float(signal_interp[k])
    else:
        cutoff_frame = cutoff_time = cutoff_dn = np.nan

    gain = get_gain(mode_cfg)

    return {
        "mode_key": mode_key,
        "name": name,
        "ramps": ramps,
        "files": loaded_files,
        "profiles": profiles,
        "frame_numbers": frame_numbers,
        "time_sec": time_sec,
        "signal_dn": signal,
        "scatter_dn": scatter,
        "yerr_dn": yerr,
        "line_dn": line,
        "deviation": deviation,
        "itime": itime,
        "gain": gain,
        "slope_frame_dn": slope_frame,
        "slope_dn_s": slope_dn_s,
        "deviation_threshold": deviation_threshold,
        "well_depth_deviation_dn": dev_dn,
        "time_deviation_s": dev_time,
        "frame_deviation": dev_frame,
        "slope_cutoff": slope_cutoff,
        "min_signal_fraction": min_signal_fraction,
        "consecutive_points": consecutive_points,
        "well_depth_cutoff_dn": cutoff_dn,
        "time_cutoff_s": cutoff_time,
        "frame_cutoff": cutoff_frame,
        "reference_read_index": reference_read_index,
        "slope_start_frame": slope_start_frame,
        "slope_stop_frame": slope_stop_frame,
    }


def save_profile_csv(outpath, setname, result):
    name = safe_name(result["name"])
    cols = [
        result["frame_numbers"],
        result["time_sec"],
        result["signal_dn"],
        result["scatter_dn"],
        result["deviation"],
        result["line_dn"],
    ]
    header = "frame_number,time_s,accumulated_signal_DN,ramp_to_ramp_std_DN,fractional_rolloff,initial_linear_model_DN"
    np.savetxt(
        os.path.join(outpath, f"{setname}_ramp_profile_{name}.csv"),
        np.column_stack(cols),
        delimiter=",",
        header=header,
        comments="",
    )


def safe_name(text):
    return str(text).replace("/", "-").replace(" ", "_")


def plot_ramp(outpath, setname, result, ylim_max_multiplier=1.08, xticks=None):
    fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")

    yerr = result["yerr_dn"]
    if yerr is not None:
        ax.errorbar(result["frame_numbers"], result["signal_dn"], yerr=yerr, fmt="o", ms=3, capsize=2, label=f"Median of {len(result['ramps'])} ramps")
    else:
        ax.plot(result["frame_numbers"], result["signal_dn"], "o", ms=3, label=f"Median of {len(result['ramps'])} ramps")

    ax.plot(
        result["frame_numbers"],
        result["line_dn"],
        "--",
        lw=1.5,
        label=f"Linear fit frames {result['slope_start_frame']}–{result['slope_stop_frame']}",
    )

    if np.isfinite(result["well_depth_deviation_dn"]):
        ax.axhline(result["well_depth_deviation_dn"], color="0.35", ls=":", lw=1)
        ax.axvline(result["frame_deviation"], color="0.35", ls=":", lw=1)
        ax.annotate(
            f"{result['deviation_threshold']:.0%} deviation\n{result['well_depth_deviation_dn']:.0f} DN",
            xy=(result["frame_deviation"], result["well_depth_deviation_dn"]),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=9,
        )

    if np.isfinite(result["well_depth_cutoff_dn"]):
        ax.axhline(result["well_depth_cutoff_dn"], color="0.55", ls="--", lw=1)
        ax.axvline(result["frame_cutoff"], color="0.55", ls="--", lw=1)

    ymin = min(0, float(np.nanmin(result["signal_dn"])))
    ymax = float(np.nanmax(result["signal_dn"])) * ylim_max_multiplier
    if ymax <= ymin:
        ymax = ymin + 1
    ax.set_ylim(ymin, ymax)

    if xticks:
        ax.set_xticks(xticks)

    ax.set_xlabel("Frame number")
    ax.set_ylabel("Accumulated signal [DN]")
    ax.set_title(f"{result['name']} ramp profile")
    ax.legend(fontsize=9)

    itime = result["itime"]

    def frame_to_time(x):
        return x * itime

    def time_to_frame(x):
        return x / itime

    secax = ax.secondary_xaxis("top", functions=(frame_to_time, time_to_frame))
    secax.set_xlabel("Time [s]")

    fig.savefig(os.path.join(outpath, f"{setname}_ramp_{safe_name(result['name'])}.png"), dpi=140)
    plt.close(fig)


def save_summary_csv(outpath, setname, results):
    rows = []
    for r in results:
        gain = r["gain"]
        dev_e = r["well_depth_deviation_dn"] * gain if gain is not None and np.isfinite(r["well_depth_deviation_dn"]) else np.nan
        cutoff_e = r["well_depth_cutoff_dn"] * gain if gain is not None and np.isfinite(r["well_depth_cutoff_dn"]) else np.nan
        rows.append([
            r["name"],
            len(r["ramps"]),
            len(r["frame_numbers"]),
            r["itime"],
            r["slope_frame_dn"],
            r["slope_dn_s"],
            r["deviation_threshold"],
            r["well_depth_deviation_dn"],
            dev_e,
            r["frame_deviation"],
            r["time_deviation_s"],
            r["slope_cutoff"],
            r["min_signal_fraction"],
            r["consecutive_points"],
            r["well_depth_cutoff_dn"],
            cutoff_e,
            r["frame_cutoff"],
            r["time_cutoff_s"],
        ])

    # Use object/string writing because first column is text.
    fn = os.path.join(outpath, f"{setname}_well_depth_summary.csv")
    header = (
        "mode,n_ramps,n_reads,frame_time_s,initial_slope_DN_per_frame,initial_slope_DN_per_s,"
        "deviation_threshold,well_depth_deviation_DN,well_depth_deviation_e,frame_deviation,time_deviation_s,"
        "slope_cutoff_fraction,min_signal_fraction,consecutive_points,well_depth_slope_cutoff_DN,well_depth_slope_cutoff_e,frame_slope_cutoff,time_slope_cutoff_s"
    )
    with open(fn, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")


def main():
    if len(sys.argv) < 2:
        print("Error: No yaml configuration file provided")
        sys.exit(1)

    yamlfile = sys.argv[1]
    with open(yamlfile) as f:
        config = yaml.safe_load(f)

    dataset = config["dataset"]
    outpath = dataset["outpath"]
    setname = dataset["name"]
    modes = dataset["modes"]
    quicklook = truthy(dataset.get("quicklook", False))

    os.makedirs(outpath, exist_ok=True)
    X0, X1, Y0, Y1 = get_roi(config)

    wd_cfg = config.get("well_depth", {})
    ylim_max_multiplier = float(wd_cfg.get("ylim_max_multiplier", 1.08))
    xticks = wd_cfg.get("xticks", None)

    logname = f"{setname}_well_depth.txt"
    results = []

    with open(os.path.join(outpath, logname), "w") as log:
        P(log, "=== Well depth / ramp profile analysis ===")
        P(log, f"Dataset: {setname}")
        P(log, f"Quicklook: {quicklook}")
        P(log, f"ROI: x[{X0}:{X1}] y[{Y0}:{Y1}]")
        P(log, "Well-depth flags require sustained positive rolloff/flattening; no flag => NaN.\n")
        P(log, f"{'mode':20s} {'ramps':>5s} {'reads':>5s} {'slope(DN/s)':>12s} {'WD rolloff(DN)':>14s} {'WD cutoff(DN)':>14s}")

        for mode_key in modes:
            try:
                r = analyze_mode(config, mode_key, X0, X1, Y0, Y1, log)
            except Exception as exc:
                P(log, f"{mode_key:20s} -- failed: {exc}")
                continue

            results.append(r)
            P(
                log,
                f"{r['name']:20s} {len(r['ramps']):5d} {len(r['frame_numbers']):5d} "
                f"{r['slope_dn_s']:12.3f} {r['well_depth_deviation_dn']:12.1f} {r['well_depth_cutoff_dn']:14.1f}",
            )
            save_profile_csv(outpath, setname, r)
            plot_ramp(outpath, setname, r, ylim_max_multiplier=ylim_max_multiplier, xticks=xticks)

    if results:
        save_summary_csv(outpath, setname, results)

    print("DONE")


if __name__ == "__main__":
    main()

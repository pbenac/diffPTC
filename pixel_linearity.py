#!/usr/bin/env python3
"""
Per-pixel linearity diagnostics for diffPTC-style YAML configs.

Analyzes one individual illuminated ramp per mode and measures, for every pixel
in the configured ROI, where that pixel's accumulated signal falls 5% and 10%
below an early linear extrapolation.

Outputs:
  1) Diagnostic Frame Number vs Signal plot for a small set of representative
     pixels chosen to span a range of nonlinearity behaviors.
  2) Histograms of the 5% and 10% nonlinearity frame numbers for all valid
     pixels in the ROI.
  3) CSV with per-pixel 5% and 10% crossing measurements.

Usage:
    python3 pixel_linearity.py config.yaml

YAML example:

pixel_linearity:
  section: illuminated
  ramp: 12345                 # optional; otherwise first configured ramp is used
  reference_read_index: 1     # zero-indexed array index
  slope_start_frame: 3        # one-indexed plotted frame number
  slope_stop_frame: 6         # one-indexed plotted frame number
  consecutive_points: 2
  min_signal_fraction: 0.10
  diagnostic_pixels: 6
  include_pre_read_signal: true
  pre_read_frames: 1.0
  hist_bins: 40

The script uses the same dataset/mode structure and FITS naming conventions as
well_depth.py and gain.py.
"""

import csv
import os
import sys

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.io import fits


def P(log, *args):
    s = " ".join(str(x) for x in args)
    print(s)
    log.write(s + "\n")
    log.flush()


def truthy(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def prefix_for_head(head):
    return "ss" if str(head).upper() == "IFS" else "si"


def effective_datapath(datapath, quicklook=False):
    return os.path.join(datapath, "ql_redux") if quicklook else datapath


def obs_filename(head, date, filenum, quicklook=False):
    suffix = "_reads" if quicklook else ""
    return f"{prefix_for_head(head)}{date}_{int(filenum):05d}{suffix}.fits"


def fits_path(datapath, head, date, filenum, quicklook=False):
    return os.path.join(
        effective_datapath(datapath, quicklook),
        obs_filename(head, date, filenum, quicklook),
    )


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


def load_region_and_header(head, date, n, datapath, X0, X1, Y0, Y1, quicklook=False):
    fn = fits_path(datapath, head, date, n, quicklook=quicklook)
    with fits.open(fn, memmap=True, do_not_scale_image_data=True) as h:
        bz = h[0].header.get("BZERO", 0)
        cube = np.asarray(h[0].data[:, Y0:Y1, X0:X1]).astype(np.float64) + bz
        header = h[0].header.copy()
    return cube, header, fn


def frame_time_from_header_or_config(header, mode_cfg):
    for key in ("ITIME", "itime", "FRAMTIME", "FRMTIME"):
        if key in header and header[key] is not None:
            try:
                return float(header[key])
            except Exception:
                pass
    return float(mode_cfg["frametime"])


def first_sustained_hit(mask, n_consecutive):
    mask = np.asarray(mask, dtype=bool)
    n_consecutive = max(1, int(n_consecutive))
    if n_consecutive == 1:
        hits = np.flatnonzero(mask)
        return int(hits[0]) if len(hits) else None
    if len(mask) < n_consecutive:
        return None
    run = np.convolve(mask.astype(int), np.ones(n_consecutive, dtype=int), mode="valid")
    hits = np.flatnonzero(run >= n_consecutive)
    return int(hits[0]) if len(hits) else None


def interpolate_crossing(frame_numbers, deviation, signal, threshold, k):
    if k is None:
        return np.nan, np.nan
    if k <= 0:
        return float(frame_numbers[k]), float(signal[k])
    k0, k1 = k - 1, k
    d0, d1 = deviation[k0], deviation[k1]
    if np.isfinite(d0) and np.isfinite(d1) and d1 != d0:
        frac = float(np.clip((threshold - d0) / (d1 - d0), 0.0, 1.0))
    else:
        frac = 1.0
    frame = frame_numbers[k0] + frac * (frame_numbers[k1] - frame_numbers[k0])
    sig = signal[k0] + frac * (signal[k1] - signal[k0])
    return float(frame), float(sig)


def analyze_pixel_trace(raw_trace, reference_read_index, slope_start_frame,
                        slope_stop_frame, consecutive_points, min_signal_fraction,
                        include_pre_read_signal=False, pre_read_frames=1.0):
    raw_trace = np.asarray(raw_trace, dtype=float)
    nf = len(raw_trace)
    if not (0 <= reference_read_index < nf):
        return None

    frame_numbers = np.arange(1, nf + 1, dtype=float)
    signal = raw_trace - raw_trace[reference_read_index]

    i0 = max(0, int(slope_start_frame) - 1)
    i1 = min(nf - 1, int(slope_stop_frame) - 1)
    if i1 <= i0:
        return None

    fit_x = frame_numbers[i0:i1 + 1]
    fit_y = signal[i0:i1 + 1]
    finite = np.isfinite(fit_x) & np.isfinite(fit_y)
    if finite.sum() < 2:
        return None

    slope, intercept = np.polyfit(fit_x[finite], fit_y[finite], 1)
    if not np.isfinite(slope) or slope <= 0:
        return None

    line = slope * frame_numbers + intercept
    pre_read_dn = slope * float(pre_read_frames) if include_pre_read_signal else 0.0
    signal_total = signal + pre_read_dn
    line_total = line + pre_read_dn

    denom = np.where(line_total > 0, line_total, np.nan)
    deviation = 1.0 - signal_total / denom

    max_signal = np.nanmax(signal_total)
    if not np.isfinite(max_signal) or max_signal <= 0:
        return None

    above_min_signal = signal_total >= float(min_signal_fraction) * max_signal
    after_fit = frame_numbers > frame_numbers[i1]

    result = {
        "frame_numbers": frame_numbers,
        "signal": signal_total,
        "line": line_total,
        "deviation": deviation,
        "slope_dn_per_frame": float(slope),
    }

    for threshold in (0.05, 0.10):
        mask = after_fit & above_min_signal & np.isfinite(deviation) & (deviation >= threshold)
        k = first_sustained_hit(mask, consecutive_points)
        frame, sig = interpolate_crossing(frame_numbers, deviation, signal_total, threshold, k)
        key = int(round(threshold * 100))
        result[f"frame_{key}pct"] = frame
        result[f"signal_{key}pct_dn"] = sig

    return result


def choose_representative_pixels(rows, n_pixels):
    n_pixels = max(1, int(n_pixels))
    source = [r for r in rows if np.isfinite(r["frame_10pct"])]
    sort_key = "frame_10pct"
    if not source:
        source = [r for r in rows if np.isfinite(r["frame_5pct"])]
        sort_key = "frame_5pct"
    if not source:
        source = list(rows)
        sort_key = None
    if not source:
        return []
    if sort_key is not None:
        source = sorted(source, key=lambda r: r[sort_key])
    if len(source) <= n_pixels:
        return source
    idx = np.unique(np.round(np.linspace(0, len(source) - 1, n_pixels)).astype(int))
    return [source[i] for i in idx]


def plot_diagnostic_pixels(outpath, setname, mode_key, mode_name, ramp, selected,
                           traces, slope_start_frame, slope_stop_frame):
    if not selected:
        return None
    fig, ax = plt.subplots(figsize=(10, 7))
    for row in selected:
        r = traces[(row["y"], row["x"])]
        label = (f"(x={row['x']}, y={row['y']}) "
                 f"5%={row['frame_5pct']:.2f}, 10%={row['frame_10pct']:.2f}")
        line_obj, = ax.plot(r["frame_numbers"], r["signal"], marker="o", linewidth=1.2,
                            markersize=3, label=label)
        ax.plot(r["frame_numbers"], r["line"], linestyle="--", linewidth=0.8,
                color=line_obj.get_color(), alpha=0.7)
        if np.isfinite(row["frame_5pct"]):
            ax.scatter([row["frame_5pct"]], [row["signal_5pct_dn"]], marker="s", s=35,
                       color=line_obj.get_color())
        if np.isfinite(row["frame_10pct"]):
            ax.scatter([row["frame_10pct"]], [row["signal_10pct_dn"]], marker="x", s=45,
                       color=line_obj.get_color())
    ax.axvspan(slope_start_frame, slope_stop_frame, alpha=0.10, label="initial linear-fit interval")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Accumulated Signal [DN]")
    ax.set_title(f"{setname} — {mode_name} — ramp {ramp}\nRepresentative pixel linearity traces")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fn = os.path.join(outpath, f"{setname}_{mode_key}_ramp{int(ramp):05d}_pixel_linearity_diagnostic.png")
    fig.savefig(fn, dpi=150)
    plt.close(fig)
    return fn


def plot_histograms(outpath, setname, mode_key, mode_name, ramp, rows, hist_bins):
    vals5 = np.asarray([r["frame_5pct"] for r in rows if np.isfinite(r["frame_5pct"])], dtype=float)
    vals10 = np.asarray([r["frame_10pct"] for r in rows if np.isfinite(r["frame_10pct"])], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 6))
    if len(vals5):
        ax.hist(vals5, bins=hist_bins, histtype="step", linewidth=1.8,
                label=f"5% nonlinearity (N={len(vals5)})")
    if len(vals10):
        ax.hist(vals10, bins=hist_bins, histtype="step", linewidth=1.8,
                label=f"10% nonlinearity (N={len(vals10)})")
    ax.set_xlabel("Frame Number at Nonlinearity Threshold")
    ax.set_ylabel("Number of Pixels")
    ax.set_title(f"{setname} — {mode_name} — ramp {ramp}\nPer-pixel nonlinearity distribution")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fn = os.path.join(outpath, f"{setname}_{mode_key}_ramp{int(ramp):05d}_pixel_linearity_hist.png")
    fig.savefig(fn, dpi=150)
    plt.close(fig)
    return fn


def write_csv(outpath, setname, mode_key, ramp, rows):
    fn = os.path.join(outpath, f"{setname}_{mode_key}_ramp{int(ramp):05d}_pixel_linearity.csv")
    fields = ["x", "y", "slope_dn_per_frame", "frame_5pct", "signal_5pct_dn",
              "frame_10pct", "signal_10pct_dn"]
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fields})
    return fn


def summarize(rows, key):
    vals = np.asarray([r[key] for r in rows if np.isfinite(r[key])], dtype=float)
    if len(vals) == 0:
        return dict(n=0, median=np.nan, p16=np.nan, p84=np.nan, min=np.nan, max=np.nan)
    return dict(n=len(vals), median=float(np.nanmedian(vals)),
                p16=float(np.nanpercentile(vals, 16)), p84=float(np.nanpercentile(vals, 84)),
                min=float(np.nanmin(vals)), max=float(np.nanmax(vals)))


def analyze_mode(config, mode_key, X0, X1, Y0, Y1, log):
    dataset = config["dataset"]
    mode_cfg = config[mode_key]
    pl_cfg = config.get("pixel_linearity", {})
    wd_cfg = config.get("well_depth", {})

    head = dataset["head"]
    date = str(dataset["date"])
    datapath = dataset["datapath"]
    outpath = dataset["outpath"]
    quicklook = truthy(dataset.get("quicklook", False))
    setname = dataset["name"]
    name = mode_cfg["name"]

    section = pl_cfg.get("section", "illuminated")
    ramps = parse_ramps(mode_cfg, section=section)
    if pl_cfg.get("ramp") is not None:
        ramp = int(pl_cfg["ramp"])
    else:
        if not ramps:
            raise ValueError(f"{name}: no ramps configured in section {section!r}")
        ramp = int(ramps[0])

    reference_read_index = int(pl_cfg.get("reference_read_index", wd_cfg.get("reference_read_index", 1)))
    slope_start_frame = int(pl_cfg.get("slope_start_frame", wd_cfg.get("slope_start_frame", 3)))
    slope_stop_frame = int(pl_cfg.get("slope_stop_frame", wd_cfg.get("slope_stop_frame", 6)))
    consecutive_points = int(pl_cfg.get("consecutive_points", wd_cfg.get("consecutive_points", 2)))
    min_signal_fraction = float(pl_cfg.get("min_signal_fraction", wd_cfg.get("min_signal_fraction", 0.10)))
    diagnostic_pixels = int(pl_cfg.get("diagnostic_pixels", 6))
    hist_bins = int(pl_cfg.get("hist_bins", 40))
    include_pre_read_signal = truthy(pl_cfg.get("include_pre_read_signal", wd_cfg.get("include_pre_read_signal", False)))
    pre_read_frames = float(pl_cfg.get("pre_read_frames", wd_cfg.get("pre_read_frames", 1.0)))

    cube, header, fn = load_region_and_header(head, date, ramp, datapath, X0, X1, Y0, Y1, quicklook=quicklook)
    if reference_read_index >= cube.shape[0]:
        raise ValueError(f"{name}: reference_read_index={reference_read_index} outside ramp length {cube.shape[0]}")
    itime = frame_time_from_header_or_config(header, mode_cfg)

    P(log, f"=== {name} ===")
    P(log, f"Ramp: {ramp}")
    P(log, f"File: {fn}")
    P(log, f"ROI: x[{X0}:{X1}] y[{Y0}:{Y1}]")
    P(log, f"Ramp reads: {cube.shape[0]}; ITIME={itime:.6f} s")
    P(log, f"reference_read_index={reference_read_index}; slope frames={slope_start_frame}-{slope_stop_frame}")

    rows = []
    traces = {}
    ny, nx = cube.shape[1], cube.shape[2]
    for iy in range(ny):
        for ix in range(nx):
            r = analyze_pixel_trace(
                cube[:, iy, ix], reference_read_index, slope_start_frame, slope_stop_frame,
                consecutive_points, min_signal_fraction, include_pre_read_signal, pre_read_frames
            )
            if r is None:
                continue
            x, y = X0 + ix, Y0 + iy
            row = {
                "x": int(x), "y": int(y), "slope_dn_per_frame": r["slope_dn_per_frame"],
                "frame_5pct": r["frame_5pct"], "signal_5pct_dn": r["signal_5pct_dn"],
                "frame_10pct": r["frame_10pct"], "signal_10pct_dn": r["signal_10pct_dn"],
            }
            rows.append(row)
            traces[(y, x)] = r

    if not rows:
        raise ValueError(f"{name}: no valid pixels found in ROI")

    selected = choose_representative_pixels(rows, diagnostic_pixels)
    diag_fn = plot_diagnostic_pixels(outpath, setname, mode_key, name, ramp, selected, traces,
                                     slope_start_frame, slope_stop_frame)
    hist_fn = plot_histograms(outpath, setname, mode_key, name, ramp, rows, hist_bins)
    csv_fn = write_csv(outpath, setname, mode_key, ramp, rows)

    s5, s10 = summarize(rows, "frame_5pct"), summarize(rows, "frame_10pct")
    P(log, f"Valid pixels analyzed: {len(rows)}")
    P(log, f"5% frame: N={s5['n']} median={s5['median']:.3f} 16-84%=[{s5['p16']:.3f}, {s5['p84']:.3f}]")
    P(log, f"10% frame: N={s10['n']} median={s10['median']:.3f} 16-84%=[{s10['p16']:.3f}, {s10['p84']:.3f}]")
    P(log, "Diagnostic pixels:")
    for row in selected:
        P(log, f"  x={row['x']} y={row['y']} 5%={row['frame_5pct']:.3f} 10%={row['frame_10pct']:.3f}")
    P(log, f"Diagnostic plot: {diag_fn}")
    P(log, f"Histogram plot: {hist_fn}")
    P(log, f"CSV: {csv_fn}")
    P(log, "")


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
    os.makedirs(outpath, exist_ok=True)

    pl_cfg = config.get("pixel_linearity", {})
    roi_key = pl_cfg.get("use_roi", "roi")
    if roi_key not in config:
        raise KeyError(f"ROI key {roi_key!r} not found in config")

    roi = config[roi_key]
    X0, X1 = int(roi["x_start"]), int(roi["x_end"])
    Y0, Y1 = int(roi["y_start"]), int(roi["y_end"])

    logname = f"{setname}_pixel_linearity.txt"
    with open(os.path.join(outpath, logname), "w") as log:
        P(log, "=== Per-pixel linearity analysis ===")
        P(log, f"YAML: {yamlfile}")
        P(log, f"ROI: x[{X0}:{X1}] y[{Y0}:{Y1}]")
        P(log, "")
        for mode_key in modes:
            analyze_mode(config, mode_key, X0, X1, Y0, Y1, log)


if __name__ == "__main__":
    main()

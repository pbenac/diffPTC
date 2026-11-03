#!/usr/bin/env python3
"""
Estimate detector bias / starting level from the first read of each ramp.

This script is intentionally lightweight and follows the same YAML style used by
other diffPTC scripts in this project.

Typical use:

    python bias_level.py config.yaml

By default it uses the illuminated ramps for each configured readout mode and
computes the median value of the first science read within the configured ROI,
after trimming the brightest and dimmest pixels.

Expected YAML structure:

    dataset:
      name: 260625_IMG_Defaults
      head: IMG
      date: "260625"
      datapath: /data20/CD5/IMG/jun25/
      outpath: /home/pbenac/diffPTC/results/
      quicklook: false
      modes: [slow52, fast10, fast06]

    roi:
      x_start: 50
      x_end: 350
      y_start: 1350
      y_end: 1650

    slow52:
      name: Slow5.2_Default
      illuminated:
        ramps: [21740, 21741, 21742]

Optional configuration:

    bias_level:
      section: illuminated        # ramp section to use; default illuminated
      read_index: 1               # default first science read; use 0 for reset frame
      trim_low_percentile: 1.0    # discard dimmest 1% of pixels
      trim_high_percentile: 99.0  # discard brightest 1% of pixels
      use_roi: roi                # default roi; can point to bias_roi if defined

Outputs:
  - <dataset>_bias_level.txt
  - <dataset>_bias_level_by_ramp.csv
  - <dataset>_bias_level_summary.csv
"""

import os
import sys
import warnings

import numpy as np
warnings.filterwarnings("ignore")

from astropy.io import fits
import yaml
import gui_io


def P(log, *args):
    """Print to terminal and write the same line to the log file."""
    s = " ".join(str(x) for x in args)
    print(s)
    log.write(s + "\n")
    log.flush()


def prefix_for_head(head):
    """Return the file prefix for the detector head."""
    return "ss" if str(head).upper() == "IFS" else "si"


def dataset_datapath(dataset):
    """Return the effective data path, including ql_redux for quicklook files."""
    datapath = dataset["datapath"]
    quicklook = bool(dataset.get("quicklook", False))
    if quicklook:
        return os.path.join(datapath, "ql_redux")
    return datapath


def obs_filename(head, date, filenum, quicklook=False):
    """Build the FITS filename for normal or quicklook-style data."""
    pref = prefix_for_head(head)
    obsnum = f"{int(filenum):05d}"
    suffix = "_reads" if quicklook else ""
    return f"{pref}{date}_{obsnum}{suffix}.fits"


def fits_path(dataset, filenum):
    """Build the full path to one ramp FITS file."""
    quicklook = bool(dataset.get("quicklook", False))
    filename = obs_filename(dataset["head"], dataset["date"], filenum, quicklook=quicklook)
    return os.path.join(dataset_datapath(dataset), filename)


def parse_ramps(mode_cfg, section="illuminated", gui=False):
    """Return ramp numbers from a mode config.

    Supported forms:

        illuminated:
          ramps: [21740, 21741]

        illuminated:
          ramp_range: [21740, 21745]   # stop-exclusive, like range()

        illuminated:
          n1: 21740
          n2: 21741
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

    if "n1" in ramp_cfg and "n2" in ramp_cfg:
        return [int(ramp_cfg["n1"]), int(ramp_cfg["n2"])]

    return []


def load_read_region(dataset, filenum, read_index, X0, X1, Y0, Y1, gui=False):
    """Load one read from one ramp/acquisition, cropped to the configured ROI."""
    if gui:
        im, _header, fn = gui_io.load_gui_frame(
            filenum, read_index, roi=(X0, X1, Y0, Y1)
        )
        return im, fn

    fn = fits_path(dataset, filenum)
    with fits.open(fn, memmap=True, do_not_scale_image_data=True) as h:
        bz = h[0].header.get("BZERO", 0)
        data = h[0].data
        if read_index < 0 or read_index >= data.shape[0]:
            raise IndexError(
                f"read_index={read_index} is outside ramp with {data.shape[0]} reads: {fn}"
            )
        im = np.asarray(data[read_index, Y0:Y1, X0:X1]).astype(np.float64) + bz
    return im, fn


def trimmed_median(image, low_pct=1.0, high_pct=99.0):
    """Compute median after discarding brightest and dimmest pixels.

    low_pct and high_pct define the included percentile range. For example,
    1 and 99 keeps the central 98% of pixels before taking the median.
    """
    arr = np.asarray(image, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")

    lo = np.percentile(arr, low_pct)
    hi = np.percentile(arr, high_pct)
    keep = (arr >= lo) & (arr <= hi)
    if keep.sum() == 0:
        return float("nan")
    return float(np.median(arr[keep]))


def summarize(values):
    """Return robust summary statistics for a list of per-ramp values."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return dict(n=0, median=np.nan, mean=np.nan, std=np.nan, min=np.nan, max=np.nan)
    return dict(
        n=int(arr.size),
        median=float(np.median(arr)),
        mean=float(np.mean(arr)),
        std=float(np.std(arr, ddof=1)) if arr.size > 1 else np.nan,
        min=float(np.min(arr)),
        max=float(np.max(arr)),
    )


def analyze_mode(dataset, mode_key, mode_cfg, X0, X1, Y0, Y1, section, read_index, low_pct, high_pct, log, gui=False):
    """Measure trimmed median first-read level for every ramp in one mode."""
    name = mode_cfg.get("name", mode_key)
    ramps = parse_ramps(mode_cfg, section=section, gui=gui)

    if not ramps:
        P(log, f"{name:20s} -- no ramps found in section {section!r}; skipped --")
        return [], None

    rows = []
    values = []
    for n in ramps:
        try:
            image, fn = load_read_region(dataset, n, read_index, X0, X1, Y0, Y1, gui=gui)
            bias_dn = trimmed_median(image, low_pct=low_pct, high_pct=high_pct)
            npix = int(np.isfinite(image).sum())
            rows.append(dict(
                mode_key=mode_key,
                mode_name=name,
                ramp=int(n),
                read_index=int(read_index),
                bias_dn=bias_dn,
                npix=npix,
                filename=fn,
            ))
            values.append(bias_dn)
        except Exception as exc:
            P(log, f"   {name}: ramp {n} failed: {exc}")

    stats = summarize(values)
    if stats["n"] > 0:
        P(
            log,
            f"{name:20s} {stats['n']:5d} ramps  "
            f"median={stats['median']:10.2f} DN  "
            f"mean={stats['mean']:10.2f} DN  "
            f"std={stats['std']:8.2f} DN  "
            f"range=[{stats['min']:.2f}, {stats['max']:.2f}] DN",
        )
    else:
        P(log, f"{name:20s} -- no valid ramps measured --")

    summary = dict(
        mode_key=mode_key,
        mode_name=name,
        n_ramps=stats["n"],
        read_index=read_index,
        trim_low_percentile=low_pct,
        trim_high_percentile=high_pct,
        median_bias_dn=stats["median"],
        mean_bias_dn=stats["mean"],
        std_bias_dn=stats["std"],
        min_bias_dn=stats["min"],
        max_bias_dn=stats["max"],
    )
    return rows, summary


def write_ramp_csv(outpath, setname, rows):
    out = os.path.join(outpath, f"{setname}_bias_level_by_ramp.csv")
    header = "mode_key,mode_name,ramp,read_index,bias_DN,npix,filename"
    with open(out, "w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(
                f"{r['mode_key']},{r['mode_name']},{r['ramp']},{r['read_index']},"
                f"{r['bias_dn']:.8g},{r['npix']},{r['filename']}\n"
            )
    return out


def write_summary_csv(outpath, setname, summaries):
    out = os.path.join(outpath, f"{setname}_bias_level_summary.csv")
    header = (
        "mode_key,mode_name,n_ramps,read_index,trim_low_percentile,"
        "trim_high_percentile,median_bias_DN,mean_bias_DN,std_bias_DN,min_bias_DN,max_bias_DN"
    )
    with open(out, "w") as f:
        f.write(header + "\n")
        for s in summaries:
            f.write(
                f"{s['mode_key']},{s['mode_name']},{s['n_ramps']},{s['read_index']},"
                f"{s['trim_low_percentile']},{s['trim_high_percentile']},"
                f"{s['median_bias_dn']:.8g},{s['mean_bias_dn']:.8g},{s['std_bias_dn']:.8g},"
                f"{s['min_bias_dn']:.8g},{s['max_bias_dn']:.8g}\n"
            )
    return out


def main():
    if len(sys.argv) < 2:
        print("Error: No yaml configuration file provided")
        sys.exit(1)

    yamlfile = sys.argv[1]
    with open(yamlfile) as f:
        config = yaml.safe_load(f)

    dataset = config["dataset"]
    gui = gui_io.is_gui_mode(config)
    setname = dataset["name"]
    outpath = dataset["outpath"]
    modes = dataset["modes"]
    os.makedirs(outpath, exist_ok=True)

    bias_cfg = config.get("bias_level", {})
    section = bias_cfg.get("section", "illuminated")
    read_index = int(bias_cfg.get("read_index", 1))
    low_pct = float(bias_cfg.get("trim_low_percentile", 1.0))
    high_pct = float(bias_cfg.get("trim_high_percentile", 99.0))
    roi_key = bias_cfg.get("use_roi", "roi")

    if roi_key not in config:
        raise KeyError(f"ROI key {roi_key!r} not found in config. Try use_roi: roi or define {roi_key}.")

    roi = config[roi_key]
    X0 = int(roi["x_start"])
    X1 = int(roi["x_end"])
    Y0 = int(roi["y_start"])
    Y1 = int(roi["y_end"])

    if not (0 <= low_pct < high_pct <= 100):
        raise ValueError("Require 0 <= trim_low_percentile < trim_high_percentile <= 100")

    logname = f"{setname}_bias_level.txt"
    all_rows = []
    summaries = []

    with open(os.path.join(outpath, logname), "w") as log:
        P(log, "=== Bias / starting-level estimate ===")
        P(log, f"Dataset: {setname}")
        P(log, f"GUI Live Mode: {gui}")
        if gui:
            P(log, f"GUI data root: {gui_io.GUI_BASE_PATH}")
        else:
            P(log, f"Quicklook: {bool(dataset.get('quicklook', False))}")
            P(log, f"Data path: {dataset_datapath(dataset)}")
        P(log, f"Ramp section: {section}")
        P(log, f"Read index: {read_index}")
        P(log, f"ROI: {roi_key} x[{X0}:{X1}] y[{Y0}:{Y1}]")
        P(log, f"Trim: keep pixels between {low_pct:g} and {high_pct:g} percentiles\n")
        P(log, f"{'mode':20s} {'N':>5s}   {'median bias':>18s} {'mean bias':>15s} {'std':>12s} {'range':>24s}")

        for mode_key in modes:
            mode_cfg = config[mode_key]
            rows, summary = analyze_mode(
                dataset, mode_key, mode_cfg, X0, X1, Y0, Y1,
                section, read_index, low_pct, high_pct, log, gui=gui,
            )
            all_rows.extend(rows)
            if summary is not None:
                summaries.append(summary)

        ramp_csv = write_ramp_csv(outpath, setname, all_rows)
        summary_csv = write_summary_csv(outpath, setname, summaries)
        P(log, "")
        P(log, f"Wrote per-ramp CSV: {ramp_csv}")
        P(log, f"Wrote summary CSV:  {summary_csv}")

    print("DONE")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Plot pixel-value traces versus frame/read number for detector ramps.

Usage:
    python3 plot_pix_trace.py config.yaml

This script follows the same YAML-driven dataset/mode conventions used by
other detector-analysis scripts in this project. For each configured mode, it
loads ramps from a selected section (default: illuminated), optionally crops to
an ROI, optionally subtracts a reference read, subsamples pixels by
pixel_stride, and writes one PNG per ramp.
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
import gui_io


def P(log, *args):
    """Print to terminal and write the same line to the log file."""
    s = " ".join(str(x) for x in args)
    print(s)
    log.write(s + "\n")
    log.flush()


def truthy(value):
    """Robustly interpret common YAML/string boolean values."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def prefix_for_head(head):
    return "ss" if str(head).upper() == "IFS" else "si"


def data_directory(datapath, quicklook=False):
    return os.path.join(datapath, "ql_redux") if quicklook else datapath


def obs_filename(head, date, filenum, quicklook=False):
    suffix = "_reads" if quicklook else ""
    return f"{prefix_for_head(head)}{str(date)}_{int(filenum):05d}{suffix}.fits"


def fits_path(datapath, head, date, filenum, quicklook=False):
    return os.path.join(
        data_directory(datapath, quicklook),
        obs_filename(head, date, filenum, quicklook=quicklook),
    )


def parse_ramps(mode_cfg, section="illuminated", gui=False):
    """Return ramp numbers from a mode configuration."""
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


def get_roi(config, use_roi):
    """Return (X0, X1, Y0, Y1), or None for full-frame loading."""
    if use_roi is None:
        return None

    if isinstance(use_roi, str):
        key = use_roi.strip()
        if key.lower() in {"", "none", "null", "full", "full_frame", "fullframe"}:
            return None
        if key not in config:
            raise KeyError(
                f"plot_pix_trace.use_roi={key!r}, but no top-level YAML block "
                f"named {key!r} exists."
            )
        roi_cfg = config[key]
    elif isinstance(use_roi, dict):
        roi_cfg = use_roi
    else:
        raise TypeError(
            "plot_pix_trace.use_roi must be a top-level ROI block name, "
            "an ROI mapping, or null."
        )

    return (
        int(roi_cfg["x_start"]),
        int(roi_cfg["x_end"]),
        int(roi_cfg["y_start"]),
        int(roi_cfg["y_end"]),
    )


def load_cube(datapath, head, date, filenum, quicklook=False, roi=None, gui=False,  # LLM-generated: main() passes gui
              pixel_stride=1):  # LLM-generated: load only the pixels that will be plotted
    """Load one ramp cube as float64 and apply FITS BZERO manually."""
    stride = int(pixel_stride)  # LLM-generated
    if stride < 1:  # LLM-generated: same error plot_all_pixel_traces raises
        raise ValueError("pixel_stride must be >= 1")  # LLM-generated

    if gui:  # LLM-generated
        data, _header, source = gui_io.load_gui_cube(filenum, roi=roi)  # LLM-generated
        return data[:, ::stride, ::stride].copy(), source  # LLM-generated: GUI mode still reads every pixel

    fn = fits_path(datapath, head, date, filenum, quicklook=quicklook)

    with fits.open(fn, memmap=True, do_not_scale_image_data=True) as h:
        bz = h[0].header.get("BZERO", 0)
        if roi is None:
            data = np.asarray(h[0].data[:, ::stride, ::stride]).astype(np.float64) + bz  # LLM-generated: apply stride while loading
        else:
            X0, X1, Y0, Y1 = roi
            data = np.asarray(h[0].data[:, Y0:Y1:stride, X0:X1:stride]).astype(np.float64) + bz  # LLM-generated: apply stride while loading

    return data, fn


def plot_all_pixel_traces(data, outfile_name, pixel_stride=1, alpha=0.03,
                           title=None, dpi=150):
    """Plot every selected pixel's value against frame/read number.

    Parameters
    ----------
    data : ndarray, shape (N_reads, ny, nx)
    outfile_name : str
        Written to ``outfile_name + '.png'``.
    pixel_stride : int
        Subsample pixels by this stride in both spatial dimensions before
        plotting. Use 1 only if every pixel is desired.
    alpha : float
        Line transparency; keep this low since many traces overlap.
    title : str, optional
        Plot title.
    dpi : int
        Output PNG resolution.
    """
    if data.ndim != 3:
        raise ValueError(f"Expected data shape (N_reads, ny, nx); got {data.shape}")

    pixel_stride = int(pixel_stride)
    if pixel_stride < 1:
        raise ValueError("pixel_stride must be >= 1")

    alpha = float(alpha)
    if alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be > 0 and <= 1")

    n_reads = data.shape[0]
    frames = np.arange(n_reads)

    sub = data[:, ::pixel_stride, ::pixel_stride]
    traces = sub.reshape(n_reads, -1)

    plt.figure(figsize=(8, 6))
    plt.plot(frames, traces, color="k", alpha=alpha, linewidth=0.5)
    plt.xlabel("Frame / read number")
    plt.ylabel("Pixel value [DN]")
    if title:
        plt.title(title)
    else:
        plt.title(
            f"All pixel ramps (pixel_stride={pixel_stride}, "
            f"n_traces={traces.shape[1]})"
        )
    plt.tight_layout()
    plt.savefig(outfile_name + ".png", dpi=int(dpi))
    plt.close()

    return traces.shape[1]


def safe_name(name):
    return (
        str(name).strip().replace(" ", "_").replace("/", "_")
        .replace("\\", "_").replace(",", "")
    )


def main():
    if len(sys.argv) < 2:
        print("Error: No YAML configuration file provided")
        print("Usage: python3 plot_pix_trace.py config.yaml")
        sys.exit(1)

    yamlfile = sys.argv[1]
    with open(yamlfile) as f:
        config = yaml.safe_load(f)

    dataset = config["dataset"]
    gui = gui_io.is_gui_mode(config)
    head = dataset.get("head", "GUI")
    date = str(dataset.get("date", ""))
    datapath = dataset.get("datapath", "")
    outpath = dataset["outpath"]
    setname = dataset["name"]
    modes = dataset["modes"]
    quicklook = False if gui else truthy(dataset.get("quicklook", False))

    plot_cfg = config.get("plot_pix_trace", {})
    section = str(plot_cfg.get("section", "illuminated"))
    pixel_stride = int(plot_cfg.get("pixel_stride", 1))
    alpha = float(plot_cfg.get("alpha", 0.03))
    use_roi = plot_cfg.get("use_roi", None)
    roi = get_roi(config, use_roi)
    subtract_reference_read = truthy(
        plot_cfg.get("subtract_reference_read", False)
    )
    reference_read_index = int(plot_cfg.get("reference_read_index", 1))
    dpi = int(plot_cfg.get("dpi", 150))

    plot_dir = os.path.join(outpath, f"{setname}_pixel_traces")
    os.makedirs(plot_dir, exist_ok=True)

    logname = os.path.join(plot_dir, f"{setname}_pixel_traces.txt")
    n_attempted = n_plotted = 0  # LLM-generated

    with open(logname, "w") as log:
        P(log, "=== Pixel trace plotting ===")
        P(log, f"dataset={setname}")
        P(log, f"GUI Live Mode={gui}")
        if gui:
            P(log, f"data root={gui_io.GUI_BASE_PATH}")
        else:
            P(log, f"head={head}; date={date}; quicklook={quicklook}")
            P(log, f"data directory={data_directory(datapath, quicklook)}")
        P(log, f"section={section}; pixel_stride={pixel_stride}; alpha={alpha}; dpi={dpi}")
        P(log, f"subtract_reference_read={subtract_reference_read}; reference_read_index={reference_read_index}")
        if roi is None:
            P(log, "ROI=full frame")
        else:
            X0, X1, Y0, Y1 = roi
            P(log, f"ROI=x[{X0}:{X1}] y[{Y0}:{Y1}]")
        P(log, "")

        for mode_key in modes:
            mode_cfg = config[mode_key]
            name = mode_cfg.get("name", mode_key)
            ramps = parse_ramps(mode_cfg, section=section, gui=gui)

            if not ramps:
                P(log, f"{name}: no ramps found in section {section!r}; skipped")
                continue

            P(log, f"{name}: ramps={ramps}")

            for n in ramps:
                n_attempted += 1  # LLM-generated
                try:
                    data, fn = load_cube(
                        datapath, head, date, n,
                        quicklook=quicklook, roi=roi, gui=gui, pixel_stride=pixel_stride,  # LLM-generated
                    )

                    if subtract_reference_read:
                        if reference_read_index < 0 or reference_read_index >= data.shape[0]:
                            raise IndexError(
                                f"reference_read_index={reference_read_index} outside "
                                f"ramp with {data.shape[0]} reads"
                            )
                        data = data - data[reference_read_index][np.newaxis, :, :]

                    ident = gui_io.identifier_label(n) if gui else f"{int(n):05d}"
                    base = os.path.join(
                        plot_dir,
                        f"{setname}_{safe_name(name)}_{ident}_pixel_traces",
                    )
                    title = f"{setname} — {name} — ramp {ident}"
                    if subtract_reference_read:
                        title += f"\nrelative to read {reference_read_index}"

                    n_traces = plot_all_pixel_traces(
                        data,
                        base,
                        pixel_stride=1,  # LLM-generated: load_cube already applied the stride
                        alpha=alpha,
                        title=title,
                        dpi=dpi,
                    )

                    P(
                        log,
                        f"   ramp {ident}: {data.shape[0]} reads, "
                        f"{data.shape[2]}x{data.shape[1]} pixels loaded, "
                        f"{n_traces} traces plotted -> {base}.png",
                    )
                    n_plotted += 1  # LLM-generated
                except Exception as exc:
                    P(log, f"   ramp {gui_io.identifier_label(n)} failed: {exc}")

    if n_attempted and not n_plotted:  # LLM-generated: do not report success when nothing was plotted
        print(f"ERROR: all {n_attempted} ramps failed; no plots written. See {logname}")  # LLM-generated
        sys.exit(1)  # LLM-generated

    print(f"DONE: plots written to {plot_dir}")


if __name__ == "__main__":
    main()

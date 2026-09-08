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
import gui_io


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


def parse_ramps(mode_cfg, section="illuminated", gui=False):
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


def load_region_and_header(head, date, n, datapath, X0, X1, Y0, Y1, quicklook=False, gui=False):
    if gui:
        cube, header, source = gui_io.load_gui_cube(n, roi=(X0, X1, Y0, Y1))
        return cube, header, source

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
    """
    Analyze all illuminated ramps for one detector mode.

    If well_depth.find_bright_pixels is False or omitted:
        Each read is reduced using the configured statistic over the entire ROI.

    If well_depth.find_bright_pixels is True:
        For each ramp, identify strongly illuminated pixels from the difference
        between the reference read and the late-ramp signal, then calculate the
        ramp profile using only those pixels.

    All ramps are then combined into a representative median ramp.
    """

    dataset = config["dataset"]
    mode_cfg = config[mode_key]
    wd_cfg = config.get("well_depth", {})
    gui = gui_io.is_gui_mode(config)

    head = dataset.get("head", "GUI")
    date = str(dataset.get("date", ""))
    datapath = dataset.get("datapath", "")
    quicklook = False if gui else truthy(dataset.get("quicklook", False))

    name = mode_cfg["name"]

    ramps = parse_ramps(
        mode_cfg,
        section="illuminated",
        gui=gui,
    )

    if not ramps:
        raise ValueError(
            f"{name}: no illuminated ramps configured"
        )

    # ============================================================
    # Configuration
    # ============================================================

    reference_read_index = int(
        wd_cfg.get("reference_read_index", 1)
    )

    slope_start_frame = int(
        wd_cfg.get("slope_start_frame", 3)
    )

    slope_stop_frame = int(
        wd_cfg.get("slope_stop_frame", 6)
    )

    deviation_threshold = float(
        wd_cfg.get("deviation_threshold", 0.05)
    )

    slope_cutoff = float(
        wd_cfg.get(
            "slope_cutoff",
            wd_cfg.get("wd_cutoff", 0.05)
        )
    )

    min_signal_fraction = float(
        wd_cfg.get("min_signal_fraction", 0.25)
    )

    consecutive_points = int(
        wd_cfg.get(
            "consecutive_points",
            wd_cfg.get("sustained_points", 2)
        )
    )

    interpolation_points = int(
        wd_cfg.get("interpolation_points", 2000)
    )

    statistic = str(
        wd_cfg.get("statistic", "median")
    ).lower()

    errorbar = str(
        wd_cfg.get("errorbar", "sem")
    ).lower()

    # ------------------------------------------------------------
    # Bright-pixel selection
    # ------------------------------------------------------------

    find_bright_pixels = truthy(
        wd_cfg.get("find_bright_pixels", False)
    )

    # A pixel must show at least this fraction of the characteristic
    # large-signal change in order to be included.
    bright_pixel_fraction = float(
        wd_cfg.get("bright_pixel_fraction", 0.20)
    )

    bright_pixel_min_pixels = int(
        wd_cfg.get("bright_pixel_min_pixels", 10)
    )

    # Number of final reads to combine when constructing the
    # bright-pixel detection image.
    bright_pixel_late_reads = int(
        wd_cfg.get("bright_pixel_late_reads", 3)
    )

    # ------------------------------------------------------------
    # ADC limit
    # ------------------------------------------------------------

    if mode_cfg.get("adc_max") is not None:
        adc_max = float(mode_cfg["adc_max"])

    elif wd_cfg.get("adc_max") is not None:
        adc_max = float(wd_cfg["adc_max"])

    elif "fast" in name.lower():
        adc_max = 4092.0

    elif "slow" in name.lower():
        adc_max = 65535.0

    else:
        adc_max = None

    # ============================================================
    # Load ramps
    # ============================================================

    raw_ramp_profiles = []
    ramp_profiles = []

    bias_levels = []
    bright_pixel_counts = []

    headers = []
    loaded_files = []

    for n in ramps:

        cube, header, fn = load_region_and_header(
            head,
            date,
            n,
            datapath,
            X0,
            X1,
            Y0,
            Y1,
            quicklook=quicklook,
            gui=gui,
        )

        if reference_read_index >= cube.shape[0]:
            raise ValueError(
                f"{name}: reference_read_index="
                f"{reference_read_index} is outside ramp "
                f"length {cube.shape[0]}"
            )

        # ========================================================
        # Option 1:
        # find illuminated pixels first
        # ========================================================

        if find_bright_pixels:

            reference_image = np.asarray(
                cube[reference_read_index],
                dtype=float
            )

            # Average/median several late reads so the mask is not
            # determined by noise in one individual read.
            nlate = max(
                1,
                min(bright_pixel_late_reads, cube.shape[0])
            )

            late_image = np.nanmedian(
                cube[-nlate:],
                axis=0
            )

            # Positive accumulated signal between the reference
            # read and late ramp.
            delta_image = (
                late_image - reference_image
            )

            finite_delta = delta_image[
                np.isfinite(delta_image)
            ]

            if finite_delta.size == 0:
                raise ValueError(
                    f"{name} ramp {n}: no finite pixels "
                    f"available for bright-pixel selection"
                )

            # Don't use the literal maximum, since one hot/bad
            # pixel could set a ridiculous threshold.
            #
            # The 99th percentile gives us a representative
            # high-signal level.
            characteristic_delta = float(
                np.nanpercentile(
                    finite_delta,
                    99.0
                )
            )
            # Existing delta-based selection
            characteristic_delta = float(
                np.nanpercentile(finite_delta, 99.0)
            )

            bright_threshold = (
                bright_pixel_fraction
                * characteristic_delta
            )

            bright_mask = (
                np.isfinite(delta_image)
                & (delta_image >= bright_threshold)
            )

            # NEW: when ADC maximum is known, retain only pixels that
            # actually approach the upper part of the ADC range.
            if adc_max is not None:

                bright_adc_fraction = float(
                    wd_cfg.get(
                        "bright_pixel_adc_fraction",
                        0.80
                    )
                )

                bright_mask &= (
                    late_image
                    >= bright_adc_fraction * adc_max
                )

            nbright = int(
                np.sum(bright_mask)
            )

            if nbright < bright_pixel_min_pixels:
                raise ValueError(
                    f"{name} ramp {n}: only {nbright} bright "
                    f"pixels found; minimum required is "
                    f"{bright_pixel_min_pixels}. "
                    f"Threshold={bright_threshold:.2f} DN"
                )

            # Build one value per detector read, using ONLY the
            # selected illuminated pixels.
            if statistic == "mean":

                raw_prof = np.asarray([
                    np.nanmean(
                        frame[bright_mask]
                    )
                    for frame in cube
                ])

            else:

                raw_prof = np.asarray([
                    np.nanmedian(
                        frame[bright_mask]
                    )
                    for frame in cube
                ])

            bright_pixel_counts.append(
                nbright
            )

            """ P(
                log,
                f"   {name} ramp {n}: "
                f"{nbright} bright pixels selected "
                f"({100.0*nbright/bright_mask.size:.2f}% of ROI), "
                f"threshold={bright_threshold:.1f} DN"
            ) """

        # ========================================================
        # Option 2:
        # original whole-ROI behavior
        # ========================================================

        else:

            raw_prof = region_stat(
                cube,
                statistic=statistic
            )

            bright_pixel_counts.append(
                cube.shape[1] * cube.shape[2]
            )

        # --------------------------------------------------------
        # Save raw bias and form accumulated-signal ramp
        # --------------------------------------------------------

        bias = float(
            raw_prof[reference_read_index]
        )

        prof = (
            raw_prof - bias
        )

        raw_ramp_profiles.append(
            raw_prof
        )

        ramp_profiles.append(
            prof
        )

        bias_levels.append(
            bias
        )

        headers.append(
            header
        )

        loaded_files.append(
            fn
        )

    # ============================================================
    # Trim all ramps to common length
    # ============================================================

    min_len = min(
        len(p)
        for p in ramp_profiles
    )

    profiles = np.asarray([
        p[:min_len]
        for p in ramp_profiles
    ])

    raw_profiles = np.asarray([
        p[:min_len]
        for p in raw_ramp_profiles
    ])

    frame_numbers = np.arange(
        1,
        min_len + 1,
        dtype=float
    )

    # ============================================================
    # Representative ramp
    # ============================================================

    signal = np.median(
        profiles,
        axis=0
    )

    raw_signal = np.median(
        raw_profiles,
        axis=0
    )

    median_bias = float(
        np.median(bias_levels)
    )

    if len(profiles) > 1:

        scatter = np.std(
            profiles,
            axis=0,
            ddof=1
        )

    else:

        scatter = np.zeros_like(
            signal
        )

    if (
        errorbar == "sem"
        and len(profiles) > 1
    ):

        yerr = (
            scatter
            / np.sqrt(len(profiles))
        )

    elif errorbar == "std":

        yerr = scatter

    else:

        yerr = None

    itime = frame_time_from_header_or_config(
        headers[0],
        mode_cfg
    )

    time_sec = (
        frame_numbers * itime
    )

    # ============================================================
    # Basic sanity checks
    # ============================================================

    if not np.all(
        np.isfinite(signal)
    ):

        raise ValueError(
            f"{name}: representative ramp contains "
            f"non-finite values"
        )

    max_signal = float(
        np.nanmax(signal)
    )

    if max_signal <= 0:

        raise ValueError(
            f"{name}: no positive accumulated "
            f"signal was measured"
        )

    # ADC headroom measured from the selected-pixel bias.
    if adc_max is not None:

        adc_available_dn = (
            adc_max
            - median_bias
        )

    else:

        adc_available_dn = np.nan

    if find_bright_pixels:

        median_nbright = int(
            np.median(
                bright_pixel_counts
            )
        )

        P(
            log,
            f"   {name}: "
            f"median bright pixels={median_nbright}, "
            f"median bias={median_bias:.1f} DN, "
            f"max accumulated={max_signal:.1f} DN"
            + (
                f", ADC max={adc_max:.0f} DN, "
                f"available headroom="
                f"{adc_available_dn:.1f} DN"
                if adc_max is not None
                else ""
            )
        )

    else:

        P(
            log,
            f"   {name}: "
            f"median bias={median_bias:.1f} DN, "
            f"max accumulated={max_signal:.1f} DN"
            + (
                f", ADC max={adc_max:.0f} DN, "
                f"available headroom="
                f"{adc_available_dn:.1f} DN"
                if adc_max is not None
                else ""
            )
        )

    # ============================================================
    # Initial linear fit
    # ============================================================

    i0 = max(
        0,
        slope_start_frame - 1
    )

    i1 = min(
        min_len - 1,
        slope_stop_frame - 1
    )

    if i1 <= i0:

        raise ValueError(
            f"{name}: slope_stop_frame must be "
            f"after slope_start_frame"
        )

    if (
        i0
        <= reference_read_index
        <= i1
    ):

        raise ValueError(
            f"{name}: linear-fit frames "
            f"{slope_start_frame}-"
            f"{slope_stop_frame} include the "
            f"reference read "
            f"(index {reference_read_index})"
        )

    fit_indices = np.arange(
        i0,
        i1 + 1
    )

    xfit = frame_numbers[
        fit_indices
    ]

    yfit = signal[
        fit_indices
    ]

    valid_fit = (
        np.isfinite(xfit)
        & np.isfinite(yfit)
    )

    if np.sum(valid_fit) < 2:

        raise ValueError(
            f"{name}: insufficient valid points "
            f"for initial linear fit"
        )

    slope_frame, intercept = np.polyfit(
        xfit[valid_fit],
        yfit[valid_fit],
        1
    )

    include_pre_read_signal = truthy(
        wd_cfg.get("include_pre_read_signal", False)
    )

    pre_read_frames = float(
        wd_cfg.get("pre_read_frames", 1.0)
    )

    if include_pre_read_signal:
        pre_read_dn = (
            slope_frame * pre_read_frames
        )
    else:
        pre_read_dn = 0.0

    P(
        log,
        f"   {name}: estimated pre-read signal="
        f"{pre_read_dn:.1f} DN "
        f"({pre_read_frames:.2f} frame intervals)"
    )

    if slope_frame <= 0:

        raise ValueError(
            f"{name}: initial slope is "
            f"non-positive "
            f"({slope_frame:.3f} DN/frame)"
        )

    slope_dn_s = (
        slope_frame / itime
    )

    line = (
        slope_frame
        * frame_numbers
        + intercept
    )

    signal_total = (
        signal + pre_read_dn
    )

    line_total = (
        line + pre_read_dn
    )
    # ============================================================
    # Fractional rolloff
    # ============================================================

    if include_pre_read_signal:
        denom = np.where(
            line_total > 0,
            line_total,
            np.nan
        )

        deviation = (
            1.0
            - signal_total / denom
        )
    else:
        denom = np.where(
            line > 0,
            line,
            np.nan
        )

        deviation = (
            1.0
            - signal / denom
        )

    # ============================================================
    # Define well-depth search region
    # ============================================================

    observed_floor = (
        min_signal_fraction
        * max_signal
    )

    if (
        adc_max is not None
        and np.isfinite(adc_available_dn)
        and adc_available_dn > 0
    ):

        search_floor = (
            min_signal_fraction
            * max_signal
        )

    else:

        search_floor = (
            observed_floor
        )

    after_fit = (
        frame_numbers
        > frame_numbers[i1]
    )

    above_min_signal = (
        signal
        >= search_floor
    )

    # ============================================================
    # Criterion 1:
    # sustained fractional rolloff
    # ============================================================

    deviation_mask = (
        after_fit
        & above_min_signal
        & np.isfinite(deviation)
        & (
            deviation
            >= deviation_threshold
        )
    )
    k_dev = first_sustained_hit(
        deviation_mask,
        consecutive_points
    )

    if k_dev is not None:

        # --------------------------------------------------------
        # Interpolate between the previous native read and the
        # first read past the deviation threshold.
        # --------------------------------------------------------

        if k_dev > 0:

            k0 = k_dev - 1
            k1 = k_dev

            d0 = deviation[k0]
            d1 = deviation[k1]

            # Linear interpolation fraction where deviation reaches
            # exactly deviation_threshold.
            if (
                np.isfinite(d0)
                and np.isfinite(d1)
                and d1 != d0
            ):

                frac = (
                    deviation_threshold - d0
                ) / (
                    d1 - d0
                )

                # Clamp to the actual interval in case of numerical
                # weirdness.
                frac = np.clip(
                    frac,
                    0.0,
                    1.0
                )

                dev_frame = (
                    frame_numbers[k0]
                    + frac
                    * (
                        frame_numbers[k1]
                        - frame_numbers[k0]
                    )
                )

                dev_time = (
                    time_sec[k0]
                    + frac
                    * (
                        time_sec[k1]
                        - time_sec[k0]
                    )
                )
                if include_pre_read_signal:
                    signal = signal_total
                dev_dn = (
                    signal[k0]
                    + frac
                    * (
                        signal[k1]
                        - signal[k0]
                    )
                )

            else:

                # Fall back to native point.
                dev_frame = float(
                    frame_numbers[k_dev]
                )

                dev_time = float(
                    time_sec[k_dev]
                )

                dev_dn = float(
                    signal[k_dev]
                )

        else:

            dev_frame = float(
                frame_numbers[k_dev]
            )

            dev_time = float(
                time_sec[k_dev]
        )

        dev_dn = float(
            signal[k_dev]
        )

    else:

        dev_frame = np.nan
        dev_time = np.nan
        dev_dn = np.nan

    # ============================================================
    # Criterion 2:
    # sustained slope reduction
    # ============================================================

    local_slope = np.gradient(
        signal,
        frame_numbers
    )

    slope_threshold = (
        slope_cutoff
        * slope_frame
    )

    slope_mask = (
        after_fit
        & above_min_signal
        & np.isfinite(local_slope)
        & (
            local_slope
            <= slope_threshold
        )
    )

    k_slope = first_sustained_hit(
        slope_mask,
        consecutive_points
    )

    if k_slope is not None:

        if k_slope > 0:

            k0 = k_slope - 1
            k1 = k_slope

            s0 = local_slope[k0]
            s1 = local_slope[k1]

            if (
                np.isfinite(s0)
                and np.isfinite(s1)
                and s1 != s0
            ):

                frac = (
                    slope_threshold - s0
                ) / (
                    s1 - s0
                )

                frac = np.clip(
                    frac,
                    0.0,
                    1.0
                )

                cutoff_frame = (
                    frame_numbers[k0]
                    + frac
                    * (
                        frame_numbers[k1]
                        - frame_numbers[k0]
                    )
                )

                cutoff_time = (
                    time_sec[k0]
                    + frac
                    * (
                        time_sec[k1]
                        - time_sec[k0]
                    )
                )
                if include_pre_read_signal:
                    signal = signal_total 

                cutoff_dn = (
                    signal[k0]
                    + frac
                    * (
                        signal[k1]
                        - signal[k0]
                    )
                )

            else:

                cutoff_frame = float(
                    frame_numbers[k_slope]
                )

                cutoff_time = float(
                    time_sec[k_slope]
                )

                cutoff_dn = float(
                    signal[k_slope]
                )

        else:

            cutoff_frame = float(
                frame_numbers[k_slope]
            )

            cutoff_time = float(
                time_sec[k_slope]
            )

            cutoff_dn = float(
                signal[k_slope]
            )

    else:

        cutoff_frame = np.nan
        cutoff_time = np.nan
        cutoff_dn = np.nan

    # ============================================================
    # ADC saturation
    #
    # adc_max is RAW DN, so compare against raw_signal.
    # ============================================================

    if adc_max is not None:

        adc_fraction = float(
            wd_cfg.get(
                "adc_saturation_fraction",
                0.99
            )
        )

        adc_mask = (
            raw_signal
            >= adc_fraction * adc_max
        )

        k_adc = first_sustained_hit(
            adc_mask,
            consecutive_points
        )

        if k_adc is not None:

            adc_frame = float(
                frame_numbers[k_adc]
            )

            adc_time = float(
                time_sec[k_adc]
            )

            if include_pre_read_signal:
                signal = signal_total

            adc_dn = float(
                signal[k_adc]
            )

            adc_raw_dn = float(
                raw_signal[k_adc]
            )

        else:

            adc_frame = np.nan
            adc_time = np.nan
            adc_dn = np.nan
            adc_raw_dn = np.nan

    else:

        adc_frame = np.nan
        adc_time = np.nan
        adc_dn = np.nan
        adc_raw_dn = np.nan

    # ============================================================
    # Interpolation retained for plotting only
    # ============================================================

    x_interp = np.linspace(
        frame_numbers[0],
        frame_numbers[-1],
        interpolation_points
    )

    signal_interp = np.interp(
        x_interp,
        frame_numbers,
        signal
    )

    line_interp = (
        slope_frame
        * x_interp
        + intercept
    )

    gain = get_gain(
        mode_cfg
    )

    # ============================================================
    # Return
    # ============================================================

    return {
        "mode_key": mode_key,
        "name": name,

        "ramps": ramps,
        "files": loaded_files,

        "profiles": profiles,
        "raw_profiles": raw_profiles,

        "frame_numbers": frame_numbers,
        "time_sec": time_sec,

        "signal_dn": signal_total,
        "raw_signal_dn": raw_signal,
        
        "include_pre_read_signal": include_pre_read_signal,
        "signal_since_read1_dn": signal,
        "pre_read_signal_dn": pre_read_dn,

        "scatter_dn": scatter,
        "yerr_dn": yerr,

        "line_dn": line,
        "deviation": deviation,

        "itime": itime,
        "gain": gain,

        "median_bias_dn": median_bias,
        "bias_levels_dn": bias_levels,

        "find_bright_pixels": find_bright_pixels,
        "bright_pixel_counts": bright_pixel_counts,
        "bright_pixel_fraction": bright_pixel_fraction,

        "adc_max": adc_max,
        "adc_available_dn": adc_available_dn,

        "slope_frame_dn": slope_frame,
        "slope_dn_s": slope_dn_s,

        "deviation_threshold": deviation_threshold,

        "well_depth_deviation_dn": dev_dn,
        "time_deviation_s": dev_time,
        "frame_deviation": dev_frame,

        "slope_cutoff": slope_cutoff,

        "well_depth_cutoff_dn": cutoff_dn,
        "time_cutoff_s": cutoff_time,
        "frame_cutoff": cutoff_frame,

        "adc_saturation_dn": adc_dn,
        "adc_saturation_raw_dn": adc_raw_dn,
        "adc_saturation_time_s": adc_time,
        "adc_saturation_frame": adc_frame,

        "min_signal_fraction": min_signal_fraction,
        "search_floor_dn": search_floor,
        "consecutive_points": consecutive_points,

        "reference_read_index": reference_read_index,
        "slope_start_frame": slope_start_frame,
        "slope_stop_frame": slope_stop_frame,

        "interp_frame_numbers": x_interp,
        "interp_signal_dn": signal_interp,
        "interp_line_dn": line_interp,
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
    gui = gui_io.is_gui_mode(config)
    quicklook = False if gui else truthy(dataset.get("quicklook", False))

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
        P(log, f"GUI Live Mode: {gui}")
        if gui:
            P(log, f"GUI data root: {gui_io.GUI_BASE_PATH}")
        else:
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

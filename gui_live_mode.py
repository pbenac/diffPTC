#!/usr/bin/env python3
"""GUI Live Mode quick analysis for MACIE GUI UpTheRamp acquisitions.

Usage:
    python3 gui_live_mode.py 20260831115412

The timestamp selects:
/usr/local/MACIE/MACIE_v6.0/macie_v6.0_ubuntu/MacieApp/data/macie/H2RG_ASIC/UpTheRamp/<timestamp>/

Individual reads are discovered as Frame_R0001_M0001_N*.fits, sorted by the
integer after _N, and stacked into a cube with shape (N_reads, ny, nx).
"""

import argparse
import glob
import os
import re
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

import gui_io

GUI_BASE_PATH = gui_io.GUI_BASE_PATH
FRAME_GLOB = gui_io.GUI_FRAME_GLOB


def gui_ramp_directory(timestamp):
    return gui_io.gui_ramp_directory(timestamp)


def frame_number_from_filename(filename):
    return gui_io.frame_number_from_filename(filename)


def discover_gui_frames(timestamp):
    return gui_io.discover_gui_frames(timestamp)


def load_gui_cube(timestamp, roi=None):
    """Compatibility wrapper around shared gui_io.load_gui_cube()."""
    cube, header, _source = gui_io.load_gui_cube(timestamp, roi=roi)
    files = gui_io.discover_gui_frames(timestamp)
    return cube, [header], files


def reference_pixel_zero_check(timestamp, read_index=1, ncols=4):
    """Count zero-valued pixels in the first detector columns for one read.

    Reads the full detector frame even when --roi is used, since the reference
    columns may lie outside the cropped science ROI.
    """
    files = discover_gui_frames(timestamp)
    if not (0 <= int(read_index) < len(files)):
        raise ValueError(
            f"reference-pixel read index {read_index} is outside acquisition "
            f"with {len(files)} reads"
        )

    fn = files[int(read_index)]
    with fits.open(fn, memmap=True, do_not_scale_image_data=True) as h:
        bz = h[0].header.get("BZERO", 0)
        refpix = np.asarray(h[0].data[:, :int(ncols)]).astype(np.float64) + bz

    finite = np.isfinite(refpix)
    zeros = finite & (refpix == 0)
    nfinite = int(np.count_nonzero(finite))
    nzero = int(np.count_nonzero(zeros))

    return {
        "read_index": int(read_index),
        "ncols": int(ncols),
        "n_pixels": nfinite,
        "n_zero": nzero,
        "zero_fraction": float(nzero / nfinite) if nfinite else np.nan,
        "filename": fn,
    }


def trimmed_median(image, low_pct=1.0, high_pct=99.0):
    arr = np.asarray(image, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return np.nan
    lo, hi = np.percentile(arr, [low_pct, high_pct])
    keep = (arr >= lo) & (arr <= hi)
    return float(np.median(arr[keep])) if np.any(keep) else np.nan


def correct(d, channel_width=512):
    d = np.asarray(d, dtype=float).copy()
    if channel_width > 0:
        for c in range(d.shape[1] // channel_width):
            sl = slice(c*channel_width, (c+1)*channel_width)
            d[:, sl] -= np.median(d[:, sl])
    d -= np.median(d, axis=1, keepdims=True)
    return d


def quick_read_noise(cube, gain=None, channel_width=512):
    """Quick CDS estimate from array indices 2-1, matching dark_noise.py convention."""
    if cube.shape[0] < 3:
        return dict(raw_dn=np.nan, corrected_dn=np.nan, raw_e=np.nan, corrected_e=np.nan)
    d = cube[2] - cube[1]
    raw_std = sigma_clipped_stats(d, sigma=3.0, maxiters=5)[2]
    corr_std = sigma_clipped_stats(correct(d, channel_width), sigma=3.0, maxiters=5)[2]
    q_var = 2.0 / 12.0
    def one(std):
        v = std**2 - q_var
        return np.sqrt(v/2.0) if v > 0 else np.nan
    raw_dn = raw_std / np.sqrt(2.0)
    corr_dn = one(corr_std)
    return dict(raw_dn=raw_dn, corrected_dn=corr_dn,
                raw_e=raw_dn*gain if gain is not None else np.nan,
                corrected_e=corr_dn*gain if gain is not None else np.nan)


def first_sustained_hit(mask, n):
    mask = np.asarray(mask, dtype=bool)
    n = max(1, int(n))
    if n == 1:
        hits = np.where(mask)[0]
        return int(hits[0]) if len(hits) else None
    if len(mask) < n:
        return None
    run = np.convolve(mask.astype(int), np.ones(n, dtype=int), mode="valid")
    hits = np.where(run >= n)[0]
    return int(hits[0]) if len(hits) else None


def interp_crossing(x0, x1, y0, y1, target):
    if not all(np.isfinite(v) for v in (x0,x1,y0,y1,target)) or y1 == y0:
        return float(x1)
    f = np.clip((target-y0)/(y1-y0), 0.0, 1.0)
    return float(x0 + f*(x1-x0))


def select_bright_pixels(cube, reference_read_index=1, adc_max=None, min_pixels=10):
    """Select pixels appropriate for well-depth measurement.

    A bright/well-depth pixel must:
      * be finite in the first science read (reference_read_index),
      * begin below the ADC rail in that read, and
      * reach the ADC rail by the final read.

    Pixels already at the ADC rail in the reference read are excluded.  This
    removes stuck/saturated pixels that remain at the rail for the whole ramp.
    """
    if adc_max is None:
        raise ValueError(
            "Cannot construct the well-depth bright-pixel mask without an ADC rail. "
            "Pass --adc-max or use a ramp that visibly reaches a recognized rail."
        )
    if not (0 <= int(reference_read_index) < cube.shape[0]):
        raise ValueError(
            f"reference_read_index={reference_read_index} is outside ramp length {cube.shape[0]}"
        )

    ref = np.asarray(cube[int(reference_read_index)], dtype=float)
    last = np.asarray(cube[-1], dtype=float)

    finite = np.isfinite(ref) & np.isfinite(last)
    starts_below_rail = ref < float(adc_max)
    reaches_rail_last = last >= float(adc_max)

    mask = finite & starts_below_rail & reaches_rail_last

    # Useful diagnostics for the selection itself.
    starts_at_rail = finite & (ref >= float(adc_max))
    stuck_at_rail = starts_at_rail & np.all(
        np.isfinite(cube) & (cube >= float(adc_max)), axis=0
    )

    nsel = int(mask.sum())
    if nsel < int(min_pixels):
        raise ValueError(
            f"Only {nsel} usable bright pixels found; need at least {min_pixels}. "
            f"Selection requires reference read < {adc_max:g} DN and final read >= {adc_max:g} DN."
        )

    info = {
        "selected_pixels": nsel,
        "starts_at_rail": int(starts_at_rail.sum()),
        "stuck_at_rail": int(stuck_at_rail.sum()),
    }
    return mask, info

def ramp_profile(cube, reference_read_index=1, statistic="median",
                 use_bright_pixels=True, adc_max=None, min_pixels=10):
    """Build the representative ramp used for well-depth analysis.

    Bright-pixel selection is the default.  The representative trace is the
    median (or mean) of pixels that start below the ADC rail in the reference
    read and reach the rail in the final read.
    """
    selection_info = None
    if use_bright_pixels:
        mask, selection_info = select_bright_pixels(
            cube,
            reference_read_index=reference_read_index,
            adc_max=adc_max,
            min_pixels=min_pixels,
        )
        reducer = np.nanmean if statistic == "mean" else np.nanmedian
        raw = np.array([reducer(frame[mask]) for frame in cube])
        npix = int(mask.sum())
    else:
        flat = cube.reshape(cube.shape[0], -1)
        raw = np.nanmean(flat, axis=1) if statistic == "mean" else np.nanmedian(flat, axis=1)
        npix = flat.shape[1]
        selection_info = {
            "selected_pixels": int(npix),
            "starts_at_rail": 0,
            "stuck_at_rail": 0,
        }

    bias = float(raw[reference_read_index])
    return raw, raw-bias, bias, npix, selection_info



def infer_observed_adc_rail(cube):
    """Infer a common ADC rail only when the data visibly reach it.

    This deliberately does not guess the detector mode from an unsaturated ramp.
    It recognizes the two common rails used by these data only when the cube
    actually contains values at/very near the rail.
    """
    raw_max = float(np.nanmax(cube))
    if raw_max >= 65534.5:
        return 65535.0
    if raw_max < 10000.0 and raw_max >= 4091.5:
        return 4092.0
    return None


def pixel_response_stats(cube, reference_read_index=1, adc_max=None):
    """Summarize individual-pixel accumulation without spatially collapsing the ROI."""
    if not (0 <= reference_read_index < cube.shape[0]):
        raise ValueError(
            f"reference_read_index={reference_read_index} is outside ramp length {cube.shape[0]}"
        )

    ref = np.asarray(cube[reference_read_index], dtype=float)
    accum = np.asarray(cube, dtype=float) - ref[None, :, :]

    raw_max = float(np.nanmax(cube))
    last_raw_max = float(np.nanmax(cube[-1]))
    max_accum = float(np.nanmax(accum))
    last_max_accum = float(np.nanmax(accum[-1]))

    per_pixel_max = np.nanmax(accum, axis=0)
    finite = np.isfinite(per_pixel_max)
    pct = {}
    if np.any(finite):
        for q in (50, 90, 95, 99, 100):
            pct[q] = float(np.nanpercentile(per_pixel_max[finite], q))
    else:
        for q in (50, 90, 95, 99, 100):
            pct[q] = np.nan

    flat_idx = int(np.nanargmax(accum))
    k, y, x = np.unravel_index(flat_idx, accum.shape)

    result = dict(
        raw_max_dn=raw_max,
        last_raw_max_dn=last_raw_max,
        max_accum_dn=max_accum,
        last_max_accum_dn=last_max_accum,
        max_accum_frame=int(k),
        max_accum_x=int(x),
        max_accum_y=int(y),
        max_accum_reference_dn=float(ref[y, x]),
        max_accum_raw_dn=float(cube[k, y, x]),
        per_pixel_max_percentiles=pct,
        adc_max=adc_max,
        adc_hit_pixels=0,
        adc_hit_fraction=0.0,
        adc_hit_pixels_last=0,
        adc_hit_fraction_last=0.0,
        adc_first_frame=None,
    )

    if adc_max is not None:
        finite_pix = np.any(np.isfinite(cube), axis=0)
        n_valid = int(np.sum(finite_pix))
        ever_hit = np.any(np.isfinite(cube) & (cube >= adc_max), axis=0)
        last_hit = np.isfinite(cube[-1]) & (cube[-1] >= adc_max)
        hit_frames = np.where(np.any(np.isfinite(cube) & (cube >= adc_max), axis=(1, 2)))[0]

        result['adc_hit_pixels'] = int(np.sum(ever_hit))
        result['adc_hit_pixels_last'] = int(np.sum(last_hit))
        result['adc_hit_fraction'] = (float(np.sum(ever_hit))/n_valid) if n_valid else np.nan
        result['adc_hit_fraction_last'] = (float(np.sum(last_hit))/n_valid) if n_valid else np.nan
        result['adc_first_frame'] = int(hit_frames[0]) if len(hit_frames) else None

    return result

def infer_frame_time(header):
    for key in ("ITIME","itime","FRAMTIME","FRMTIME","TFRAME","FRAMESEC"):
        if key in header and header[key] is not None:
            try:
                return float(header[key])
            except Exception:
                pass
    return None


def analyze_well_depth(cube, header, reference_read_index=1,
                       slope_start_frame=3, slope_stop_frame=6,
                       slope_cutoff=0.50, min_signal_fraction=0.25,
                       consecutive_points=2, statistic="median",
                       use_bright_pixels=True, bright_pixel_min_pixels=10,
                       adc_max=None, include_pre_read_signal=True,
                       pre_read_frames=1.0, frametime=None):
    raw, signal, bias, npix, selection_info = ramp_profile(
        cube,
        reference_read_index=reference_read_index,
        statistic=statistic,
        use_bright_pixels=use_bright_pixels,
        adc_max=adc_max,
        min_pixels=bright_pixel_min_pixels,
    )

    frames = np.arange(1, len(signal)+1, dtype=float)
    i0 = max(0, slope_start_frame-1)
    i1 = min(len(signal)-1, slope_stop_frame-1)
    if i1 <= i0:
        raise ValueError("Invalid early-fit frame range")

    slope, intercept = np.polyfit(frames[i0:i1+1], signal[i0:i1+1], 1)
    line = slope*frames + intercept
    pre = slope*pre_read_frames if include_pre_read_signal else 0.0
    total = signal + pre
    line_total = line + pre
    rolloff = 1.0 - total/np.where(np.abs(line_total)>1e-12, line_total, np.nan)

    max_signal = float(np.nanmax(total))
    valid = (frames > frames[i1]) & (total >= min_signal_fraction*max_signal)

    def rolloff_crossing(threshold):
        mask = valid & np.isfinite(rolloff) & (rolloff >= threshold)
        k = first_sustained_hit(mask, consecutive_points)
        if k is not None and k > 0:
            frame = interp_crossing(
                frames[k-1], frames[k], rolloff[k-1], rolloff[k], threshold
            )
            dn = float(np.interp(frame, frames, total))
            return frame, dn
        if k is not None:
            return float(frames[k]), float(total[k])
        return np.nan, np.nan

    frame_5, wd_5 = rolloff_crossing(0.05)
    frame_10, wd_10 = rolloff_crossing(0.10)

    # Keep the old slope-cutoff diagnostic, but it is no longer the reported WD.
    local = np.gradient(total, frames)
    slope_threshold = slope_cutoff*slope
    if slope >= 0:
        cutmask = valid & np.isfinite(local) & (local <= slope_threshold)
    else:
        cutmask = valid & np.isfinite(local) & (local >= slope_threshold)
    k = first_sustained_hit(cutmask, consecutive_points)
    if k is not None and k > 0:
        cut_frame = interp_crossing(
            frames[k-1], frames[k], local[k-1], local[k], slope_threshold
        )
        cut_dn = float(np.interp(cut_frame, frames, total))
    elif k is not None:
        cut_frame, cut_dn = float(frames[k]), float(total[k])
    else:
        cut_frame = cut_dn = np.nan

    adc_dn = np.nan
    if adc_max is not None:
        hits = np.where(np.isfinite(raw) & (raw >= adc_max))[0]
        if len(hits):
            adc_dn = float(total[hits[0]])

    ft = float(frametime) if frametime is not None else infer_frame_time(header)
    valid_rolloff = rolloff[valid & np.isfinite(rolloff)]
    valid_slope = local[valid & np.isfinite(local)]

    return dict(
        bias_dn=bias,
        selected_pixels=npix,
        selection_info=selection_info,
        slope_dn_frame=float(slope),
        slope_dn_s=float(slope/ft) if ft else np.nan,
        pre_read_dn=float(pre),
        max_signal_dn=max_signal,
        well_depth_5pct_dn=float(wd_5),
        well_depth_10pct_dn=float(wd_10),
        frame_5pct=float(frame_5),
        frame_10pct=float(frame_10),
        cutoff_dn=float(cut_dn),
        adc_well_dn=float(adc_dn),
        max_rolloff=float(np.nanmax(valid_rolloff)) if len(valid_rolloff) else np.nan,
        min_slope_fraction=(
            float(np.nanmin(valid_slope/slope))
            if slope != 0 and len(valid_slope) else np.nan
        ),
        frame_time_s=ft,
    )


def fmt(x, nd=2):
    return "not reached" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def parse_args():
    p = argparse.ArgumentParser(description="GUI Live Mode quick detector analysis with bright-pixel well-depth selection")
    p.add_argument("timestamp")
    p.add_argument("--roi", nargs=4, type=int, metavar=("X0","X1","Y0","Y1"))
    p.add_argument("--gain", type=float)
    p.add_argument("--frametime", type=float)
    p.add_argument("--adc-max", type=float)
    p.add_argument("--reference-read-index", type=int, default=1)
    p.add_argument("--slope-start-frame", type=int, default=3)
    p.add_argument("--slope-stop-frame", type=int, default=6)
    p.add_argument("--slope-cutoff", type=float, default=0.50)
    p.add_argument("--min-signal-fraction", type=float, default=0.25)
    p.add_argument("--consecutive-points", type=int, default=2)
    p.add_argument("--statistic", choices=("median","mean"), default="median")
    p.add_argument("--all-pixels-well-depth", action="store_true",
                   help="Use the full ROI for well-depth analysis instead of the default bright-pixel mask")
    p.add_argument("--bright-pixel-min-pixels", type=int, default=10)
    p.add_argument("--no-pre-read-correction", action="store_true")
    p.add_argument("--pre-read-frames", type=float, default=1.0)
    p.add_argument("--bias-read-index", type=int, default=1)
    p.add_argument("--bias-trim-low", type=float, default=1.0)
    p.add_argument("--bias-trim-high", type=float, default=99.0)
    p.add_argument("--channel-width", type=int, default=512)
    p.add_argument("--pixel-stats", action="store_true",
                   help="Report individual-pixel raw/accumulated maxima and ADC-hit fractions")
    return p.parse_args()


def main():
    a = parse_args()
    roi = tuple(a.roi) if a.roi else None
    cube, headers, files = load_gui_cube(a.timestamp, roi)
    bias = trimmed_median(cube[a.bias_read_index], a.bias_trim_low, a.bias_trim_high)
    noise = quick_read_noise(cube, a.gain, a.channel_width)
    observed_adc_max = infer_observed_adc_rail(cube)
    effective_adc_max = a.adc_max if a.adc_max is not None else observed_adc_max
    pixstats = pixel_response_stats(cube, a.reference_read_index, effective_adc_max) if a.pixel_stats else None
    refzero = reference_pixel_zero_check(a.timestamp, read_index=1, ncols=4)
    wd = analyze_well_depth(
        cube, headers[0],
        reference_read_index=a.reference_read_index,
        slope_start_frame=a.slope_start_frame,
        slope_stop_frame=a.slope_stop_frame,
        slope_cutoff=a.slope_cutoff,
        min_signal_fraction=a.min_signal_fraction,
        consecutive_points=a.consecutive_points,
        statistic=a.statistic,
        use_bright_pixels=not a.all_pixels_well_depth,
        bright_pixel_min_pixels=a.bright_pixel_min_pixels,
        adc_max=effective_adc_max,
        include_pre_read_signal=not a.no_pre_read_correction,
        pre_read_frames=a.pre_read_frames,
        frametime=a.frametime,
    )

    print("\n=== GUI LIVE MODE ===")
    print(f"Acquisition: {a.timestamp}")
    print(f"Directory:   {gui_ramp_directory(a.timestamp)}")
    print(f"Reads:       {cube.shape[0]}")
    print(f"Shape:       {cube.shape[2]} x {cube.shape[1]}")
    print(f"Files:       {os.path.basename(files[0])} ... {os.path.basename(files[-1])}\n")
    print("BIAS / STARTING LEVEL")
    print(f"  Bias (read index {a.bias_read_index}): {bias:.2f} DN\n")

    print("REFERENCE PIXEL CHECK")
    print(
        f"  First {refzero['ncols']} detector columns, read index "
        f"{refzero['read_index']} (read 1):"
    )
    if refzero["n_zero"] > 0:
        print(
            f"  WARNING: {refzero['n_zero']} of {refzero['n_pixels']} reference pixels "
            f"are exactly 0 DN ({100*refzero['zero_fraction']:.3f}%)"
        )
    else:
        print(f"  Zero-valued reference pixels: 0 of {refzero['n_pixels']} (none)")
    print()

    print("READ NOISE (quick single-ramp CDS)")
    print(f"  Raw:       {fmt(noise['raw_dn'])} DN")
    print(f"  Corrected: {fmt(noise['corrected_dn'])} DN")
    if a.gain is not None:
        print(f"  Raw:       {fmt(noise['raw_e'])} e-")
        print(f"  Corrected: {fmt(noise['corrected_e'])} e-")
    print("\nWELL DEPTH / RAMP RESPONSE")
    if a.all_pixels_well_depth:
        print("  Pixel selection:       ALL ROI PIXELS (--all-pixels-well-depth)")
    else:
        print("  Pixel selection:       BRIGHT PIXELS ONLY")
        print(
            f"                         reference read < {effective_adc_max:.0f} DN "
            f"and final read >= {effective_adc_max:.0f} DN"
        )
        sel = wd["selection_info"]
        print(f"  Pixels starting at rail: {sel['starts_at_rail']} (excluded)")
        print(f"  Pixels stuck at rail:    {sel['stuck_at_rail']} (excluded)")
    print(f"  Pixels used:          {wd['selected_pixels']}")
    print(f"  Reference bias:       {wd['bias_dn']:.2f} DN")
    print(f"  Initial slope:        {wd['slope_dn_frame']:.2f} DN/frame")
    if np.isfinite(wd['slope_dn_s']):
        print(f"                        {wd['slope_dn_s']:.2f} DN/s")
    print(f"  Pre-read correction:  {wd['pre_read_dn']:.2f} DN")
    print(f"  Maximum accumulated:  {wd['max_signal_dn']:.2f} DN")
    print(f"  5.0% rolloff WD:      {fmt(wd['well_depth_5pct_dn'])} DN")
    print(f"  10.0% rolloff WD:     {fmt(wd['well_depth_10pct_dn'])} DN")
    if np.isfinite(wd['frame_5pct']):
        print(f"    5% crossing frame:  {wd['frame_5pct']:.2f}")
    if np.isfinite(wd['frame_10pct']):
        print(f"    10% crossing frame: {wd['frame_10pct']:.2f}")
    print(f"  Slope-cutoff diagnostic: {fmt(wd['cutoff_dn'])} DN")
    if effective_adc_max is not None:
        print(f"  ADC rail used:        {effective_adc_max:.0f} DN")
        print(f"  ADC-rail WD:          {fmt(wd['adc_well_dn'])} DN  (representative selected-pixel trace)")
    print(f"  Max rolloff:          {100*wd['max_rolloff']:.2f}%")
    print(f"  Min slope fraction:   {wd['min_slope_fraction']:.3f}")
    if a.gain is not None and np.isfinite(wd['well_depth_5pct_dn']):
        print(f"  5% rolloff WD:        {wd['well_depth_5pct_dn']*a.gain:.0f} e-")
    if a.gain is not None and np.isfinite(wd['well_depth_10pct_dn']):
        print(f"  10% rolloff WD:       {wd['well_depth_10pct_dn']*a.gain:.0f} e-")

    if pixstats is not None:
        print("\nINDIVIDUAL-PIXEL RESPONSE")
        print(f"  Maximum raw DN (any pixel/read): {pixstats['raw_max_dn']:.2f} DN")
        print(f"  Maximum raw DN (last read):      {pixstats['last_raw_max_dn']:.2f} DN")
        print(f"  Maximum accumulated (any pixel): {pixstats['max_accum_dn']:.2f} DN")
        print(f"  Maximum accumulated (last read): {pixstats['last_max_accum_dn']:.2f} DN")
        print(
            f"  Max-accum pixel:                  x={pixstats['max_accum_x']} "
            f"y={pixstats['max_accum_y']} frame_index={pixstats['max_accum_frame']}"
        )
        print(
            f"    reference={pixstats['max_accum_reference_dn']:.2f} DN, "
            f"raw={pixstats['max_accum_raw_dn']:.2f} DN"
        )
        p = pixstats['per_pixel_max_percentiles']
        print(
            "  Per-pixel maximum accumulation:  "
            f"median={p[50]:.2f}, p90={p[90]:.2f}, p95={p[95]:.2f}, "
            f"p99={p[99]:.2f}, max={p[100]:.2f} DN"
        )
        if effective_adc_max is not None:
            print(f"  ADC rail:                         {effective_adc_max:.0f} DN")
            print(
                f"  Pixels ever reaching ADC rail:    {pixstats['adc_hit_pixels']} "
                f"({100*pixstats['adc_hit_fraction']:.3f}%)"
            )
            print(
                f"  Pixels at ADC rail on last read:  {pixstats['adc_hit_pixels_last']} "
                f"({100*pixstats['adc_hit_fraction_last']:.3f}%)"
            )
            if pixstats['adc_first_frame'] is not None:
                print(f"  First ADC-hit array index:         {pixstats['adc_first_frame']}")
        elif a.adc_max is None:
            print("  ADC rail:                         not specified / not visibly reached")

    print()


if __name__ == "__main__":
    main()

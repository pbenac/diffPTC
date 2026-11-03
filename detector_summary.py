#!/usr/bin/env python3
"""
Compact final detector-characterization summary table.

Usage:
    python3 detector_summary.py config.yaml

Outputs:
    <dataset>_detector_summary.csv
    <dataset>_detector_summary.txt

Columns:
    Mode
    Read-noise (DN)
    Reset-noise (DN)
    Bias level (median) (DN)
    5% non-linearity threshold (DN)
    10% non-linearity threshold (DN)
    Hard saturate threshold (DN)
    Gain (e-/DN)
    Dynamic Range 5% (e-)
    Dynamic Range 10% (e-)
    Read-noise (e-)
    Reset-noise (e-)

Conventions:
- Read/reset noise follow dark_noise.py.
- Bias follows bias_level.py (trimmed median of read index 1 by default).
- Well-depth uses only bright pixels: reference read < ADC rail AND final read >= ADC rail.
- 5%, 10%, and hard-saturate thresholds are reported in detector DN (no bias subtraction).
- Dynamic range (e-) = (raw threshold - median bias) * gain.
- Hard saturate uses the slope-cutoff criterion.
- Gain is taken from the mode config when present; otherwise a PTC fallback is calculated.
  The fallback matches gain.py (LLM-generated note): it honors analysis.first_read/last_read
  (per-mode values override the top-level analysis block) and skips corrupt/zero frames.

Optional YAML additions:
summary:
  use_roi: roi
  reference_read_index: 1
  slope_start_frame: 3
  slope_stop_frame: 6
  consecutive_points: 2
  min_signal_fraction: 0.10
  slope_cutoff: 0.05
  include_pre_read_signal: true
  pre_read_frames: 1.0
  bias_section: illuminated
  bias_read_index: 1
  trim_low_percentile: 1.0
  trim_high_percentile: 99.0
  dark_section: dark
  channel_width: 512
  # well_depth_ramp: 12345
"""

import csv
import itertools
import os
import sys
import warnings

import numpy as np
import yaml
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

warnings.filterwarnings("ignore")


def truthy(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def prefix_for_head(head):
    return "ss" if str(head).upper() == "IFS" else "si"


def data_directory(dataset):
    p = dataset["datapath"]
    return os.path.join(p, "ql_redux") if truthy(dataset.get("quicklook", False)) else p


def obs_filename(dataset, n):
    suffix = "_reads" if truthy(dataset.get("quicklook", False)) else ""
    return f"{prefix_for_head(dataset['head'])}{dataset['date']}_{int(n):05d}{suffix}.fits"


def fits_path(dataset, n):
    return os.path.join(data_directory(dataset), obs_filename(dataset, n))


def parse_ramps(mode_cfg, section="illuminated"):
    cfg = mode_cfg.get(section, {})
    if section == "dark" and not cfg:
        cfg = mode_cfg.get("darks", {})

    if "ramps" in cfg:
        return [int(x) for x in cfg["ramps"]]
    if "ramp_range" in cfg:
        a, b = cfg["ramp_range"]
        return list(range(int(a), int(b)))
    if "range" in cfg:
        a, b = cfg["range"]
        return list(range(int(a), int(b)))
    if "start" in cfg and "stop" in cfg:
        return list(range(int(cfg["start"]), int(cfg["stop"])))
    if "n1" in cfg and "n2" in cfg:
        return [int(cfg["n1"]), int(cfg["n2"])]
    return []


def get_gain(mode_cfg):
    for key in ("gain", "conversion_gain", "gain_e_per_dn"):
        if mode_cfg.get(key) is not None:
            return float(mode_cfg[key])
    return None


def get_adc_max(mode_cfg):
    if mode_cfg.get("adc_max") is not None:
        return float(mode_cfg["adc_max"])
    name = str(mode_cfg.get("name", "")).lower()
    if "fast" in name:
        return 4092.0
    if "slow" in name:
        return 65535.0
    return None


def load_cube_roi(dataset, n, X0, X1, Y0, Y1):
    with fits.open(fits_path(dataset, n), memmap=True, do_not_scale_image_data=True) as h:
        bz = h[0].header.get("BZERO", 0)
        return np.asarray(h[0].data[:, Y0:Y1, X0:X1], dtype=np.float64) + bz


# LLM-generated (this function): read one read of the ROI; None if j is past the end of the ramp
def load_read_roi(dataset, n, j, X0, X1, Y0, Y1):
    with fits.open(fits_path(dataset, n), memmap=True, do_not_scale_image_data=True) as h:
        bz = h[0].header.get("BZERO", 0)
        data = h[0].data
        if j >= data.shape[0]:
            return None
        return np.asarray(data[j, Y0:Y1, X0:X1], dtype=np.float64) + bz


# LLM-generated (this function): read reads 0..last_read of the ROI; also return the ramp's read count
def load_cube_reads(dataset, n, X0, X1, Y0, Y1, last_read=None):
    with fits.open(fits_path(dataset, n), memmap=True, do_not_scale_image_data=True) as h:
        bz = h[0].header.get("BZERO", 0)
        data = h[0].data
        nf = data.shape[0]
        stop = None if last_read is None else max(int(last_read) + 1, 2)
        return np.asarray(data[:stop, Y0:Y1, X0:X1], dtype=np.float64) + bz, nf


def load_full_read(dataset, n, j):
    with fits.open(fits_path(dataset, n), memmap=True, do_not_scale_image_data=True) as h:
        bz = h[0].header.get("BZERO", 0)
        return np.asarray(h[0].data[j], dtype=np.float64) + bz


def fmt(x, nd=3):
    return "nan" if not np.isfinite(x) else f"{x:.{nd}f}"


# ---------- Bias ----------

def trimmed_median(image, low=1.0, high=99.0):
    a = np.asarray(image, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if not len(a):
        return np.nan
    lo, hi = np.percentile(a, [low, high])
    a = a[(a >= lo) & (a <= hi)]
    return float(np.median(a)) if len(a) else np.nan


def bias_level(dataset, mode_cfg, X0, X1, Y0, Y1, cfg):
    ramps = parse_ramps(mode_cfg, cfg.get("bias_section", "illuminated"))
    j = int(cfg.get("bias_read_index", 1))
    low = float(cfg.get("trim_low_percentile", 1.0))
    high = float(cfg.get("trim_high_percentile", 99.0))

    vals = []
    for n in ramps:
        image = load_read_roi(dataset, n, j, X0, X1, Y0, Y1)  # LLM-generated: read j only
        if image is not None:  # LLM-generated
            vals.append(trimmed_median(image, low, high))  # LLM-generated

    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if len(vals) else np.nan


# ---------- Read + reset noise ----------

def correct_noise_image(d, channel_width=512):
    d = np.asarray(d, dtype=float).copy()
    if channel_width and channel_width > 0:
        for x0 in range(0, d.shape[1], channel_width):
            x1 = min(x0 + channel_width, d.shape[1])
            d[:, x0:x1] -= np.median(d[:, x0:x1])
    d -= np.median(d, axis=1, keepdims=True)
    return d


def clipped_std_roi(d, X0, X1, Y0, Y1):
    return float(sigma_clipped_stats(d[Y0:Y1, X0:X1], sigma=3.0, maxiters=5)[2])


def noise_values(dataset, mode_cfg, X0, X1, Y0, Y1, cfg):
    ramps = parse_ramps(mode_cfg, cfg.get("dark_section", "dark"))
    channel_width = int(cfg.get("channel_width", 512))
    if not ramps:
        return np.nan, np.nan

    diff_stds = []
    for n in ramps:
        f1 = load_full_read(dataset, n, 1)
        f2 = load_full_read(dataset, n, 2)
        dc = correct_noise_image(f2 - f1, channel_width)
        diff_stds.append(clipped_std_roi(dc, X0, X1, Y0, Y1))

    q_var = 2.0 / 12.0
    med_diff_std = float(np.median(diff_stds))
    read_var = med_diff_std**2 - q_var
    read_dn = np.sqrt(read_var / 2.0) if read_var > 0 else np.nan
    read_noq_dn = med_diff_std / np.sqrt(2.0)

    reset_diff_vars = []
    for a, b in zip(ramps[::2], ramps[1::2]):
        a1 = load_full_read(dataset, a, 1)
        b1 = load_full_read(dataset, b, 1)
        dc = correct_noise_image(a1 - b1, channel_width)
        reset_diff_vars.append(clipped_std_roi(dc, X0, X1, Y0, Y1) ** 2)

    if reset_diff_vars:
        var_ab = float(np.median(reset_diff_vars))
        reset_var = var_ab / 2.0 - read_noq_dn**2
        reset_dn = np.sqrt(reset_var) if reset_var > 0 else np.nan
    else:
        reset_dn = np.nan

    return float(read_dn), float(reset_dn)


# ---------- Well depth / nonlinearity ----------

def first_sustained_hit(mask, n):
    mask = np.asarray(mask, dtype=bool)
    n = max(1, int(n))
    if n == 1:
        hit = np.flatnonzero(mask)
        return int(hit[0]) if len(hit) else None
    if len(mask) < n:
        return None
    run = np.convolve(mask.astype(int), np.ones(n, dtype=int), mode="valid")
    hit = np.flatnonzero(run >= n)
    return int(hit[0]) if len(hit) else None


def interp_y_at_crossing(criterion, y, target, k):
    if k is None:
        return np.nan
    if k <= 0:
        return float(y[k])
    k0, k1 = k - 1, k
    c0, c1 = criterion[k0], criterion[k1]
    if np.isfinite(c0) and np.isfinite(c1) and c1 != c0:
        f = float(np.clip((target - c0) / (c1 - c0), 0.0, 1.0))
    else:
        f = 1.0
    return float(y[k0] + f * (y[k1] - y[k0]))


def well_depth_one_ramp(dataset, mode_cfg, n, X0, X1, Y0, Y1, cfg):
    cube = load_cube_roi(dataset, n, X0, X1, Y0, Y1)
    nf = cube.shape[0]

    ref = int(cfg.get("reference_read_index", 1))
    start_frame = int(cfg.get("slope_start_frame", 3))
    stop_frame = int(cfg.get("slope_stop_frame", 6))
    consecutive = int(cfg.get("consecutive_points", 2))
    min_frac = float(cfg.get("min_signal_fraction", 0.10))
    slope_cutoff = float(cfg.get("slope_cutoff", 0.05))
    include_pre = truthy(cfg.get("include_pre_read_signal", False))
    pre_frames = float(cfg.get("pre_read_frames", 1.0))
    adc = get_adc_max(mode_cfg)

    if adc is None:
        raise ValueError(f"{mode_cfg.get('name', 'mode')}: adc_max required for bright-pixel selection")
    if ref >= nf:
        raise ValueError(f"reference_read_index={ref} outside ramp length {nf}")

    ref_im = cube[ref]
    last_im = cube[-1]
    bright = np.isfinite(ref_im) & np.isfinite(last_im) & (ref_im < adc) & (last_im >= adc)
    npix = int(np.count_nonzero(bright))
    if npix == 0:
        return dict(n_pixels=0, wd5=np.nan, wd10=np.nan, hard=np.nan)

    raw = np.array([np.median(cube[j][bright]) for j in range(nf)], dtype=float)
    frames = np.arange(1, nf + 1, dtype=float)
    signal = raw - raw[ref]

    i0 = max(0, start_frame - 1)
    i1 = min(nf - 1, stop_frame - 1)
    if i1 <= i0:
        raise ValueError("slope_stop_frame must be after slope_start_frame")

    slope, intercept = np.polyfit(frames[i0:i1 + 1], signal[i0:i1 + 1], 1)
    if not np.isfinite(slope) or slope <= 0:
        return dict(n_pixels=npix, wd5=np.nan, wd10=np.nan, hard=np.nan)

    line = slope * frames + intercept
    pre_dn = slope * pre_frames if include_pre else 0.0
    signal_total = signal + pre_dn
    line_total = line + pre_dn

    denom = np.where(line_total > 0, line_total, np.nan)
    deviation = 1.0 - signal_total / denom

    max_signal = np.nanmax(signal_total)
    above_min = signal_total >= min_frac * max_signal
    after_fit = frames > frames[i1]
    search = above_min & after_fit

    out = {"n_pixels": npix}
    for pct, key in ((0.05, "wd5"), (0.10, "wd10")):
        mask = search & np.isfinite(deviation) & (deviation >= pct)
        k = first_sustained_hit(mask, consecutive)
        out[key] = interp_y_at_crossing(deviation, raw, pct, k)

    local = np.gradient(signal_total, frames)
    slope_fraction = local / slope
    mask = search & np.isfinite(slope_fraction) & (slope_fraction <= slope_cutoff)
    k = first_sustained_hit(mask, consecutive)
    out["hard"] = interp_y_at_crossing(slope_fraction, raw, slope_cutoff, k)
    return out


def well_depth_values(dataset, mode_cfg, X0, X1, Y0, Y1, cfg):
    ramps = parse_ramps(mode_cfg, "illuminated")
    override = mode_cfg.get("well_depth_ramp", cfg.get("well_depth_ramp"))
    if override is not None:
        ramps = [int(override)]

    vals = [well_depth_one_ramp(dataset, mode_cfg, n, X0, X1, Y0, Y1, cfg) for n in ramps]

    def med(key):
        a = np.asarray([v[key] for v in vals], dtype=float)
        a = a[np.isfinite(a)]
        return float(np.median(a)) if len(a) else np.nan

    return med("wd5"), med("wd10"), med("hard")


# ---------- Gain fallback ----------

def row_detrend(img):
    return img - np.median(img, axis=1, keepdims=True)


def make_pairs(mode_cfg, ramps):
    icfg = mode_cfg.get("illuminated", {})
    if "pairs" in icfg:
        return [(int(a), int(b)) for a, b in icfg["pairs"]]
    if len(ramps) < 2:
        return []
    pairing = str(icfg.get("pairing", "first_two" if len(ramps) == 2 else "all_pairs")).lower()
    if pairing in ("first_two", "first", "first_pair"):
        return [(ramps[0], ramps[1])]
    if pairing in ("adjacent", "consecutive"):
        return list(zip(ramps[:-1], ramps[1:]))
    if pairing in ("all_pairs", "all", "combinations"):
        return list(itertools.combinations(ramps, 2))
    raise ValueError(f"Unknown pairing {pairing!r}")


# LLM-generated (this function): same conversion as gain.py
def as_optional_int(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null"):
        return None
    return int(value)


# LLM-generated (this function): same read bounds and errors as gain.py
def selected_read_bounds(nf, first_read=None, last_read=None):
    first = 2 if first_read is None else int(first_read)
    last = nf - 1 if last_read is None else int(last_read)
    if first < 2:
        raise ValueError(
            f"first_read={first} is invalid: the PTC reference is read index 1, "
            "so first_read must be >= 2."
        )
    if last < first:
        raise ValueError(f"last_read={last} must be >= first_read={first}.")
    if first >= nf:
        raise ValueError(f"first_read={first} is outside a ramp with {nf} reads.")
    return first, min(last, nf - 1)


def calculate_gain(dataset, mode_cfg, X0, X1, Y0, Y1, analysis_cfg=None):  # LLM-generated: analysis_cfg holds read limits
    ramps = parse_ramps(mode_cfg, "illuminated")
    pairs = make_pairs(mode_cfg, ramps)
    if not pairs:
        return np.nan

    detrend = truthy(mode_cfg.get("detrend", False))
    fixed_xmax = mode_cfg.get("fixed_xmax")
    fixed_xmax = None if fixed_xmax in (None, "", "None", "null") else float(fixed_xmax)
    # LLM-generated: per-mode read limits override the top-level analysis block, as in gain.py
    analysis_cfg = analysis_cfg or {}  # LLM-generated
    mode_analysis = mode_cfg.get("analysis", {}) or {}  # LLM-generated
    first_read = as_optional_int(mode_analysis.get("first_read", analysis_cfg.get("first_read")))  # LLM-generated
    last_read = as_optional_int(mode_analysis.get("last_read", analysis_cfg.get("last_read")))  # LLM-generated
    S, V = [], []

    for n1, n2 in pairs:
        A, nf_a = load_cube_reads(dataset, n1, X0, X1, Y0, Y1, last_read)  # LLM-generated: load only reads needed
        B, nf_b = load_cube_reads(dataset, n2, X0, X1, Y0, Y1, last_read)  # LLM-generated: load only reads needed
        nf = min(nf_a, nf_b)  # LLM-generated: full read counts
        if nf < 3:
            continue
        try:  # LLM-generated
            read_first, read_last = selected_read_bounds(nf, first_read, last_read)  # LLM-generated
        except ValueError as exc:  # LLM-generated: report it and leave gain as nan, like gain.py skipping the mode
            print(f"   {mode_cfg.get('name', 'mode')}: gain fallback failed: {exc}")  # LLM-generated
            return np.nan  # LLM-generated
        A1, B1 = A[1], B[1]
        mA1, mB1 = np.median(A1), np.median(B1)  # LLM-generated
        nbad = 0  # LLM-generated
        for j in range(read_first, read_last + 1):  # LLM-generated
            # LLM-generated: skip corrupt/truncated frames, same test as gain.py
            if np.median(A[j]) < 0.5 * mA1 or np.median(B[j]) < 0.5 * mB1:  # LLM-generated
                nbad += 1  # LLM-generated
                continue  # LLM-generated
            da = A[j] - A1
            db = B[j] - B1
            d = db - da
            if detrend:
                d = row_detrend(d)
            sd = sigma_clipped_stats(d, sigma=3.0, maxiters=5)[2]
            s = 0.5 * (np.median(da) + np.median(db))
            v = sd**2 / 2.0
            if np.isfinite(s) and np.isfinite(v):
                S.append(s)
                V.append(v)
        if nbad:  # LLM-generated
            print(f"   [{n1}/{n2}: skipped {nbad} corrupt/zero frames]")  # LLM-generated

    S = np.asarray(S, dtype=float)
    V = np.asarray(V, dtype=float)
    if len(S) < 2:
        return np.nan

    smax = float(np.max(S)) if fixed_xmax is None else fixed_xmax
    good = np.isfinite(S) & np.isfinite(V) & (S > 0.05 * smax) & (S < 0.90 * smax)
    if np.count_nonzero(good) < 2:
        return np.nan

    slope, _ = np.polyfit(S[good], V[good], 1)
    return float(1.0 / slope) if slope > 0 else np.nan


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 detector_summary.py config.yaml")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        config = yaml.safe_load(f)

    dataset = config["dataset"]
    scfg = config.get("summary", {})
    roi_key = scfg.get("use_roi", "roi")
    roi = config[roi_key]
    X0, X1 = int(roi["x_start"]), int(roi["x_end"])
    Y0, Y1 = int(roi["y_start"]), int(roi["y_end"])

    outpath = dataset["outpath"]
    setname = dataset["name"]
    os.makedirs(outpath, exist_ok=True)
    rows = []

    for mode_key in dataset["modes"]:
        mode_cfg = config[mode_key]
        name = mode_cfg.get("name", mode_key)
        print(f"Analyzing {name} ...")

        gain = get_gain(mode_cfg)
        if gain is None:
            gain = calculate_gain(dataset, mode_cfg, X0, X1, Y0, Y1, config.get("analysis", {}))  # LLM-generated

        read_dn, reset_dn = noise_values(dataset, mode_cfg, X0, X1, Y0, Y1, scfg)
        bias_dn = bias_level(dataset, mode_cfg, X0, X1, Y0, Y1, scfg)
        wd5_raw, wd10_raw, hard_raw = well_depth_values(dataset, mode_cfg, X0, X1, Y0, Y1, scfg)

        dr5_e = (wd5_raw - bias_dn) * gain if np.isfinite(wd5_raw) and np.isfinite(bias_dn) and np.isfinite(gain) else np.nan
        dr10_e = (wd10_raw - bias_dn) * gain if np.isfinite(wd10_raw) and np.isfinite(bias_dn) and np.isfinite(gain) else np.nan
        read_e = read_dn * gain if np.isfinite(read_dn) and np.isfinite(gain) else np.nan
        reset_e = reset_dn * gain if np.isfinite(reset_dn) and np.isfinite(gain) else np.nan

        rows.append({
            "Mode": name,
            "Read-noise (DN)": read_dn,
            "Reset-noise (DN)": reset_dn,
            "Bias level (median) (DN)": bias_dn,
            "5% non-linearity threshold (DN)": wd5_raw,
            "10% non-linearity threshold (DN)": wd10_raw,
            "Hard saturate threshold (DN)": hard_raw,
            "Gain (e-/DN)": gain,
            "Dynamic Range 5% (e-)": dr5_e,
            "Dynamic Range 10% (e-)": dr10_e,
            "Read-noise (e-)": read_e,
            "Reset-noise (e-)": reset_e,
        })

    if not rows:
        raise RuntimeError("No modes found in dataset.modes")

    csv_fn = os.path.join(outpath, f"{setname}_detector_summary.csv")
    txt_fn = os.path.join(outpath, f"{setname}_detector_summary.txt")
    fields = list(rows[0].keys())

    with open(csv_fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()})

    header = (
        f"{'Mode':20s} {'RN DN':>9s} {'Reset DN':>9s} {'Bias DN':>10s} "
        f"{'5% DN':>10s} {'10% DN':>10s} {'Hard DN':>10s} {'Gain':>9s} "
        f"{'DR5 e-':>12s} {'DR10 e-':>12s} {'RN e-':>10s} {'Reset e-':>10s}"
    )

    with open(txt_fn, "w") as f:
        print(header)
        f.write(header + "\n")
        for r in rows:
            line = (
                f"{r['Mode'][:20]:20s} "
                f"{fmt(r['Read-noise (DN)'],2):>9s} "
                f"{fmt(r['Reset-noise (DN)'],2):>9s} "
                f"{fmt(r['Bias level (median) (DN)'],1):>10s} "
                f"{fmt(r['5% non-linearity threshold (DN)'],1):>10s} "
                f"{fmt(r['10% non-linearity threshold (DN)'],1):>10s} "
                f"{fmt(r['Hard saturate threshold (DN)'],1):>10s} "
                f"{fmt(r['Gain (e-/DN)'],3):>9s} "
                f"{fmt(r['Dynamic Range 5% (e-)'],1):>12s} "
                f"{fmt(r['Dynamic Range 10% (e-)'],1):>12s} "
                f"{fmt(r['Read-noise (e-)'],1):>10s} "
                f"{fmt(r['Reset-noise (e-)'],1):>10s}"
            )
            print(line)
            f.write(line + "\n")

    print(f"\nWrote:\n  {csv_fn}\n  {txt_fn}")


if __name__ == "__main__":
    main()

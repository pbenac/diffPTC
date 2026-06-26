#!/usr/bin/env python3
"""
Read noise and reset (kTC) noise from dark ramps, per readout mode.

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
from astropy.stats import sigma_clipped_stats
import yaml


def P(log, *a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.write(s + "\n")
    log.flush()


def data_directory(datapath, quicklook=False):
    """Return the directory containing FITS files for normal or quicklook products."""
    return os.path.join(datapath, "ql_redux") if quicklook else datapath


def obs_filename(head, date, n, quicklook=False):
    pref = "ss" if head == "IFS" else "si"
    suffix = "_reads" if quicklook else ""
    return f"{pref}{date}_{int(n):05d}{suffix}.fits"


def frame(head, date, n, datapath, j, quicklook=False):
    fn = os.path.join(data_directory(datapath, quicklook), obs_filename(head, date, n, quicklook=quicklook))
    with fits.open(fn, memmap=True, do_not_scale_image_data=True) as h:
        bz = h[0].header.get("BZERO", 0)
        return np.asarray(h[0].data[j]).astype(np.float64) + bz


def correct(d, channel_width=512):
    """Subtract per-channel bias and then per-row median."""
    d = d.copy()
    if channel_width and channel_width > 0:
        nchan = d.shape[1] // channel_width
        for c in range(nchan):
            sl = slice(c * channel_width, (c + 1) * channel_width)
            d[:, sl] -= np.median(d[:, sl])
    d -= np.median(d, axis=1, keepdims=True)
    return d


def rms(d, X0, X1, Y0, Y1):
    """Sigma-clipped std for analysis region and full frame."""
    reg = sigma_clipped_stats(d[Y0:Y1, X0:X1], sigma=3.0, maxiters=5)[2]
    full = sigma_clipped_stats(d, sigma=3.0, maxiters=5)[2]
    return reg, full


def parse_ramps(mode_cfg):
    dark_cfg = mode_cfg.get("dark", mode_cfg.get("darks", {}))
    if "ramps" in dark_cfg:
        return list(dark_cfg["ramps"])
    if "ramp_range" in dark_cfg:
        start, stop = dark_cfg["ramp_range"]
        return list(range(int(start), int(stop)))
    if "range" in dark_cfg:
        start, stop = dark_cfg["range"]
        return list(range(int(start), int(stop)))
    if "start" in dark_cfg and "stop" in dark_cfg:
        return list(range(int(dark_cfg["start"]), int(dark_cfg["stop"])))
    return []


def get_gain(mode_cfg):
    for key in ("gain", "conversion_gain", "gain_e_per_dn"):
        if key in mode_cfg:
            return float(mode_cfg[key])
    return None


def read_noise_for_mode(head, date, datapath, ramps, gain, X0, X1, Y0, Y1, channel_width, quicklook=False):
    raw_reg = []
    cor_reg = []
    cor_full = []

    for n in ramps:
        d = frame(head, date, n, datapath, 2, quicklook=quicklook) - frame(head, date, n, datapath, 1, quicklook=quicklook)
        raw_reg.append(rms(d, X0, X1, Y0, Y1)[0])
        dc = correct(d, channel_width=channel_width)
        r, f = rms(dc, X0, X1, Y0, Y1)
        cor_reg.append(r)
        cor_full.append(f)

    q_var = 2.0 / 12.0  # quantization variance of a 2-read difference, DN^2

    def rn_dn(stdarr):
        v = np.median(stdarr) ** 2 - q_var
        return np.sqrt(v / 2.0) if v > 0 else float("nan")

    sr_raw = np.median(raw_reg) / np.sqrt(2) * gain
    sr_creg_dn = rn_dn(cor_reg)
    sr_creg = sr_creg_dn * gain
    sr_cful = rn_dn(cor_full) * gain
    sr_creg_noq_dn = np.median(cor_reg) / np.sqrt(2)

    rstvar = []
    for a, b in zip(ramps[::2], ramps[1::2]):
        d = frame(head, date, a, datapath, 1, quicklook=quicklook) - frame(head, date, b, datapath, 1, quicklook=quicklook)
        dc = correct(d, channel_width=channel_width)
        rstvar.append(rms(dc, X0, X1, Y0, Y1)[0] ** 2)

    var_ab = np.median(rstvar) if rstvar else float("nan")
    reset_var_dn = var_ab / 2.0 - sr_creg_noq_dn ** 2
    reset_dn = np.sqrt(reset_var_dn) if reset_var_dn > 0 else float("nan")
    reset_e = reset_dn * gain

    return dict(
        sr_creg=sr_creg,
        sr_cful=sr_cful,
        sr_raw=sr_raw,
        reset_e=reset_e,
        sread_dn=sr_creg_dn,
        noq_dn=sr_creg_noq_dn,
        var_ab=var_ab,
        g=gain,
        nramp=len(ramps),
    )


def make_diagnostic_plot(summary, config, head, date, datapath, outpath, setname, X0, X1, Y0, Y1, channel_width, quicklook=False):
    if not summary:
        return

    dark_analysis = config.get("dark_analysis", {})
    diagnostic_mode = dark_analysis.get("diagnostic_mode", next(iter(summary)))
    if diagnostic_mode not in config:
        return
    ramps = parse_ramps(config[diagnostic_mode])
    if not ramps:
        return

    n = ramps[0]
    d = frame(head, date, n, datapath, 2, quicklook=quicklook) - frame(head, date, n, datapath, 1, quicklook=quicklook)
    dc = correct(d, channel_width=channel_width)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for a, im, ti in [
        (ax[0], d, "raw F2-F1"),
        (ax[1], dc, "after channel + row-median subtraction"),
    ]:
        m, _, s = sigma_clipped_stats(im, sigma=3)
        h = a.imshow(im, origin="lower", cmap="coolwarm", vmin=m - 4 * s, vmax=m + 4 * s)
        a.set_title(f"{config[diagnostic_mode]['name']} dark: {ti}", fontsize=9)
        plt.colorbar(h, ax=a, shrink=.8, label="DN")
        if channel_width and channel_width > 0:
            for c in range(1, d.shape[1] // channel_width):
                a.axvline(c * channel_width, color="k", lw=.4, ls=":")

    ax[2].plot(np.median(d, axis=0), lw=.6, label="raw")
    ax[2].plot(np.median(dc, axis=0), lw=.6, label="corrected")
    if channel_width and channel_width > 0:
        for c in range(1, d.shape[1] // channel_width):
            ax[2].axvline(c * channel_width, color="k", lw=.4, ls=":")
    ax[2].set_xlabel("column x")
    ax[2].set_ylabel("column median [DN]")
    ax[2].set_title("channel structure removed")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outpath, f"{setname}_darks_correction.png"), dpi=110)
    plt.close(fig)


def make_summary_plot(summary, outpath, setname):
    if not summary:
        return
    names = list(summary.keys())
    rd = [summary[n]["sr_creg"] for n in names]
    rs = [summary[n]["reset_e"] for n in names]

    fig, ax = plt.subplots(figsize=(7, 4.8))
    x = np.arange(len(names))
    w = 0.38
    ax.bar(x - w / 2, rd, w, label="read noise (CDS, corrected)")
    ax.bar(x + w / 2, rs, w, label="reset noise (kTC)")
    for i in range(len(names)):
        ax.text(x[i] - w / 2, rd[i], f"{rd[i]:.0f}", ha="center", va="bottom", fontsize=9)
        if np.isfinite(rs[i]):
            ax.text(x[i] + w / 2, rs[i], f"{rs[i]:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("noise [e-]")
    ax.set_title("Read noise & reset noise by readout mode (darks)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outpath, f"{setname}_darks_noise_summary.png"), dpi=110)
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("Error: No yaml configuration file provided")
        sys.exit(1)

    yamlfile = sys.argv[1]
    with open(yamlfile) as f:
        config = yaml.safe_load(f)

    dataset = config["dataset"]
    head = dataset["head"]
    date = dataset["date"]
    datapath = dataset["datapath"]
    quicklook = bool(dataset.get("quicklook", False))
    outpath = dataset["outpath"]
    setname = dataset["name"]
    modes = dataset["modes"]

    os.makedirs(outpath, exist_ok=True)

    X0 = config["dark_roi"]["x_start"]
    X1 = config["dark_roi"]["x_end"]
    Y0 = config["dark_roi"]["y_start"]
    Y1 = config["dark_roi"]["y_end"]

    dark_analysis = config.get("dark_analysis", {})
    channel_width = int(dark_analysis.get("channel_width", 512))

    logname = f"{setname}_darks_noise.txt"
    with open(os.path.join(outpath, logname), "w") as log:
        P(log, "=== Read noise & reset noise from darks ===")
        P(log, f"region x[{X0}:{X1}] y[{Y0}:{Y1}]; channels = {channel_width}-col vertical stripes")
        P(log, f"quicklook={quicklook}; data directory={data_directory(datapath, quicklook)}\n")
        P(log, "read noise = std(F2-F1)/sqrt2 (within-ramp CDS), channel+row corrected, Sheppard quantization-corrected.")
        P(log, "reset noise = sqrt( Var(A1-B1)/2 - read^2 ) across independent ramps.\n")
        P(log, f"{'mode':9s} {'gain':>7s} {'ramps':>6s} | {'READ NOISE (e-)':>22s} | {'RESET NOISE (e-)':>16s}")
        P(log, f"{'':9s} {'e-/DN':>7s} {'':>6s} | {'rawCDS':>7s} {'corr(reg)':>9s} {'corr(full)':>10s} | {'corr(reg)':>10s}")

        summary = {}
        for mode in modes:
            mode_cfg = config[mode]
            name = mode_cfg["name"]
            ramps = parse_ramps(mode_cfg)
            gain = get_gain(mode_cfg)

            if not ramps:
                P(log, f"{name:9s} -- no dark ramps configured; skipped --")
                continue
            if gain is None:
                P(log, f"{name:9s} -- no gain configured; skipped --")
                continue
            if len(ramps) < 2:
                P(log, f"{name:9s} -- need at least 2 dark ramps; skipped --")
                continue

            s = read_noise_for_mode(head, date, datapath, ramps, gain, X0, X1, Y0, Y1, channel_width, quicklook=quicklook)
            summary[name] = s
            P(log, f"{name:9s} {gain:7.3f} {len(ramps):6d} | {s['sr_raw']:7.1f} {s['sr_creg']:9.1f} {s['sr_cful']:10.1f} | {s['reset_e']:10.1f}")

        P(log, "\nDetails (corrected, region):")
        for name, s in summary.items():
            note = (f"reset noise {s['reset_e']:.1f} e-" if not np.isnan(s["reset_e"])
                    else "reset noise consistent with ~0 (Var/2 below read-noise floor)")
            P(log, f"  {name}: read noise {s['sr_creg']:.1f} e- (= {s['sread_dn']:.2f} DN); raw(uncorrected) {s['sr_raw']:.1f} e-; {note}")

    make_diagnostic_plot(summary, config, head, date, datapath, outpath, setname, X0, X1, Y0, Y1, channel_width, quicklook=quicklook)
    make_summary_plot(summary, outpath, setname)
    print("DONE")


if __name__ == "__main__":
    main()

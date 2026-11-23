#!/usr/bin/env python3
# LLM-generated (Claude). Variant of dark_noise.py; dark_noise.py itself is unchanged.
"""
Measured read noise (digitization noise INCLUDED) and reset noise from dark ramps.

Same inputs, region, channel/row correction and sigma clipping as dark_noise.py.
The only difference: the read noise is reported as measured. The 2/12 DN^2
quantization (digitization) variance is NOT subtracted.

    python dark_noise_measured.py datasets/<...>.yaml

Reported per mode (all from the channel+row corrected F2-F1 difference, median over ramps):
  CDS read noise      = std(F2-F1)                 <- "CDS read noise (incl. digitization)"
  per-read read noise = std(F2-F1) / sqrt(2)
  reset noise         = sqrt( Var(A1-B1)/2 - (per-read read noise)^2 ), identical to
                        dark_noise.py. Var(A1-B1)/2 and the subtracted read term each
                        carry the same 1/12 DN^2, so digitization cancels here either way.

For reference only (e.g. for the PSR text), the log also lists the digitization-
subtracted values that dark_noise.py reports:
  CDS  = sqrt(std^2 - 2/12)         per-read = sqrt((std^2 - 2/12)/2)  (= dark_noise.py "corr(reg)")
"""
import os
import sys

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dark_noise as dn  # loaders, correction and clipping shared with dark_noise.py
import data_paths
import gui_io
import run_dirs

Q_VAR_CDS = 2.0 / 12.0  # quantization variance of a 2-read difference, DN^2 (reference only)


def measured_noise_for_mode(head, date, datapath, ramps, gain, X0, X1, Y0, Y1, channel_width,
                            quicklook=False, gui=False):
    raw_reg = []
    cor_reg = []
    cor_full = []

    for n in ramps:
        d = (dn.frame(head, date, n, datapath, 2, quicklook=quicklook, gui=gui)
             - dn.frame(head, date, n, datapath, 1, quicklook=quicklook, gui=gui))
        raw_reg.append(dn.rms(d, X0, X1, Y0, Y1)[0])
        dc = dn.correct(d, channel_width=channel_width)
        r, f = dn.rms(dc, X0, X1, Y0, Y1)
        cor_reg.append(r)
        cor_full.append(f)

    cds_dn = np.median(cor_reg)                    # measured CDS noise, digitization included
    read_dn = np.median(cor_reg) / np.sqrt(2)      # same expression as dark_noise.py noq_dn
    cds_full_dn = np.median(cor_full)
    cds_raw_dn = np.median(raw_reg)                # no channel/row correction

    # Reset noise: same arithmetic, in the same order, as dark_noise.read_noise_for_mode.
    rstvar = []
    for a, b in zip(ramps[::2], ramps[1::2]):
        d = (dn.frame(head, date, a, datapath, 1, quicklook=quicklook, gui=gui)
             - dn.frame(head, date, b, datapath, 1, quicklook=quicklook, gui=gui))
        dc = dn.correct(d, channel_width=channel_width)
        rstvar.append(dn.rms(dc, X0, X1, Y0, Y1)[0] ** 2)

    var_ab = np.median(rstvar) if rstvar else float("nan")
    reset_var_dn = var_ab / 2.0 - read_dn ** 2
    reset_dn = np.sqrt(reset_var_dn) if reset_var_dn > 0 else float("nan")

    # Reference only: digitization-subtracted values (what dark_noise.py reports).
    v = cds_dn ** 2 - Q_VAR_CDS
    cds_noq_dn = np.sqrt(v) if v > 0 else float("nan")
    read_noq_dn = np.sqrt(v / 2.0) if v > 0 else float("nan")

    return dict(
        g=gain,
        nramp=len(ramps),
        cds_dn=cds_dn, cds_e=cds_dn * gain,
        read_dn=read_dn, read_e=read_dn * gain,
        cds_full_dn=cds_full_dn, cds_full_e=cds_full_dn * gain,
        cds_raw_dn=cds_raw_dn, cds_raw_e=cds_raw_dn * gain,
        reset_dn=reset_dn, reset_e=reset_dn * gain,
        var_ab=var_ab,
        cds_noq_dn=cds_noq_dn, cds_noq_e=cds_noq_dn * gain,
        read_noq_dn=read_noq_dn, read_noq_e=read_noq_dn * gain,
    )


def make_summary_plot(summary, outpath, setname):
    if not summary:
        return
    names = list(summary.keys())
    cds = [summary[n]["cds_e"] for n in names]
    rs = [summary[n]["reset_e"] for n in names]

    fig, ax = plt.subplots(figsize=(7, 4.8))
    x = np.arange(len(names))
    w = 0.38
    ax.bar(x - w / 2, cds, w, label="CDS read noise (measured, incl. digitization)")
    ax.bar(x + w / 2, rs, w, label="reset noise (kTC)")
    for i in range(len(names)):
        ax.text(x[i] - w / 2, cds[i], f"{cds[i]:.1f}", ha="center", va="bottom", fontsize=9)
        if np.isfinite(rs[i]):
            ax.text(x[i] + w / 2, rs[i], f"{rs[i]:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("noise [e-]")
    ax.set_title("Measured CDS read noise & reset noise by readout mode (darks)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outpath, f"{setname}_darks_noise_measured_summary.png"), dpi=110)
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("Error: No yaml configuration file provided")
        sys.exit(1)

    yamlfile = sys.argv[1]
    with open(yamlfile) as f:
        config = yaml.safe_load(f)

    dataset = config["dataset"]
    gui = gui_io.is_gui_mode(config)
    head = dataset.get("head", "GUI")
    date = dataset.get("date", "")
    datapath = dataset.get("datapath", "")
    quicklook = False if gui else bool(dataset.get("quicklook", False))
    config["dataset"]["outpath"] = run_dirs.output_dir(config, sys.argv[1], "dark_noise_measured")
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

    logname = f"{setname}_darks_noise_measured.txt"
    with open(os.path.join(outpath, logname), "w") as log:
        P = dn.P
        P(log, "=== Measured read noise (digitization INCLUDED) & reset noise from darks ===")
        P(log, f"region x[{X0}:{X1}] y[{Y0}:{Y1}]; channels = {channel_width}-col vertical stripes")
        if gui:
            P(log, f"GUI Live Mode=True; data root={gui_io.GUI_BASE_PATH}\n")
        else:
            P(log, f"quicklook={quicklook}; data directory={data_paths.describe(datapath, quicklook)}\n")
        P(log, "CDS read noise = std(F2-F1), channel+row corrected; quantization NOT subtracted.")
        P(log, "per-read read noise = CDS / sqrt2.")
        P(log, "reset noise = sqrt( Var(A1-B1)/2 - per-read^2 ) (as dark_noise.py; digitization cancels).\n")
        P(log, f"{'mode':9s} {'gain':>7s} {'ramps':>6s} | {'CDS READ NOISE (e-)':>26s} | "
               f"{'per-read':>9s} | {'RESET (e-)':>10s}")
        P(log, f"{'':9s} {'e-/DN':>7s} {'':>6s} | {'rawCDS':>7s} {'corr(reg)':>9s} {'corr(full)':>10s} | "
               f"{'corr(reg)':>9s} | {'corr(reg)':>10s}")

        summary = {}
        for mode in modes:
            mode_cfg = config[mode]
            name = mode_cfg["name"]
            ramps = dn.parse_ramps(mode_cfg, gui=gui)
            gain = dn.get_gain(mode_cfg)

            if not ramps:
                P(log, f"{name:9s} -- no dark ramps configured; skipped --")
                continue
            if gain is None:
                P(log, f"{name:9s} -- no gain configured; skipped --")
                continue
            if len(ramps) < 2:
                P(log, f"{name:9s} -- need at least 2 dark ramps; skipped --")
                continue

            s = measured_noise_for_mode(head, date, datapath, ramps, gain, X0, X1, Y0, Y1,
                                        channel_width, quicklook=quicklook, gui=gui)
            summary[name] = s
            P(log, f"{name:9s} {gain:7.3f} {len(ramps):6d} | {s['cds_raw_e']:7.2f} {s['cds_e']:9.2f} "
                   f"{s['cds_full_e']:10.2f} | {s['read_e']:9.2f} | {s['reset_e']:10.2f}")

        P(log, "\nDetails (corrected, region; measured = digitization included):")
        for name, s in summary.items():
            reset = (f"reset noise {s['reset_e']:.2f} e- (= {s['reset_dn']:.3f} DN)"
                     if not np.isnan(s["reset_e"])
                     else "reset noise consistent with ~0 (Var/2 below read-noise floor)")
            P(log, f"  {name}: CDS read noise {s['cds_e']:.2f} e- (= {s['cds_dn']:.3f} DN); "
                   f"per-read {s['read_e']:.2f} e- (= {s['read_dn']:.3f} DN); {reset}")

        P(log, "\nReference only, digitization (2/12 DN^2 per CDS) subtracted, as in dark_noise.py:")
        for name, s in summary.items():
            P(log, f"  {name}: CDS {s['cds_noq_e']:.2f} e- (= {s['cds_noq_dn']:.3f} DN); "
                   f"per-read {s['read_noq_e']:.2f} e- (= {s['read_noq_dn']:.3f} DN)")

    make_summary_plot(summary, outpath, setname)
    print("DONE")


if __name__ == "__main__":
    main()

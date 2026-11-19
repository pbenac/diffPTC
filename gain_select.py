#!/usr/bin/env python3
# LLM-generated (Claude). New file; gain.py is not modified.
"""
Gain PTCs for hand-picked modes, each taken from whatever dataset you chose.

gain.py fits every mode of ONE dataset YAML and draws one comparison figure. When the
reported gains come from different days (for example IMG Fast0.6 and Slow5.2 from
Aug 28 but Fast1.0 from Aug 24), there is no single figure that shows them. This
script takes a short "selection" YAML that names, for each reported value, the dataset
YAML and the mode inside it, and writes:

  <name>_<label>_ptc.png    one figure per selected mode: the PTC with its fit, each
                            pair's points in its own colour, and the per-pair gains
  <name>gain_compare.png    gain.py's own comparison figure for the selection (up to 3)
  <name>_summary.txt        source dataset, date, ramps, pairs, reads, gain, error, RN
  <name>_ptc_<label>.csv, <name>_pair_gains_<label>.csv   same format as gain.py

The numbers are computed by gain.py's own functions (gain_diff_many_pairs, fit_gain)
with the same arguments gain.py would pass, so each gain is bit-identical to running
gain.py on that dataset with the same read limits.

Selection YAML:

  name: CD6_IMG_reported           # prefix for output files and the run folder name
  outpath: results/CD6/selected/   # relative to the current directory, or absolute;
                                   # outputs go in <outpath>/<name>/ (or <name>_1/, ...
                                   # if a previous run is there; see run_dirs.py)
  # run_subdir: false              # write straight into outpath instead
  entries:
    - dataset: datasets/CD6/260828_IMG_defaults.yaml   # path to a gain.py YAML
      mode: fast06                 # the mode KEY in that YAML (not its 'name')
      label: Fast0.6               # optional; defaults to the mode's 'name'
      last_read: 60                # optional; overrides the dataset's read limits
      first_read: 2                # optional
      expect_gain: 38.212          # optional; reports whether the result matches

Dataset paths are tried relative to the current directory, then relative to the
selection YAML's folder. Read limits not given in the entry fall back exactly as in
gain.py: the mode's analysis block, then the dataset's top-level analysis block, then
reads 2 to the end of the ramp.

Usage:
  python gain_select.py selection.yaml
"""
import os
import sys

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gain
import gui_io
import run_dirs


def resolve_path(path, base_dir):
    """Return path as given if it exists, else relative to base_dir."""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    alt = os.path.join(base_dir, path)
    if os.path.exists(alt):
        return alt
    raise FileNotFoundError(f"dataset YAML not found: {path} (also tried {alt})")


def read_limits(entry, mode_cfg, analysis_cfg):
    """first_read/last_read resolved as in gain.main, with entry overrides on top."""
    mode_analysis = mode_cfg.get("analysis", {}) or {}
    out = []
    for key in ("first_read", "last_read"):
        if key in entry:
            out.append(gain.as_optional_int(entry[key]))
        else:
            out.append(gain.as_optional_int(
                mode_analysis.get(key, analysis_cfg.get(key, None))))
    return out


def run_entry(entry, base_dir, log):
    """Fit one selected mode. Returns (label, result dict) or raises."""
    ypath = resolve_path(entry["dataset"], base_dir)
    with open(ypath) as f:
        config = yaml.safe_load(f)

    mode = entry["mode"]
    if mode not in config:
        raise KeyError(f"mode key {mode!r} not in {ypath}; "
                       f"keys there: {[k for k in config if isinstance(config[k], dict)]}")

    # Same setup as gain.main.
    gui = gui_io.is_gui_mode(config)
    ds = config["dataset"]
    head = ds.get("head", "GUI")
    date = ds.get("date", "")
    datapath = ds.get("datapath", "")
    quicklook = False if gui else bool(ds.get("quicklook", False))
    analysis_cfg = config.get("analysis", {}) or {}
    X0, X1 = config["roi"]["x_start"], config["roi"]["x_end"]
    Y0, Y1 = config["roi"]["y_start"], config["roi"]["y_end"]

    mode_cfg = config[mode]
    name = mode_cfg["name"]
    label = str(entry.get("label", name))
    detrend = mode_cfg.get("detrend", False)
    fixed_xmax = gain.as_optional_float(mode_cfg.get("fixed_xmax", None))
    first_read, last_read = read_limits(entry, mode_cfg, analysis_cfg)

    ramps = gain.parse_ramps(mode_cfg, section="illuminated", gui=gui)
    pairs = gain.make_pairs(mode_cfg, ramps, section="illuminated", gui=gui)
    if len(pairs) == 0:
        raise ValueError(f"{label}: need at least 2 illuminated ramps")

    read_text = (f"reads={first_read if first_read is not None else 2}:"
                 f"{last_read if last_read is not None else 'end'} (inclusive)")
    gain.P(log, f"   {label}: dataset={ds.get('name')} ({ypath}), mode key={mode}, "
                f"date={date}, datapath={datapath}")
    gain.P(log, f"   {label}: ROI x[{X0}:{X1}] y[{Y0}:{Y1}]; ramps={ramps}; "
                f"pairs={pairs}; {read_text}")

    r = gain.gain_diff_many_pairs(
        head, date, pairs, datapath, X0, X1, Y0, Y1, log,
        detrend=detrend, fixed_xmax=fixed_xmax, quicklook=quicklook, gui=gui,
        first_read=first_read, last_read=last_read,
    )
    r["itime"] = mode_cfg["frametime"]
    r["meta"] = dict(dataset=ds.get("name"), yaml=ypath, mode_key=mode, date=date,
                     roi=(X0, X1, Y0, Y1), ramps=ramps, read_text=read_text,
                     detrend=detrend, expect=entry.get("expect_gain"))
    return label, r


def plot_one(outpath, setname, label, r):
    """One figure for one mode: PTC by pair with the combined fit, and per-pair gains."""
    m = r["meta"]
    prs = r["pair_results"]
    fig, (ax, axg) = plt.subplots(1, 2, figsize=(13, 5.2),
                                  gridspec_kw=dict(width_ratios=[1.6, 1]))

    cmap = plt.get_cmap("viridis")
    for i, pr in enumerate(prs):
        c = cmap(i / max(len(prs) - 1, 1))
        ax.plot(pr["S"], pr["V"], ".", color=c, ms=5,
                label=f"{pr['pair'][0]}/{pr['pair'][1]}")
    # Points outside the combined fit window, ringed in grey.
    out = ~r["good"]
    ax.plot(r["S"][out], r["V"][out], "o", mfc="none", mec="0.5", ms=8,
            label="not in fit")
    xx = np.linspace(0, r["smax"], 100)
    ax.plot(xx, r["slope"] * xx + r["inter"], "k-", lw=1.5, label="combined fit")
    ax.set_xlabel("signal S [DN]")
    ax.set_ylabel(r"variance/image [DN$^2$]")
    ax.set_title(f"{label}: g = {r['gain']:.3f} ± {r['gain_err']:.3f} e-/DN, "
                 f"RN = {r['rn_e']:.1f} e-\n"
                 f"{m['dataset']} ({m['date']}), ramps {m['ramps'][0]}–{m['ramps'][-1]}, "
                 f"{m['read_text']}" + (", row-detrended" if m["detrend"] else ""),
                 fontsize=10)
    ax.legend(fontsize=7, ncol=2, title="pair", title_fontsize=7)

    g = [pr["gain"] for pr in prs]
    ge = [pr["gain_err"] for pr in prs]
    x = np.arange(len(prs))
    axg.axhspan(r["gain"] - r["gain_err"], r["gain"] + r["gain_err"], color="0.85")
    axg.axhline(r["gain"], color="k", lw=1, label="combined")
    axg.errorbar(x, g, yerr=ge, fmt="o", capsize=3)
    axg.set_xticks(x)
    axg.set_xticklabels([f"{a}/{b}" for a, b in (pr["pair"] for pr in prs)],
                        rotation=60, fontsize=7)
    axg.set_ylabel("gain [e-/DN]")
    spread = np.std(g, ddof=1) if len(g) > 1 else float("nan")
    axg.set_title(f"per-pair gain (std {spread:.2f} e-/DN)", fontsize=10)
    axg.legend(fontsize=8)

    fig.tight_layout()
    safe = label.replace(" ", "_").replace("/", "-").replace(",", "")
    fn = os.path.join(outpath, f"{setname}_{safe}_ptc.png")
    fig.savefig(fn, dpi=110)
    plt.close(fig)
    return fn


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    sel_path = sys.argv[1]
    with open(sel_path) as f:
        sel = yaml.safe_load(f)
    base_dir = os.path.dirname(os.path.abspath(sel_path))
    setname = sel["name"]
    entries = sel["entries"]
    # Per-selection run folder, as for the other scripts (run_dirs.py).
    outpath = run_dirs.output_dir(
        {"dataset": {"outpath": sel["outpath"], "name": setname,
                     "run_subdir": sel.get("run_subdir", True)}},
        sel_path, "gain_select")

    labels = [str(e.get("label", e["mode"])) for e in entries]
    if len(set(labels)) != len(labels):
        sys.exit(f"Labels must be unique; got {labels}. Add 'label:' to the entries.")

    results = {}
    with open(os.path.join(outpath, f"{setname}_summary.txt"), "w") as log:
        gain.P(log, f"=== selected gain PTCs: {setname} (selection {sel_path}) ===\n")
        for entry in entries:
            try:
                label, r = run_entry(entry, base_dir, log)
            except Exception as exc:
                gain.P(log, f"{entry.get('label', entry.get('mode'))} -- failed: {exc}; skipped --\n")
                continue
            results[label] = r

            npairs = len(r["pair_results"])
            nreads = min(pr["nreads_used"] for pr in r["pair_results"])
            gain.P(log, f"{label:12s} ITIME={r['itime']:.4f} pairs={npairs} reads={nreads} "
                        f"maxDN={r['smax']:.0f} gain={r['gain']:.3f} ± {r['gain_err']:.3f} e-/DN "
                        f"RN={r['rn_e']:.1f} e-")
            exp = r["meta"]["expect"]
            if exp is not None:
                # Reported values are rounded, so compare at the precision given.
                s = str(exp)
                nd = len(s.split(".")[1]) if "." in s else 0
                ok = round(r["gain"], nd) == round(float(exp), nd)
                gain.P(log, f"{'':12s} expected {exp}: "
                            + ("MATCH" if ok else f"DIFFERENT (got {r['gain']:.{nd}f})"))

            png = plot_one(outpath, setname, label, r)
            gain.P(log, f"{'':12s} figure: {png}\n")

            safe = label.replace(" ", "_").replace("/", "-").replace(",", "")
            np.savetxt(os.path.join(outpath, f"{setname}_ptc_{safe}.csv"),
                       np.column_stack([r["S"], r["V"]]),
                       header="signal_DN,var_single_DN2", delimiter=",", comments="")
            gain.save_pair_summary(outpath, setname, safe, r["pair_results"])

    if not results:
        sys.exit("No selected mode could be fitted; see the summary file.")
    if len(results) <= 3:
        # gain.py's comparison figure has room for three modes in its top row.
        gain.plot_results(outpath, setname, results)
    else:
        print(f"{len(results)} modes selected; skipping the combined figure (max 3).")


if __name__ == "__main__":
    main()

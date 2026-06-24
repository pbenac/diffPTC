import os
import sys
from itertools import combinations

import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["image.origin"] = "lower"
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.stats import sigma_clipped_stats


def prefix_for_head(head):
    return "ss" if head == "IFS" else "si"


def fits_path(datapath, head, date, filenum):
    obsnum_string = f"{int(filenum):05d}"
    filename = prefix_for_head(head) + date + "_" + obsnum_string + ".fits"
    return os.path.join(datapath, filename)


def parse_ramps(mode_cfg, section="illuminated"):
    """Return ramp numbers from a mode config.

    Preferred:
        illuminated:
          ramps: [4433, 4434, 4435]

    Also supported:
        illuminated:
          ramp_range: [4433, 4436]  # stop-exclusive, like Python range()

    Backward compatible:
        illuminated:
          n1: 4433
          n2: 4434
    """
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


def make_pairs(mode_cfg, ramps, section="illuminated"):
    """Return ramp-number pairs according to the mode config."""
    ramp_cfg = mode_cfg.get(section, {})

    if "pairs" in ramp_cfg:
        return [(int(a), int(b)) for a, b in ramp_cfg["pairs"]]

    if len(ramps) < 2:
        return []

    pairing = ramp_cfg.get("pairing", mode_cfg.get("pairing", None))
    if pairing is None:
        pairing = "first_two" if len(ramps) == 2 else "all_pairs"

    pairing = str(pairing).lower()

    if pairing in ("first_two", "first", "first_pair"):
        return [(ramps[0], ramps[1])]

    if pairing in ("adjacent", "consecutive"):
        return list(zip(ramps[:-1], ramps[1:]))

    if pairing in ("all_pairs", "all", "combinations"):
        return list(combinations(ramps, 2))

    raise ValueError(
        f"Unknown pairing mode {pairing!r}. Use all_pairs, adjacent, first_two, "
        "or explicit illuminated.pairs."
    )


def make_diff(filenum1, filenum2, outfile_name, datapath, head, date, title=None):
    file1 = fits_path(datapath, head, date, filenum1)
    file2 = fits_path(datapath, head, date, filenum2)

    f1 = fits.getdata(file1).astype(float)
    f2 = fits.getdata(file2).astype(float)
    diff = f2 - f1

    plt.figure()
    dat = diff[-1] - diff[0]
    m, _, s = sigma_clipped_stats(dat, sigma=3)
    plt.imshow(dat, vmin=m - 4 * s, vmax=m + 4 * s)
    plt.colorbar()
    if title:
        plt.title(title)
    plt.savefig(outfile_name + ".png")
    plt.close()
    fits.writeto(outfile_name + ".fits", diff, overwrite=True)


def make_cropped_diff(filenum1, filenum2, X0, X1, Y0, Y1, outfile_name, datapath, head, date, title=None):
    file1 = fits_path(datapath, head, date, filenum1)
    file2 = fits_path(datapath, head, date, filenum2)

    inds = np.s_[:, Y0:Y1, X0:X1]

    f1 = fits.getdata(file1).astype(float)[inds]
    f2 = fits.getdata(file2).astype(float)[inds]
    diff = f2 - f1

    plt.figure()
    dat = diff[-1] - diff[0]
    m, _, s = sigma_clipped_stats(dat, sigma=3)
    plt.imshow(dat, vmin=m - 4 * s, vmax=m + 4 * s)
    plt.colorbar()
    if title:
        plt.title(title)
    plt.savefig(outfile_name + "crop.png")
    plt.close()
    fits.writeto(outfile_name + "crop.fits", diff, overwrite=True)


def main():
    if len(sys.argv) < 2:
        print("Error: No yaml configuration file provided")
        sys.exit(1)

    yamlfile = sys.argv[1]
    with open(yamlfile) as f:
        config = yaml.safe_load(f)

    head = config["dataset"]["head"]
    date = config["dataset"]["date"]
    datapath = config["dataset"]["datapath"]
    outpath = config["dataset"]["outpath"]
    setname = config["dataset"]["name"]
    modes = config["dataset"].get("modes", ["slow52", "fast10", "fast06"])

    X0 = config["roi"]["x_start"]
    X1 = config["roi"]["x_end"]
    Y0 = config["roi"]["y_start"]
    Y1 = config["roi"]["y_end"]

    diff_dir = os.path.join(outpath, "diffs")
    os.makedirs(diff_dir, exist_ok=True)

    for mode in modes:
        mode_cfg = config[mode]
        name = mode_cfg["name"]
        ramps = parse_ramps(mode_cfg, section="illuminated")
        pairs = make_pairs(mode_cfg, ramps, section="illuminated")

        if not pairs:
            print(f"{name}: no illuminated ramp pairs configured; skipped")
            continue

        print(f"{name}: illuminated ramps={ramps}; pairs={pairs}")
        for n1, n2 in pairs:
            base = os.path.join(diff_dir, f"{setname}_{name}_{int(n1):05d}_{int(n2):05d}_difference")
            title = f"{setname} {name}: {int(n2):05d} - {int(n1):05d}"
            make_diff(n1, n2, base, datapath, head, date, title=title)
            make_cropped_diff(n1, n2, X0, X1, Y0, Y1, base, datapath, head, date, title=title + " crop")


if __name__ == "__main__":
    main()

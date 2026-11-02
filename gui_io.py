#!/usr/bin/env python3
"""Shared I/O helpers for MACIE MSAC GUI Live Mode acquisitions.

GUI Live Mode is enabled by setting::

    basic:
      gui: true

A GUI acquisition is identified by its timestamp directory, for example
``20260831115412``.  The fixed acquisition root is intentionally defined here
once so it never needs to appear in YAML files or command-line invocations.

Each acquisition directory contains one FITS file per read, named like::

    Frame_R0001_M0001_N0001.fits
    Frame_R0001_M0001_N0002.fits
    ...

The files are sorted by the integer after ``_N`` and assembled into the same
array convention used by the rest of this codebase: ``(N_reads, ny, nx)``.
ROI tuples are ``(X0, X1, Y0, Y1)`` and are applied as
``image[Y0:Y1, X0:X1]`` before data are copied into the output cube.
"""

import glob
import os
import re

import numpy as np
from astropy.io import fits


GUI_BASE_PATH = (
    "/usr/local/MACIE/MACIE_v6.0/"
    "macie_v6.0_ubuntu/MacieApp/data/macie/"
    "H2RG_ASIC/UpTheRamp"
)

GUI_FRAME_GLOB = "Frame_R0001_M0001_N*.fits"


def truthy(value):
    """Robustly interpret YAML/string boolean values."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def is_gui_mode(config):
    """Return True when the config requests GUI Live Mode.

    The canonical switch is ``basic.gui``. ``dataset.gui`` is accepted as a
    compatibility fallback so older experimental configs are not brittle.
    """
    basic = config.get("basic", {}) or {}
    if "gui" in basic:
        return truthy(basic.get("gui"))
    return truthy((config.get("dataset", {}) or {}).get("gui", False))


def gui_ramp_directory(timestamp):
    """Return the fixed GUI acquisition directory for one timestamp."""
    return os.path.join(GUI_BASE_PATH, str(timestamp))


def frame_number_from_filename(filename):
    """Extract the integer read number following ``_N``."""
    match = re.search(r"_N(\d+)\.fits$", os.path.basename(filename), re.IGNORECASE)
    if match is None:
        raise ValueError(f"Could not parse GUI frame number from {filename!r}")
    return int(match.group(1))


def discover_gui_frames(timestamp):
    """Find and numerically sort all read FITS files for one GUI acquisition."""
    ramp_dir = gui_ramp_directory(timestamp)
    if not os.path.isdir(ramp_dir):
        raise FileNotFoundError(f"GUI acquisition directory does not exist: {ramp_dir}")

    filenames = glob.glob(os.path.join(ramp_dir, GUI_FRAME_GLOB))
    if not filenames:
        raise FileNotFoundError(
            f"No files matching {GUI_FRAME_GLOB!r} found in {ramp_dir}"
        )

    return sorted(filenames, key=frame_number_from_filename)


def gui_acquisition_ready(timestamp, min_reads=1):
    """Return True when a GUI timestamp directory contains at least min_reads."""
    try:
        return len(discover_gui_frames(timestamp)) >= int(min_reads)
    except Exception:
        return False


def _read_2d_image(filename):
    """Load one GUI read as float64 and apply FITS BZERO manually."""
    with fits.open(filename, memmap=True, do_not_scale_image_data=True) as h:
        bz = h[0].header.get("BZERO", 0)
        data = np.squeeze(np.asarray(h[0].data))
        if data.ndim != 2:
            raise ValueError(
                f"Expected one 2-D image per GUI FITS file; {filename} has shape "
                f"{np.asarray(h[0].data).shape}"
            )
        image = data.astype(np.float64) + bz
        header = dict(h[0].header)
    return image, header


def load_gui_frame(timestamp, read_index, roi=None):
    """Load one zero-indexed read from a GUI acquisition.

    Returns ``(image, header, filename)``.
    """
    filenames = discover_gui_frames(timestamp)
    j = int(read_index)
    if j < 0 or j >= len(filenames):
        raise IndexError(
            f"read_index={j} outside GUI acquisition {timestamp} with "
            f"{len(filenames)} reads"
        )

    image, header = _read_2d_image(filenames[j])
    if roi is not None:
        X0, X1, Y0, Y1 = map(int, roi)
        image = image[Y0:Y1, X0:X1]

    return image, header, filenames[j]


def load_gui_cube(timestamp, roi=None):
    """Assemble individual GUI FITS reads into ``(N_reads, ny, nx)``.

    The ROI is applied read-by-read before stacking, so only the requested
    spatial window is retained in memory when an ROI is supplied.

    Returns ``(cube, first_header, source)`` where ``source`` is the timestamp
    directory path.
    """
    filenames = discover_gui_frames(timestamp)
    frames = []
    first_header = None

    for filename in filenames:
        image, header = _read_2d_image(filename)
        if first_header is None:
            first_header = header

        if roi is not None:
            X0, X1, Y0, Y1 = map(int, roi)
            image = image[Y0:Y1, X0:X1]

        frames.append(image)

    shapes = {frame.shape for frame in frames}
    if len(shapes) != 1:
        raise ValueError(
            f"GUI frames for {timestamp} do not all have the same shape: {sorted(shapes)}"
        )

    return np.stack(frames, axis=0), first_header, gui_ramp_directory(timestamp)


def gui_identifiers(mode_cfg, section="illuminated"):
    """Return GUI timestamp identifiers from one mode/section.

    Preferred YAML::

        illuminated:
          gui_timestamps:
            - '20260831115412'
            - '20260831120135'

    ``timestamps`` and a single ``timestamp`` are accepted aliases.
    """
    section_cfg = mode_cfg.get(section, {}) or {}

    for key in ("gui_timestamps", "timestamps"):
        if key in section_cfg:
            return [str(x) for x in section_cfg[key]]

    if "timestamp" in section_cfg:
        return [str(section_cfg["timestamp"])]

    return []


def parse_acquisitions(mode_cfg, section="illuminated", gui=False):
    """Return timestamp strings in GUI mode, otherwise normal ramp numbers."""
    if gui:
        return gui_identifiers(mode_cfg, section=section)

    ramp_cfg = mode_cfg.get(section, {}) or {}
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


def identifier_label(identifier):
    """Filesystem/log-safe human-readable acquisition identifier."""
    return str(identifier)

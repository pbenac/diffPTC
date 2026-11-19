# LLM-generated (Claude). Shared FITS file lookup that lets 'datapath' be a list of folders.
"""
Find a ramp's FITS file when a dataset's ramps are spread over several folders.

A night that runs past midnight puts later ramps in the next night's folder, and their
names carry the next date (si261119_00288.fits), while a ramp that started before
midnight can be written after it and keep the old date (nov19/si261118_00280.fits).
So in a dataset YAML, 'datapath' may now be a list:

  dataset:
    datapath:
      - /data20/CD7/IMG/nov18/
      - /data20/CD7/IMG/nov19/
    date: '261118'      # still required, but with a list it is not used to find files

With a list, each ramp is looked up by its prefix and number in every listed folder
(in its ql_redux/ subfolder for quicklook data), whatever the date in its name. Exactly
one match is required: two files with the same number (for example after an OBSNUM
reset) stop the run with an error naming both. If nothing matches, the path the single-
folder code would have used (first folder, configured date) is returned, so "file not
ready" and "file not found" behave as before.

With a single string, paths are built exactly as before.
"""
import os
import re

_NAME = re.compile(r"^(ss|si)(\d*)(_\d+.*\.fits)$")


def data_directory(datapath, quicklook=False):
    """Folder for normal or quicklook files (one datapath string)."""
    return os.path.join(datapath, "ql_redux") if quicklook else datapath


def is_list(datapath):
    return isinstance(datapath, (list, tuple))


def find(datapath, quicklook, filename):
    """Full path for 'filename' (as the calling script built it from the configured date).

    datapath: a folder (str) or a list of folders.
    """
    if not is_list(datapath):
        return os.path.join(data_directory(datapath, quicklook), filename)

    folders = [data_directory(str(p), quicklook) for p in datapath]
    if not folders:
        raise ValueError("datapath is an empty list")
    m = _NAME.match(filename)
    if m is None:
        raise ValueError(f"cannot match {filename!r} by number; expected names like si261118_00288.fits")
    pattern = re.compile(re.escape(m.group(1)) + r"\d+" + re.escape(m.group(3)) + "$")

    hits = []
    for folder in folders:
        try:
            names = os.listdir(folder)
        except FileNotFoundError:
            continue
        hits.extend(os.path.join(folder, n) for n in sorted(names) if pattern.match(n))
    if len(hits) > 1:
        raise ValueError(f"more than one file matches {filename!r} in datapath: {hits}")
    if hits:
        return hits[0]
    return os.path.join(folders[0], filename)


def describe(datapath, quicklook=False):
    """Folder text for log lines."""
    if is_list(datapath):
        return ", ".join(data_directory(str(p), quicklook) for p in datapath)
    return data_directory(datapath, quicklook)

# LLM-generated (Claude). Per-dataset output folders so a rerun never overwrites results.
"""
Where an analysis script writes its outputs.

Each script calls output_dir() at the start of main(). Outputs go to

  <outpath>/<dataset name>/        e.g. results/20261118_IMG_darks_120s/

If that folder already holds a run of the SAME script, the next free folder is used:
<dataset name>_1, then _2, and so on. Different scripts on one dataset share a folder
(gain.py then dark_noise.py both land in 20261118_IMG_darks_120s/); running gain.py on
it again goes to 20261118_IMG_darks_120s_1/, and a second dark_noise.py run joins it there.

Each run leaves two files beside its outputs, which also mark the folder as used by
that script:
  config_<script>.yaml     exact copy of the YAML that was run
  run_info_<script>.txt    command, time, host, working folder, git commit/branch/changes

To write straight into outpath as before (used by llm_tools/run_regression.py so its
comparison with main lines up), set under dataset:   run_subdir: false
"""
import datetime
import os
import shutil
import socket
import subprocess
import sys


def _enabled(value):
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "no", "off", "0")
    return value is not False and value != 0


def _git(args, cwd):
    try:
        r = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def output_dir(config, yaml_path, script):
    """Create and return the folder this run writes to (see module docstring)."""
    ds = config["dataset"]
    base = ds["outpath"]
    if not _enabled(ds.get("run_subdir", True)):
        os.makedirs(base, exist_ok=True)
        return base

    name = str(ds["name"])
    if not name or os.sep in name or name in (".", ".."):
        raise ValueError(f"dataset name {name!r} cannot be used as a folder name")

    i = 0
    while True:
        folder = os.path.join(base, name if i == 0 else f"{name}_{i}")
        if not os.path.exists(os.path.join(folder, f"run_info_{script}.txt")):
            break
        i += 1
    os.makedirs(folder, exist_ok=True)

    shutil.copyfile(yaml_path, os.path.join(folder, f"config_{script}.yaml"))
    code = os.path.dirname(os.path.abspath(sys.argv[0])) or "."
    changes = _git(["status", "--porcelain", "--untracked-files=no"], code)
    with open(os.path.join(folder, f"run_info_{script}.txt"), "w") as f:
        f.write(f"script: {script}\n")
        f.write(f"command: {' '.join([sys.executable] + sys.argv)}\n")
        f.write(f"yaml: {os.path.abspath(yaml_path)}\n")
        f.write(f"started: {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"host: {socket.gethostname()}\n")
        f.write(f"working folder: {os.getcwd()}\n")
        f.write(f"code folder: {code}\n")
        f.write(f"git commit: {_git(['rev-parse', 'HEAD'], code)}\n")
        f.write(f"git branch: {_git(['rev-parse', '--abbrev-ref', 'HEAD'], code)}\n")
        f.write("uncommitted changes to tracked files: "
                + ("none" if changes == "" else "unknown" if changes == "unknown"
                   else "\n" + changes) + "\n")
    print(f"Writing outputs to {folder}")
    return folder

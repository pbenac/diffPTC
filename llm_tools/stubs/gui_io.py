"""LLM-generated STUB of the repo's `gui_io` module (not tracked in the repo).

Only used for local synthetic tests when the real gui_io is unavailable.
Implements just enough for non-GUI mode.
"""
GUI_BASE_PATH = ""


def is_gui_mode(config):
    return False


def gui_acquisition_ready(*a, **k):
    raise RuntimeError("gui_io stub: GUI mode not supported")


def load_gui_cube(*a, **k):
    raise RuntimeError("gui_io stub: GUI mode not supported")


def gui_identifiers(*a, **k):
    raise RuntimeError("gui_io stub: GUI mode not supported")

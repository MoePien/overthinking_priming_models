"""Where data and outputs live. Set OGEN_DATA and OGEN_OUT, or use ./data and ./outputs in the repo root."""
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = pathlib.Path(os.environ.get("OGEN_DATA", ROOT / "data"))
OUT = pathlib.Path(os.environ.get("OGEN_OUT", ROOT / "outputs"))

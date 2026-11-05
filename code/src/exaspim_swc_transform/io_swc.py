"""Read SWC files with their ``# OFFSET`` header applied."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from allensdk.core import swc
from allensdk.core.swc import Morphology


def read_swc_offset(path: Path) -> np.ndarray | None:
    """Return the ``# OFFSET`` header as an array, or ``None`` if there is none."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("# OFFSET"):
                return np.array([float(x) for x in line.split()[2:5]], dtype=float)
    return None


def read_swc(path: Path) -> Morphology:
    """Read an SWC and add its offset to every node."""
    morph = swc.read_swc(str(path))
    offset = read_swc_offset(path)
    if offset is not None:
        for node in morph.compartment_list:
            node["x"] += offset[0]
            node["y"] += offset[1]
            node["z"] += offset[2]
    return morph

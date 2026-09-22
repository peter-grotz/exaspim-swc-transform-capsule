"""SWC I/O helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from allensdk.core import swc
from allensdk.core.swc import Morphology


def read_swc_offset(path: Path) -> np.ndarray | None:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("# OFFSET"):
                return np.array([float(x) for x in line.split()[2:5]], dtype=float)
    return None


def read_swc(path: Path, add_offset: bool = True) -> Morphology:
    morph = swc.read_swc(str(path))
    if add_offset:
        offset = read_swc_offset(path)
        if offset is not None:
            for node in morph.compartment_list:
                node["x"] += offset[0]
                node["y"] += offset[1]
                node["z"] += offset[2]
    return morph

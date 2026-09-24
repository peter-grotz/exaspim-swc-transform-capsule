"""Choose the template-to-CCF transform for a sample.

Samples reconstructed before the pipeline standardised on v1.5 had their CCF coordinates
produced with other template-to-CCF versions. Re-transforming them with v1.5 would move
their neurons relative to what was published, so each keeps the version it was originally
transformed with. Everything else uses v1.5.

The exceptions live in ``code/template_to_ccf_by_sample.csv``, taken from the
``exaspim_swc_transform`` step recorded in each sample's processed-reconstruction asset.
Samples whose version was never recorded -- all produced by the legacy transform -- are
marked ``legacy`` and refused here: they need the legacy pipeline.

The registration itself does not record this reliably. It hardcodes v1.4 for its own CCF
products, so its metadata says v1.4 for every sample regardless of what was applied to the
reconstructions.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TEMPLATE_TO_CCF = "reg_exaspim_template_to_ccf_25um_v1.5"
"""Template-to-CCF asset used for every sample not listed as an exception."""

LEGACY = "legacy"
"""Table value for a sample whose original transform version is unknown."""

TABLE = Path(__file__).resolve().parents[2] / "template_to_ccf_by_sample.csv"
"""Exceptions table, shipped with the capsule so each run is pinned to its commit."""


class LegacySampleError(Exception):
    """Raised for a sample that was transformed with an unrecorded legacy version."""


@dataclass(frozen=True)
class TemplateSelection:
    """The template-to-CCF asset chosen for a sample, and why.

    Attributes
    ----------
    asset : str
        Mount name of the template-to-CCF data asset, e.g.
        ``"reg_exaspim_template_to_ccf_25um_v1.3"``.
    basis : str
        Why this asset was chosen, for the stage record.
    """

    asset: str
    basis: str


def load_table(path: Path = TABLE) -> dict[str, tuple[str, str]]:
    """Read the per-sample exceptions table.

    Parameters
    ----------
    path : Path, optional
        CSV with columns ``sample_id``, ``template_to_ccf``, ``basis``.

    Returns
    -------
    dict[str, tuple[str, str]]
        Sample id to ``(template_to_ccf, basis)``.

    Raises
    ------
    ValueError
        If a sample appears twice, since the table would then be ambiguous.
    """
    table: dict[str, tuple[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sample = row["sample_id"].strip()
            if sample in table:
                raise ValueError(f"Sample {sample} appears more than once in {path}")
            table[sample] = (row["template_to_ccf"].strip(), row["basis"].strip())
    return table


def select_template_to_ccf(
    sample_id: str, table: dict[str, tuple[str, str]] | None = None
) -> TemplateSelection:
    """Choose the template-to-CCF asset for a sample.

    Parameters
    ----------
    sample_id : str
        Six-digit subject id.
    table : dict[str, tuple[str, str]] | None, optional
        Exceptions table. Read from :data:`TABLE` when omitted.

    Returns
    -------
    TemplateSelection
        The asset to use and the reason.

    Raises
    ------
    LegacySampleError
        If the sample was transformed with a legacy version that was never recorded.
    """
    entries = load_table() if table is None else table
    if sample_id not in entries:
        return TemplateSelection(DEFAULT_TEMPLATE_TO_CCF, "default: not a pre-v1.5 sample")
    asset, basis = entries[sample_id]
    if asset == LEGACY:
        raise LegacySampleError(
            f"Sample {sample_id} was transformed with the legacy transform, which recorded no "
            f"template-to-CCF version ({basis}). It needs the legacy pipeline."
        )
    return TemplateSelection(asset, f"per-sample table: {basis}")

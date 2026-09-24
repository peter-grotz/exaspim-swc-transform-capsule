"""Find the manual CCF refinement displacement field for a sample, if one was made.

Manual refinement writes a single ``.nrrd`` displacement field to
``s3://<bucket>/<dataset>/manual_ccf_refinement/``. When it is there it is applied after
the template-to-CCF transform; when it is not, the sample is transformed without one. That
is the normal state for a sample nobody has refined yet, so absence is logged and recorded
rather than treated as an error.

More than one ``.nrrd`` is an error: applying the wrong refinement moves every neuron, and
nothing here can tell which one was meant.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

REFINEMENT_DIR = "manual_ccf_refinement"
"""Folder under a processed dataset that holds the manual CCF refinement."""

FIELD_SUFFIX = ".nrrd"
"""Extension of the displacement field."""


class _Paginator(Protocol):
    """The part of a boto3 paginator used here."""

    def paginate(self, **kwargs: str) -> Iterable[dict]:
        """Yield pages of a listing.

        Parameters
        ----------
        **kwargs : str
            ``Bucket`` and ``Prefix``.
        """


class S3ListingClient(Protocol):
    """The part of a boto3 S3 client used here."""

    def get_paginator(self, operation: str) -> _Paginator:
        """Return a paginator for an operation.

        Parameters
        ----------
        operation : str
            E.g. ``"list_objects_v2"``.
        """

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        """Download an object to a local file.

        Parameters
        ----------
        bucket : str
            Bucket.
        key : str
            Object key.
        filename : str
            Local destination.
        """


class AmbiguousDisplacementFieldError(Exception):
    """Raised when the refinement folder holds more than one displacement field."""


@dataclass(frozen=True)
class DisplacementField:
    """Outcome of looking for a sample's manual refinement.

    Attributes
    ----------
    local_path : str
        Where the field was downloaded to, or ``""`` when none was found.
    source : str
        The S3 URI it came from, or the location that was searched.
    status : str
        ``"applied"`` or ``"not found; skipped"``, for the stage record.
    """

    local_path: str
    source: str
    status: str


def fetch_displacement_field(
    client: S3ListingClient, bucket: str, dataset: str, destination: Path
) -> DisplacementField:
    """Download the sample's displacement field, if its refinement folder holds one.

    Parameters
    ----------
    client : S3ListingClient
        An S3 client.
    bucket : str
        Bucket holding the processed dataset.
    dataset : str
        Processed dataset name.
    destination : Path
        Directory to download into.

    Returns
    -------
    DisplacementField
        The local path and source when found; an empty path and a "skipped" status when not.

    Raises
    ------
    AmbiguousDisplacementFieldError
        If the folder holds more than one ``.nrrd``.
    """
    prefix = f"{dataset}/{REFINEMENT_DIR}/"
    searched = f"s3://{bucket}/{prefix}"
    keys = sorted(
        obj["Key"]
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
        for obj in page.get("Contents", [])
        if obj["Key"].endswith(FIELD_SUFFIX)
    )
    if not keys:
        logger.warning("No displacement field in %s; transforming without one", searched)
        return DisplacementField("", searched, "not found; skipped")
    if len(keys) > 1:
        listing = "\n".join(f"  s3://{bucket}/{key}" for key in keys)
        raise AmbiguousDisplacementFieldError(
            f"{len(keys)} displacement fields in {searched}; expected one:\n{listing}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    local = destination / Path(keys[0]).name
    client.download_file(bucket, keys[0], str(local))
    source = f"s3://{bucket}/{keys[0]}"
    logger.info("Applying displacement field %s", source)
    return DisplacementField(str(local), source, "applied")

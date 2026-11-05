"""Find a sample's manual CCF refinement displacement field on S3."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

REFINEMENT_DIR = "manual_ccf_refinement"
FIELD_SUFFIX = ".nrrd"


class AmbiguousDisplacementFieldError(Exception):
    """Raised when the refinement folder holds more than one displacement field."""


@dataclass(frozen=True)
class DisplacementField:
    """A downloaded displacement field and the S3 URI it came from."""

    local_path: str
    source: str


def fetch_displacement_field(
    client, bucket: str, dataset: str, destination: Path
) -> DisplacementField | None:
    """Download the field in ``<dataset>/manual_ccf_refinement/``, or return ``None``.

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
        logger.warning("No displacement field in %s", searched)
        return None
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
    return DisplacementField(str(local), source)

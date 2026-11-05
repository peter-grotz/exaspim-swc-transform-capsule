"""Download a sample's sample-to-template transforms from S3."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

BUCKET_DEFAULT = "aind-open-data"

TRANSFORMS = (
    "{subject}_to_exaSPIM_SyN_0GenericAffine.mat",
    "{subject}_to_exaSPIM_SyN_1InverseWarp.nii.gz",
)
"""Sample-to-template transforms the registration writes under ``ccf_alignment/``."""


def s3_client():
    """Return an anonymous S3 client for the public data bucket."""
    import boto3
    from botocore import UNSIGNED
    from botocore.client import Config

    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def stage_transforms(
    client, bucket: str, dataset: str, subject_id: str, destination: Path
) -> list[str]:
    """Download the dataset's sample-to-template affine and inverse warp.

    Files already in ``destination`` are reused, so it should be unique to the dataset.
    """
    paths = []
    for template in TRANSFORMS:
        name = template.format(subject=subject_id)
        target = destination / name
        if not (target.is_file() and target.stat().st_size > 0):
            target.parent.mkdir(parents=True, exist_ok=True)
            key = f"{dataset}/ccf_alignment/{name}"
            logger.info("Staging s3://%s/%s", bucket, key)
            client.download_file(bucket, key, str(target))
        paths.append(str(target))
    return paths

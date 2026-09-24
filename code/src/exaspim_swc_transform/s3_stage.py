"""Download a sample's sample-to-template registration from S3.

Only the two sample-to-template transforms are staged. Everything else the transform needs
is obtained elsewhere: the dataset itself is resolved through the metadata registry
(:mod:`exaspim_swc_processing.datasets`), the acquisition through
:mod:`exaspim_swc_processing.acquisition`, and the reference-volume geometry is derived
(:mod:`exaspim_swc_transform.reference`) rather than downloaded.

The transform file names -- ``<subject>_to_exaSPIM_SyN_*`` -- are the registration
capsule's output contract, not a dataset naming convention.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from botocore.client import BaseClient

logger = logging.getLogger(__name__)

BUCKET_DEFAULT = "aind-open-data"
"""Bucket used when a dataset specification does not name one."""

TRANSFORMS = (
    "{subject}_to_exaSPIM_SyN_0GenericAffine.mat",
    "{subject}_to_exaSPIM_SyN_1InverseWarp.nii.gz",
)
"""Sample-to-template transforms the registration writes under ``ccf_alignment/``."""


def s3_client() -> BaseClient:
    """Return an anonymous S3 client for the public data bucket.

    Returns
    -------
    BaseClient
        A boto3 S3 client configured for unsigned requests.
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.client import Config

    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def stage_registration_bundle(
    dataset: str,
    subject_id: str,
    bucket: str = BUCKET_DEFAULT,
    dest_root: str = "/scratch/reg_bundle",
) -> str:
    """Download a dataset's sample-to-template transforms into a local bundle.

    Parameters
    ----------
    dataset : str
        Processed dataset name, as resolved from the registry.
    subject_id : str
        Its subject, which names the transform files.
    bucket : str, optional
        Bucket holding the dataset.
    dest_root : str, optional
        Local directory to stage into.

    Returns
    -------
    str
        The bundle root, laid out as ``<dest_root>/ccf_alignment/<transform files>``.
    """
    client = s3_client()
    root = Path(dest_root)
    for template in TRANSFORMS:
        name = template.format(subject=subject_id)
        target = root / "ccf_alignment" / name
        if target.is_file() and target.stat().st_size > 0:
            logger.info("Already staged: %s", name)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        key = f"{dataset}/ccf_alignment/{name}"
        logger.info("Staging s3://%s/%s", bucket, key)
        client.download_file(bucket, key, str(target))
    return str(root)

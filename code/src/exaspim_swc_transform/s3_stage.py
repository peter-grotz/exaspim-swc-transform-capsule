"""Stage a sample's per-sample CCF registration files from S3 into a local transform_dir bundle.

The exaSPIM processed dataset on s3://aind-open-data/<dataset>/ccf_alignment/
holds the per-sample registration files, while acquisition.json lives at the
processed-dataset root:

    s3://aind-open-data/<dataset>/acquisition.json

The acquisition.json file is staged locally as:

    ccf_alignment/registration_metadata/acquisition_<dataset_id>.json

This preserves the existing local ccf_alignment/[registration_metadata/] layout
so transform_resolution.resolve_inputs works unchanged when pointed at the
staged root.

Anonymous access is used because aind-open-data is public; this matches the
OUTPUT_PREFIX / --no-sign-request pattern.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

BUCKET_DEFAULT = "aind-open-data"


def _client():
    import boto3
    from botocore import UNSIGNED
    from botocore.client import Config

    return boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED),
    )


def s3_client():
    """Return an anonymous S3 client for the public data bucket."""
    return _client()


def resolve_dataset(
    spec: str,
    bucket: str = BUCKET_DEFAULT,
):
    """Resolve (bucket, dataset_dir) from an S3 URI, exaSPIM_* dataset name, or sample ID."""
    s = spec.strip().strip("'\"")

    # Full S3 URI:
    # s3://aind-open-data/exaSPIM_784896_.../...
    if s.startswith("s3://"):
        u = urlparse(s)
        return u.netloc, u.path.lstrip("/").split("/")[0]

    # Dataset directory name:
    # exaSPIM_784896_2025-08-19_..._processed_...
    if s.startswith("exaSPIM_"):
        return bucket, s

    # Bare specimen/sample ID:
    # 784896
    if re.fullmatch(r"\d{5,}", s):
        cli = _client()
        datasets = set()

        for page in cli.get_paginator("list_objects_v2").paginate(
            Bucket=bucket,
            Prefix=f"exaSPIM_{s}_",
            Delimiter="/",
        ):
            for cp in page.get("CommonPrefixes", []):
                name = cp["Prefix"].rstrip("/")

                if "_processed_" in name:
                    datasets.add(name)

        # Prefer the newest processed dataset that actually has a
        # ccf_alignment directory.
        for ds in sorted(datasets, reverse=True):
            result = cli.list_objects_v2(
                Bucket=bucket,
                Prefix=f"{ds}/ccf_alignment/",
                MaxKeys=1,
            )

            if result.get("KeyCount"):
                return bucket, ds

        # Fall back to newest processed dataset if none expose
        # ccf_alignment.
        if datasets:
            return bucket, sorted(datasets)[-1]

        raise FileNotFoundError(
            f"No processed dataset for sample {s} "
            f"in s3://{bucket}"
        )

    raise ValueError(
        f"Unrecognized processed-dataset spec: {spec!r}"
    )


def stage_registration_bundle(
    spec: str,
    dest_root: str = "/scratch/reg_bundle",
    bucket: str = BUCKET_DEFAULT,
) -> str:
    """Stage the five required registration files into a local bundle.

    S3 source layout:

        <dataset>/
        ├── acquisition.json
        └── ccf_alignment/
            ├── <id>_to_exaSPIM_SyN_0GenericAffine.mat
            ├── <id>_to_exaSPIM_SyN_1InverseWarp.nii.gz
            └── registration_metadata/
                ├── <id>_10um_loaded_zarr_img.nii.gz
                └── <id>_10um_resampled_zarr_img.nii.gz

    Local staged layout:

        <dest_root>/
        └── ccf_alignment/
            ├── <id>_to_exaSPIM_SyN_0GenericAffine.mat
            ├── <id>_to_exaSPIM_SyN_1InverseWarp.nii.gz
            └── registration_metadata/
                ├── acquisition_<id>.json
                ├── <id>_10um_loaded_zarr_img.nii.gz
                └── <id>_10um_resampled_zarr_img.nii.gz

    Returns:
        The local staged root directory.
    """
    bucket, ds = resolve_dataset(spec, bucket)

    match = re.search(r"\d{5,}", ds)
    if match is None:
        raise ValueError(
            f"Could not extract dataset/sample ID from dataset name: {ds}"
        )

    dataset_id = match.group(0)

    align = f"{ds}/ccf_alignment"
    # Explicit source -> destination mapping is intentional.
    #
    # acquisition.json lives at the processed-dataset root in S3, but
    # downstream transform-resolution code expects it inside the staged
    # registration_metadata directory with the sample ID in its filename.
    files = [
        (
            f"{align}/{dataset_id}_to_exaSPIM_SyN_0GenericAffine.mat",
            f"ccf_alignment/{dataset_id}_to_exaSPIM_SyN_0GenericAffine.mat",
        ),
        (
            f"{align}/{dataset_id}_to_exaSPIM_SyN_1InverseWarp.nii.gz",
            f"ccf_alignment/{dataset_id}_to_exaSPIM_SyN_1InverseWarp.nii.gz",
        ),
        (
            f"{ds}/acquisition.json",
            f"ccf_alignment/registration_metadata/"
            f"acquisition_{dataset_id}.json",
        ),
    ]

    # The two reference volumes under registration_metadata/ used to be staged here,
    # 7.8 GB for 794492. Nothing reads their voxels and 20 of 60 processed assets never
    # published them, so their geometry is derived instead -- see
    # exaspim_swc_transform.reference.

    cli = _client()
    dest = Path(dest_root)

    print(f"[s3-stage] dataset={ds} -> {dest}")

    for key, rel in files:
        out = dest / rel
        out.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if out.exists() and out.stat().st_size > 0:
            print(f"[s3-stage] cached {rel}")
            continue

        print(
            f"[s3-stage] downloading "
            f"s3://{bucket}/{key} -> {rel} ..."
        )

        cli.download_file(
            bucket,
            key,
            str(out),
        )

    return str(dest)
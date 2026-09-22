"""Assemble the images ``RegistrationPipeline`` needs, without reading reference volumes.

``RegistrationPipeline.load_images`` reads four images. Two are mounted data assets -- the
CCF template and the exaSPIM template -- and are read here unchanged. The other two are
the registration's own reference volumes, 1.4-1.8 GB each, published under
``registration_metadata/`` by only 40 of the 60 processed exaSPIM assets.

Those two are replaced by stand-ins carrying the right geometry and no voxels. See
:mod:`exaspim_swc_processing.reference` for why that is sufficient and
:mod:`exaspim_swc_processing.registration` for where the geometry comes from.

One upstream call has to be neutralised for this to work:
:func:`disable_overlay_normalization`. Its docstring explains why.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np
from exaspim_swc_processing.reference import apply_geometry, reference_arrays
from exaspim_swc_processing.registration import (
    HEADER_FETCH_BYTES,
    find_template_to_ccf_asset,
    PROCESSING_RECORD,
    VolumeGeometry,
    loaded_geometry,
    parse_nifti_geometry,
    parse_registration_record,
    RegistrationRecordError,
    reconcile,
    registration_volume_key,
    resampled_geometry,
    zarr_level_key,
    zarr_shape,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegistrationInputs:
    """What the registration recorded about the space it worked in.

    Attributes
    ----------
    loaded : VolumeGeometry
        Geometry of the volume the registration read from the zarr.
    resampled : VolumeGeometry
        Geometry of the isotropically resampled volume.
    template_to_ccf_asset : str | None
        Name of the template-to-CCF data asset the registration used, or ``None`` when
        the record names none and the caller must fall back to its default.
    """

    loaded: VolumeGeometry
    resampled: VolumeGeometry
    template_to_ccf_asset: str | None


@dataclass(frozen=True)
class ReferenceImages:
    """What ``preprocess_coords`` and ``apply_transforms_to_points`` consume.

    Attributes
    ----------
    ccf : object
        The CCF template, read from its mounted asset.
    exaspim_template : object
        The exaSPIM template, read from its mounted asset.
    brain : np.ndarray
        Stand-in for the loaded volume. Only its shape is read.
    resampled : np.ndarray
        Stand-in for the resampled volume. Nothing reads it.
    resampled_image : object
        A minimal image carrying the resampled volume's geometry.
    """

    ccf: object
    exaspim_template: object
    brain: np.ndarray
    resampled: np.ndarray
    resampled_image: object


def disable_overlay_normalization() -> None:
    """Make ``ImageVisualizer.perc_normalization`` a pass-through.

    It rescales a volume so its 2nd and 98th percentiles map to 0 and 1, purely so the QC
    overlays render legibly. ``preprocess_coords`` calls it on both reference volumes
    *before* the ``if cell_filename:`` guard that would skip the rendering, and
    ``load_images`` calls it on the CCF template. In this pipeline no overlay is drawn, so
    every result is discarded.

    Two reasons to disable it rather than feed it data. It asserts that its two
    percentiles differ, which constant stand-ins cannot satisfy. And on real volumes it is
    the most expensive operation in the stage -- ``np.percentile`` partitions a copy of
    1.6 billion voxels, and the rescaling allocates another array -- for a result nothing
    reads.

    No transformed coordinate depends on it, so this cannot change scientific output.
    """
    from aind_exaspim_register_cells.visualization import ImageVisualizer

    ImageVisualizer.perc_normalization = staticmethod(lambda image, *_args, **_kwargs: image)
    logger.info("Disabled overlay normalization; reference volumes are geometry-only")


def _published_geometry(client, bucket: str, key: str) -> VolumeGeometry | None:
    """Read a published volume's geometry from its header, if the volume exists.

    Parameters
    ----------
    client : object
        An S3 client.
    bucket : str
        Bucket holding the dataset.
    key : str
        Key of the ``.nii.gz``.

    Returns
    -------
    VolumeGeometry | None
        The geometry, or ``None`` when the volume was not published.
    """
    try:
        body = client.get_object(Bucket=bucket, Key=key, Range=f"bytes=0-{HEADER_FETCH_BYTES - 1}")
    except Exception as error:  # noqa: BLE001 - absence is the common, expected case
        logger.info("No published volume at %s (%s)", key, type(error).__name__)
        return None
    return parse_nifti_geometry(body["Body"].read())


def resolve_registration(
    client, bucket: str, dataset: str, dataset_id: str
) -> RegistrationInputs:
    """Obtain both reference geometries, deriving or reading them as available.

    Two disjoint groups of processed assets exist, and neither source covers both:

    * Older assets published the reference volumes but recorded the registration under
      the v1 schema, whose ``parameters`` are empty. Nothing can be derived; the headers
      are read instead.
    * Newer assets record a full v2 pass but publish no volumes. Nothing can be read; the
      geometry is derived.

    Where an asset has both, the derivation is checked against the headers rather than
    assumed, so the two paths are known to agree wherever that can be tested.

    Parameters
    ----------
    client : object
        An S3 client.
    bucket : str
        Bucket holding the dataset.
    dataset : str
        Processed dataset name.
    dataset_id : str
        Numeric subject id.

    Returns
    -------
    RegistrationInputs
        Both geometries and the template-to-CCF asset the registration used.

    Raises
    ------
    RegistrationRecordError
        If the geometry can be neither derived nor read.
    """
    record = _load_record(client, bucket, dataset)
    asset = find_template_to_ccf_asset(record) if record is not None else None
    if asset is None:
        logger.warning(
            "No template-to-CCF asset named in the record for %s; the caller's default "
            "will be used, which may not be the version this sample was registered with",
            dataset,
        )
    else:
        logger.info("Registration used template-to-CCF asset %s", asset)

    derived = _derive_geometry(client, bucket, record)
    published = tuple(
        _published_geometry(client, bucket, registration_volume_key(dataset, dataset_id, name))
        for name in ("loaded", "resampled")
    )

    if derived is not None and all(p is not None for p in published):
        for name, from_record, from_header in zip(("loaded", "resampled"), derived, published):
            reconcile(from_record, from_header, name)
        logger.info("Derived geometry matches both published volumes")
        return RegistrationInputs(derived[0], derived[1], asset)
    if derived is not None:
        logger.info("No published volumes; using the geometry derived from the record")
        return RegistrationInputs(derived[0], derived[1], asset)
    if all(p is not None for p in published):
        logger.info("Registration record is unusable; read geometry from published headers")
        return RegistrationInputs(published[0], published[1], asset)
    raise RegistrationRecordError(
        f"Cannot obtain reference geometry for {dataset}: the registration record does "
        "not describe a 10 um pass and the reference volumes were not published"
    )


def _load_record(client, bucket: str, dataset: str) -> dict | None:
    """Fetch the registration's own record, if it exists.

    Parameters
    ----------
    client : object
        An S3 client.
    bucket : str
        Bucket holding the dataset.
    dataset : str
        Processed dataset name.

    Returns
    -------
    dict | None
        The decoded record, or ``None`` when it cannot be read.
    """
    try:
        body = client.get_object(Bucket=bucket, Key=f"{dataset}/{PROCESSING_RECORD}")["Body"]
    except Exception as error:  # noqa: BLE001 - an absent record is an expected case
        logger.info("Cannot read the registration record (%s)", type(error).__name__)
        return None
    return json.loads(body.read())


def _derive_geometry(
    client, bucket: str, record: dict | None
) -> tuple[VolumeGeometry, VolumeGeometry] | None:
    """Derive both geometries from the registration record, if it is usable.

    Parameters
    ----------
    client : object
        An S3 client, for reading the zarr level's ``.zarray``.
    bucket : str
        Bucket holding the dataset.
    record : dict | None
        The decoded registration record.

    Returns
    -------
    tuple[VolumeGeometry, VolumeGeometry] | None
        The geometries, or ``None`` when the record does not describe a 10 um pass.
    """
    if record is None:
        return None
    try:
        pass_ = parse_registration_record(record)
        zarray = json.loads(
            client.get_object(Bucket=bucket, Key=zarr_level_key(pass_))["Body"].read()
        )
    except RegistrationRecordError as error:
        logger.info("Cannot derive geometry from the record: %s", error)
        return None
    except Exception as error:  # noqa: BLE001 - an unreadable zarr level is expected
        logger.info("Cannot read the zarr level (%s)", type(error).__name__)
        return None

    loaded = loaded_geometry(pass_, zarr_shape(zarray))
    resampled = resampled_geometry(loaded)
    logger.info(
        "Derived geometry from level %d of %s: loaded %s, resampled %s",
        pass_.level,
        pass_.input_uri,
        loaded.shape,
        resampled.shape,
    )
    return loaded, resampled


def build_reference_images(
    ccf_path: str,
    exaspim_template_path: str,
    loaded: VolumeGeometry,
    resampled: VolumeGeometry,
) -> ReferenceImages:
    """Assemble what ``load_images`` would have returned.

    The two templates are read from disk. The two reference volumes are not: they become
    a shape and a geometry, which is all their consumers read.

    Parameters
    ----------
    ccf_path : str
        Path to the CCF template.
    exaspim_template_path : str
        Path to the exaSPIM template.
    loaded : VolumeGeometry
        Geometry of the volume the registration read from the zarr.
    resampled : VolumeGeometry
        Geometry of the isotropically resampled volume.

    Returns
    -------
    ReferenceImages
        The images to pass to the transform.
    """
    import ants

    ccf = ants.image_read(ccf_path)
    exaspim_template = ants.image_read(exaspim_template_path)
    # load_images does this: the template inherits the CCF's geometry outright.
    exaspim_template.set_spacing(ccf.spacing)
    exaspim_template.set_origin(ccf.origin)
    exaspim_template.set_direction(ccf.direction)

    brain, resampled_array = reference_arrays(loaded, resampled)
    resampled_image = apply_geometry(
        ants.from_numpy(np.zeros((1, 1, 1), dtype="float32")), resampled
    )
    return ReferenceImages(
        ccf=ccf,
        exaspim_template=exaspim_template,
        brain=brain,
        resampled=resampled_array,
        resampled_image=resampled_image,
    )

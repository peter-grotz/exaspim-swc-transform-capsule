"""Build the images ``RegistrationPipeline`` needs without reading the reference volumes.

The CCF and exaSPIM templates are read from their mounted assets. The registration's two
reference volumes are replaced by stand-ins carrying only their geometry; see
:mod:`exaspim_swc_processing.reference`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np

from exaspim_swc_processing.reference import apply_geometry, reference_arrays
from exaspim_swc_processing.registration import (
    HEADER_FETCH_BYTES,
    RegistrationRecordError,
    VolumeGeometry,
    loaded_geometry,
    parse_nifti_geometry,
    parse_registration_record,
    reconcile,
    registration_volume_key,
    resampled_geometry,
    zarr_level_key,
    zarr_shape,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReferenceImages:
    """What ``preprocess_coords`` and ``apply_transforms_to_points`` consume."""

    ccf: object
    exaspim_template: object
    brain: np.ndarray
    resampled: np.ndarray
    resampled_image: object


def disable_overlay_normalization() -> None:
    """Make ``ImageVisualizer.perc_normalization`` a pass-through.

    ``preprocess_coords`` calls it on both reference volumes even when no overlay is
    drawn, and it fails on the constant stand-ins. No coordinate depends on it.
    """
    from aind_exaspim_register_cells.visualization import ImageVisualizer

    ImageVisualizer.perc_normalization = staticmethod(lambda image, *_args, **_kwargs: image)


def _published_geometry(client, bucket: str, key: str) -> VolumeGeometry | None:
    """Read a published volume's geometry from its header, or ``None`` if absent."""
    try:
        body = client.get_object(Bucket=bucket, Key=key, Range=f"bytes=0-{HEADER_FETCH_BYTES - 1}")
    except Exception as error:  # noqa: BLE001 - absence is the common case
        logger.info("No published volume at %s (%s)", key, type(error).__name__)
        return None
    return parse_nifti_geometry(body["Body"].read())


def _derive_geometry(
    client, bucket: str, record: dict
) -> tuple[VolumeGeometry, VolumeGeometry] | None:
    """Derive both geometries from the registration record, or ``None`` if it cannot."""
    try:
        pass_ = parse_registration_record(record)
        zarray = json.loads(
            client.get_object(Bucket=bucket, Key=zarr_level_key(pass_))["Body"].read()
        )
    except RegistrationRecordError as error:
        logger.info("Cannot derive geometry from the record: %s", error)
        return None
    except Exception as error:  # noqa: BLE001 - an unreadable zarr header is the same failure
        logger.info("Cannot read the registered zarr's header (%s)", type(error).__name__)
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


def resolve_geometry(
    client, bucket: str, dataset: str, subject_id: str, record: dict
) -> tuple[VolumeGeometry, VolumeGeometry]:
    """Return the loaded and resampled geometries, from the record or the published headers.

    Older records cannot be derived from, and newer datasets publish no volumes. Where
    both are available they are checked against each other.

    Raises
    ------
    RegistrationRecordError
        If the geometry can be neither derived nor read.
    """
    derived = _derive_geometry(client, bucket, record)
    published = tuple(
        _published_geometry(client, bucket, registration_volume_key(dataset, subject_id, name))
        for name in ("loaded", "resampled")
    )
    if derived is not None and all(p is not None for p in published):
        for name, from_record, from_header in zip(("loaded", "resampled"), derived, published):
            reconcile(from_record, from_header, name)
        logger.info("Derived geometry matches both published volumes")
        return derived
    if derived is not None:
        return derived
    if all(p is not None for p in published):
        logger.info("Using the geometry in the published volume headers")
        return published[0], published[1]
    raise RegistrationRecordError(
        f"Cannot obtain reference geometry for {dataset}: the registration record does "
        "not describe a 10 um pass and the reference volumes were not published"
    )


def build_reference_images(
    ccf_path: str,
    exaspim_template_path: str,
    loaded: VolumeGeometry,
    resampled: VolumeGeometry,
) -> ReferenceImages:
    """Read the templates and build geometry-only stand-ins for the reference volumes."""
    import ants

    ccf = ants.image_read(ccf_path)
    exaspim_template = ants.image_read(exaspim_template_path)
    # As load_images does: the template takes the CCF's geometry.
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

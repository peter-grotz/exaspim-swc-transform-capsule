"""Transform exaSPIM SWC reconstructions from specimen space into CCF space."""

import argparse
import logging
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from aind_exaspim_register_cells import RegistrationPipeline
from allensdk.core.swc import Compartment, Morphology
from exaspim_swc_transform.io_swc import read_swc
from exaspim_swc_transform.reference import (
    ReferenceImages,
    build_reference_images,
    disable_overlay_normalization,
    resolve_geometry,
)
from exaspim_swc_transform.s3_stage import (
    BUCKET_DEFAULT,
    resolve_dataset,
    s3_client,
    stage_registration_bundle,
)
from exaspim_swc_transform.template_selection import LegacySampleError, select_template_to_ccf
from exaspim_swc_transform.transform_resolution import ResolvedInputs, resolve_inputs

from exaspim_swc_processing.acquisition import AcquisitionNotFoundError, resolve_acquisition
from exaspim_swc_processing.naming import ReconstructionNameError, parse_stem
from exaspim_swc_processing.registration import dataset_name
from exaspim_swc_processing.stage import (
    UPSTREAM_STAGES,
    build_stage_process,
    carry_forward,
    resolve_code,
    write_stage_process,
)

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/results"))
OUTPUT_STAGE = "alignment"
STEP_NAME = "exaspim_swc_transform"
VOXEL_SPACE_DIR = "final-voxel"
"""Refinement output in voxel units; not a valid transform input."""

WORLD_SPACE_DIR = "final-world"
"""Refinement output in physical units, which the registration expects."""

CCF_RESOLUTION_UM = 10.0
"""Edge length of a CCF template voxel. Transformed points are indices; this scales to um."""

logger = logging.getLogger(STEP_NAME)


def parse_args() -> argparse.Namespace:
    """Read the App Builder parameters.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swc-dir", default=os.environ.get("SWC_DIR", ""))
    parser.add_argument("--processed-dataset", default=os.environ.get("PROCESSED_DATASET", ""))
    parser.add_argument("--df-asset", default=os.environ.get("DF_ASSET", ""))
    parser.add_argument("--experimenters", default=os.environ.get("EXPERIMENTERS", ""))
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        default=os.environ.get("FAIL_FAST", "").lower() in {"1", "true", "yes"},
        help="Abort on the first SWC that fails rather than skipping it.",
    )
    return parser.parse_args()


def transform_one(
    swc_path: Path,
    pipeline: RegistrationPipeline,
    images: ReferenceImages,
    destination: Path,
) -> None:
    """Transform a single reconstruction into CCF space and write it.

    Parameters
    ----------
    swc_path : Path
        Reconstruction in specimen space.
    pipeline : RegistrationPipeline
        The registration pipeline, already holding the loaded transforms.
    images : ReferenceImages
        Templates and the geometry-only stand-ins for the reference volumes.
    destination : Path
        Where to write the transformed reconstruction.
    """
    morph = read_swc(swc_path, add_offset=True)
    coords = np.array([[c["x"], c["y"], c["z"]] for c in morph.compartment_list])
    # cell_filename=None skips ImageVisualizer overlay rendering, which is the only
    # consumer of voxel data in this stage.
    prepped = pipeline.preprocess_coords(coords, images.brain, images.resampled, None)
    idx_pts, _ = pipeline.apply_transforms_to_points(
        prepped, images.resampled_image, images.exaspim_template, images.ccf, None
    )
    transformed = []
    for index, node in enumerate(morph.compartment_list):
        compartment = Compartment(**node)
        compartment["x"] = idx_pts[index][0] * CCF_RESOLUTION_UM
        compartment["y"] = idx_pts[index][1] * CCF_RESOLUTION_UM
        compartment["z"] = idx_pts[index][2] * CCF_RESOLUTION_UM
        transformed.append(compartment)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Morphology(transformed).save(str(destination))


def cell_subjects(swc_dir: Path) -> set[str]:
    """Return the subject ids named by the reconstructions to be transformed.

    Parameters
    ----------
    swc_dir : Path
        Directory of reconstructions, named ``<neuron>-<subject>-<annotator>.swc``.

    Returns
    -------
    set[str]
        Distinct subject ids. An unparseable name contributes ``"?<stem>"`` so it is
        reported rather than silently ignored.
    """
    subjects: set[str] = set()
    for path in swc_dir.rglob("*.swc"):
        try:
            subjects.add(parse_stem(path.stem).subject_id)
        except ReconstructionNameError:
            subjects.add(f"?{path.stem}")
    return subjects


def scratch_dir() -> Path:
    """Return this stage's scratch directory, created if needed.

    Returns
    -------
    Path
        ``/scratch`` under Code Ocean, ``/tmp`` elsewhere.
    """
    root = Path("/scratch") if Path("/scratch").is_dir() else Path("/tmp")
    path = root / "exaspim_swc_transform"
    path.mkdir(parents=True, exist_ok=True)
    return path


def reconstruction_root(swc_dir: Path) -> Path:
    """Return the mounted asset directory that holds the upstream stage outputs.

    A reconstruction asset connected to the pipeline mounts at ``/data/<asset>/``, so its
    ``refinement/`` and ``dispatch/`` sit one level below ``/data`` rather than directly
    in it. Walking up from the reconstructions finds that directory without assuming the
    asset's name.

    Parameters
    ----------
    swc_dir : Path
        Directory of specimen-space reconstructions, e.g.
        ``/data/<asset>/refinement/final-voxel``.

    Returns
    -------
    Path
        The nearest ancestor holding a stage directory, or ``DATA_DIR`` if none does.
    """
    for candidate in (swc_dir, *swc_dir.parents):
        if any((candidate / stage).is_dir() for stage in UPSTREAM_STAGES):
            return candidate
    return DATA_DIR


def locate_bundle(spec: str) -> str:
    """Return a local directory holding the registration bundle.

    Parameters
    ----------
    spec : str
        A mounted directory, an S3 URI, a dataset name, or a subject id.

    Returns
    -------
    str
        Path to the bundle, staged from S3 when it was not already local.
    """
    if spec and Path(spec).is_dir():
        return spec
    return stage_registration_bundle(spec)


def locate_dataset(spec: str, bundle: str) -> tuple[str, str] | None:
    """Identify the processed dataset the registration record belongs to.

    Parameters
    ----------
    spec : str
        Whatever was passed as ``--processed-dataset``.
    bundle : str
        The local bundle directory, which may carry the name in its path.

    Returns
    -------
    tuple[str, str] | None
        Bucket and dataset name, or ``None`` when neither can be determined.
    """
    bucket = urlparse(spec).netloc or BUCKET_DEFAULT
    # A mounted path or URI carries the dataset name; a bare subject id needs a lookup.
    dataset = dataset_name(spec, bundle)
    if dataset:
        return bucket, dataset
    try:
        return resolve_dataset(spec)
    except (ValueError, FileNotFoundError) as error:
        logger.error("Could not determine the processed dataset from %r: %s", spec, error)
        return None


def load_reference_images(resolved: ResolvedInputs, bucket: str, dataset: str) -> ReferenceImages:
    """Assemble the images the transform reads, without loading reference volumes.

    ``RegistrationPipeline.load_images`` would read two NIfTI volumes of 1.4-1.8 GB
    whose voxels nothing consumes, and which many datasets never published. Only their
    shape and geometry are used, so both are reconstructed from the registration record.

    Parameters
    ----------
    resolved : ResolvedInputs
        Resolved input paths.
    bucket : str
        Bucket holding the processed dataset.
    dataset : str
        Processed dataset name.

    Returns
    -------
    ReferenceImages
        Templates plus geometry-only stand-ins.
    """
    disable_overlay_normalization()
    loaded, resampled = resolve_geometry(s3_client(), bucket, dataset, resolved.dataset_id)
    return build_reference_images(
        resolved.ccf_path, resolved.exaspim_template_path, loaded, resampled
    )


def transform_all(
    swc_dir: Path,
    pipeline: RegistrationPipeline,
    images: ReferenceImages,
    destination: Path,
    fail_fast: bool,
) -> tuple[int, list[str]]:
    """Transform every reconstruction found under a directory.

    Parameters
    ----------
    swc_dir : Path
        Directory of specimen-space reconstructions.
    pipeline : RegistrationPipeline
        The registration pipeline.
    images : ReferenceImages
        Templates and reference geometry.
    destination : Path
        Directory to write CCF-space reconstructions to.
    fail_fast : bool
        Abort on the first failure rather than skipping it.

    Returns
    -------
    tuple[int, list[str]]
        How many were found, and the stems that failed.
    """
    swc_paths = sorted(swc_dir.rglob("*.swc"))
    logger.info("Transforming %d reconstruction(s)", len(swc_paths))
    failures: list[str] = []
    for swc_path in swc_paths:
        try:
            transform_one(swc_path, pipeline, images, destination / f"{swc_path.stem}.swc")
        except Exception as error:  # noqa: BLE001 - one bad cell must not lose the run
            if fail_fast:
                raise
            logger.warning("Failed to transform %s: %s", swc_path.name, error)
            failures.append(swc_path.stem)
    return len(swc_paths), failures


def carry_acquisition(source: str, output_root: Path) -> bool:
    """Republish ``acquisition.json`` for the resample stage.

    The resample stage converts specimen-space reconstructions between voxel and
    physical units using the acquisition's ``coordinate_transformations``, and this is
    the only stage that has the file.

    Parameters
    ----------
    source : str
        Path to the acquisition file.
    output_root : Path
        This stage's output directory.

    Returns
    -------
    bool
        Whether the file was carried forward.
    """
    path = Path(source)
    if not path.is_file():
        logger.warning("No acquisition file at %s; downstream scaling will be derived", path)
        return False
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, output_root / "acquisition.json")
    return True


def run() -> int:
    """Transform every reconstruction found, then record the stage.

    Returns
    -------
    int
        Process exit status. Non-zero when any reconstruction failed.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    started = datetime.now(timezone.utc)

    swc_dir = Path(args.swc_dir)
    if not args.swc_dir or not swc_dir.is_dir():
        logger.error("--swc-dir must name a directory of reconstructions; got %r", args.swc_dir)
        return 1
    # The registration converts physical coordinates to voxels itself, dividing x and y by
    # the 0.748 anisotropy. Voxel-space input is therefore converted twice and lands scaled
    # 1.337x in-plane and displaced -- plausibly shaped, so it would not fail on its own.
    if swc_dir.name == VOXEL_SPACE_DIR:
        logger.error(
            "--swc-dir points at %s/, which is in voxel units; transform needs the "
            "physical-space reconstructions in %s/: %s",
            VOXEL_SPACE_DIR,
            WORLD_SPACE_DIR,
            swc_dir.parent / WORLD_SPACE_DIR,
        )
        return 1

    bundle = locate_bundle(args.processed_dataset)
    located = locate_dataset(args.processed_dataset, bundle)
    if located is None:
        return 1
    bucket, dataset = located

    # Every cell must belong to the sample whose registration is applied to it; a cell from
    # another subject would be transformed with the wrong warp and still look plausible.
    sample = re.search(r"exaSPIM_(\d{6})_", dataset)
    sample_id = sample.group(1) if sample else ""
    subjects = cell_subjects(swc_dir)
    if subjects != {sample_id}:
        logger.error(
            "Reconstructions in %s name subject(s) %s, but the registration is for %s",
            swc_dir,
            ", ".join(sorted(subjects)) or "none",
            sample_id or dataset,
        )
        return 1

    # Pre-v1.5 samples keep the template-to-CCF version their CCF coordinates were
    # originally produced with; legacy samples, whose version is unknown, are refused.
    try:
        template = select_template_to_ccf(sample_id)
    except LegacySampleError as error:
        logger.error("%s", error)
        return 1
    logger.info("Template-to-CCF: %s (%s)", template.asset, template.basis)

    # Nextflow hands the next stage only this stage's results, so the upstream outputs
    # have to be republished or they leave the chain.
    carried = carry_forward(reconstruction_root(swc_dir), RESULTS_DIR, UPSTREAM_STAGES)
    logger.info("Carried forward: %s", ", ".join(carried) or "nothing")

    resolved = resolve_inputs(Path(bundle), args.df_asset, template_to_ccf_asset=template.asset)
    output_root = RESULTS_DIR / OUTPUT_STAGE
    swc_out_dir = output_root / "aligned_swcs"

    # The registry holds what the indexer published; a staged bundle may carry a stale
    # copy or none. check_orientation reads the axes from this, so a wrong one
    # mis-registers every cell without failing.
    try:
        acquisition_path, acquisition_source = resolve_acquisition(
            dataset,
            scratch_dir() / "acquisition.json",
            local_fallback=resolved.acquisition_file,
        )
    except AcquisitionNotFoundError as error:
        logger.error("%s", error)
        return 1
    logger.info(
        "Acquisition from %s", acquisition_source.value if acquisition_source else "staged bundle"
    )

    # RegistrationPipeline requires an output_dir but writes there only for the overlays,
    # which load_reference_images disables.
    debug_dir = scratch_dir() / "overlays"
    debug_dir.mkdir(parents=True, exist_ok=True)

    pipeline = RegistrationPipeline(
        dataset_id=resolved.dataset_id,
        output_dir=str(debug_dir),
        acquisition_file=str(acquisition_path),
        brain_path=resolved.brain_path,
        resampled_brain_path=resolved.resampled_brain_path,
        brain_to_exaspim_transform_path=resolved.brain_to_exaspim_transform_path,
        exaspim_to_ccf_transform_path=resolved.exaspim_to_ccf_transform_path,
        ccf_path=resolved.ccf_path,
        exaspim_template_path=resolved.exaspim_template_path,
        transform_res=[CCF_RESOLUTION_UM] * 3,
        level=2,
        manual_transform_path=resolved.manual_transform_path,
    )
    images = load_reference_images(resolved, bucket, dataset)

    found, failures = transform_all(swc_dir, pipeline, images, swc_out_dir, args.fail_fast)
    shutil.rmtree(debug_dir, ignore_errors=True)
    transformed = found - len(failures)
    acquisition_carried = carry_acquisition(str(acquisition_path), output_root)

    write_stage_process(
        build_stage_process(
            STEP_NAME,
            resolve_code(
                "exaspim-swc-transform",
                url="https://github.com/peter-grotz/exaspim-swc-transform-capsule",
            ),
            start_time=started,
            output_path=OUTPUT_STAGE,
            parameters={
                "swc_dir": args.swc_dir,
                "processed_dataset": args.processed_dataset,
                "df_asset": args.df_asset,
            },
            output_parameters={
                "aligned_swc_dir": str(swc_out_dir),
                "input_swc_count": found,
                "transformed_swc_count": transformed,
                "failed": failures,
                "dataset_id": resolved.dataset_id,
                "acquisition_carried_forward": acquisition_carried,
                "acquisition_source": acquisition_source.value if acquisition_source else "staged",
                "stages_carried_forward": carried,
                "template_to_ccf": template.asset,
                "template_to_ccf_basis": template.basis,
            },
            experimenters=[e.strip() for e in args.experimenters.split(",") if e.strip()],
            notes=f"Transformed {transformed} of {found} reconstructions into CCF space.",
        ),
        output_root,
    )

    logger.info("Transformed %d of %d", transformed, found)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())

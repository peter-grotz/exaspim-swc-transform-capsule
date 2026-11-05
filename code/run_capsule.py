"""Transform exaSPIM SWC reconstructions from specimen space into CCF space."""

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from aind_exaspim_register_cells import RegistrationPipeline
from allensdk.core.swc import Compartment, Morphology
from exaspim_swc_transform.displacement import (
    AmbiguousDisplacementFieldError,
    fetch_displacement_field,
)
from exaspim_swc_transform.io_swc import read_swc
from exaspim_swc_transform.reference import (
    ReferenceImages,
    build_reference_images,
    disable_overlay_normalization,
    resolve_geometry,
)
from exaspim_swc_transform.s3_stage import BUCKET_DEFAULT, s3_client, stage_transforms

from exaspim_swc_processing.acquisition import AcquisitionNotFoundError, resolve_acquisition
from exaspim_swc_processing.datasets import (
    DatasetMismatchError,
    DatasetNotFoundError,
    ProcessedDataset,
    registry_sources,
    resolve_processed_dataset,
)
from exaspim_swc_processing.naming import ReconstructionNameError, parse_stem
from exaspim_swc_processing.registration import (
    PROCESSING_RECORD,
    RegistrationRecordError,
    parse_registration_record,
    recorded_template_to_ccf,
)
from exaspim_swc_processing.stage import (
    PROCESSED_DATASET_FILE,
    build_stage_process,
    carry_forward,
    resolve_code,
    write_stage_process,
)

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/results"))
SCRATCH_DIR = Path(os.environ.get("SCRATCH_DIR", "/scratch"))
OUTPUT_STAGE = "alignment"
STEP_NAME = "exaspim_swc_transform"
CARRIED_STAGES = ("dispatch", "refinement")
"""Upstream stages republished for packaging. Never this stage's own outputs."""

WORLD_SPACE_DIR = "final-world"
"""Refinement output in microns, the only valid input."""

CCF_TEMPLATE = DATA_DIR / "allen_mouse_ccf/average_template/average_template_10.nii.gz"
EXASPIM_TEMPLATE = (
    DATA_DIR / "exaspim_template_7subjects_nomask_10um_round6_template_only/fixed_median.nii.gz"
)
CCF_RESOLUTION_UM = 10.0
"""Edge length of a CCF template voxel. Transformed points are indices; this scales to um."""

REGISTRATION_LEVEL = 2
"""Pyramid level of the 10 um registration pass, which the registration capsule hard-codes."""

logger = logging.getLogger(STEP_NAME)


def parse_args() -> argparse.Namespace:
    """Read the App Builder parameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swc-dir", default=os.environ.get("SWC_DIR", ""))
    parser.add_argument("--processed-dataset", default=os.environ.get("PROCESSED_DATASET", ""))
    parser.add_argument("--experimenters", default=os.environ.get("EXPERIMENTERS", ""))
    parser.add_argument(
        "--allow-no-displacement-field",
        default=os.environ.get("ALLOW_NO_DISPLACEMENT_FIELD", ""),
        help='Set to "True" to transform without a manual CCF refinement when none exists.',
    )
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
    """Transform one reconstruction into CCF space and write it."""
    morph = read_swc(swc_path)
    coords = np.array([[c["x"], c["y"], c["z"]] for c in morph.compartment_list])
    # cell_filename=None skips the overlay rendering.
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


def cell_subjects(swc_paths: list[Path]) -> set[str]:
    """Subject ids named by the reconstructions; an unparseable name gives ``"?<stem>"``."""
    subjects: set[str] = set()
    for path in swc_paths:
        try:
            subjects.add(parse_stem(path.stem).subject_id)
        except ReconstructionNameError:
            subjects.add(f"?{path.stem}")
    return subjects


def reconstruction_root(swc_dir: Path) -> Path:
    """The mounted asset holding ``refinement/`` and ``dispatch/``, found by walking up."""
    for candidate in (swc_dir, *swc_dir.parents):
        if any((candidate / stage).is_dir() for stage in CARRIED_STAGES):
            return candidate
    return DATA_DIR


def recorded_image_path(asset_root: Path) -> str:
    """The image the reconstructions were traced on, from ``refinement/data_process.json``."""
    record = asset_root / "refinement" / "data_process.json"
    try:
        payload = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.info("No readable %s; the traced image is unknown", record)
        return ""
    return str(((payload.get("code") or {}).get("parameters") or {}).get("image_path") or "")


def locate_dataset(client, spec: str, image_path: str) -> ProcessedDataset | None:
    """Resolve the processed dataset through DocDB, falling back to S3."""
    try:
        dataset = resolve_processed_dataset(
            spec, registry_sources(), client, BUCKET_DEFAULT, image_path
        )
    except (DatasetNotFoundError, DatasetMismatchError) as error:
        logger.error("%s", error)
        return None
    logger.info(
        "Processed dataset %s (subject %s) resolved from %s",
        dataset.name,
        dataset.subject_id,
        dataset.source,
    )
    return dataset


def read_registration_record(client, dataset: ProcessedDataset) -> dict | None:
    """The dataset's ``ccf_alignment/processing.json``, or ``None`` (logged) if unreadable."""
    key = f"{dataset.name}/{PROCESSING_RECORD}"
    try:
        return json.loads(client.get_object(Bucket=dataset.bucket, Key=key)["Body"].read())
    except Exception as error:  # noqa: BLE001 - an absent or unreadable record is one failure
        logger.error("Cannot read s3://%s/%s (%s)", dataset.bucket, key, type(error).__name__)
        return None


def recorded_level(record: dict) -> int:
    """The level the 10 um pass records, or the hard-coded level if the record predates it."""
    try:
        return parse_registration_record(record).level
    except RegistrationRecordError:
        return REGISTRATION_LEVEL


def existing(path: Path, what: str) -> str | None:
    """``path`` as a string if it is a file, else ``None`` (logged)."""
    if path.is_file():
        return str(path)
    logger.error("Missing %s: %s", what, path)
    return None


def transform_all(
    swc_paths: list[Path],
    pipeline: RegistrationPipeline,
    images: ReferenceImages,
    destination: Path,
    fail_fast: bool,
) -> list[str]:
    """Transform every reconstruction and return the stems that failed."""
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
    return failures


def run() -> int:
    """Transform every reconstruction, then record the stage. Returns the exit status."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    started = datetime.now(timezone.utc)
    allow_no_field = args.allow_no_displacement_field.strip().lower() == "true"

    swc_dir = Path(args.swc_dir)
    if swc_dir.name != WORLD_SPACE_DIR or not swc_dir.is_dir():
        logger.error("--swc-dir must be a %s/ directory; got %r", WORLD_SPACE_DIR, args.swc_dir)
        return 1
    swc_paths = sorted(swc_dir.glob("*.swc"))
    asset_root = reconstruction_root(swc_dir)

    client = s3_client()
    dataset = locate_dataset(client, args.processed_dataset, recorded_image_path(asset_root))
    if dataset is None:
        return 1
    subject_id = dataset.subject_id

    # A cell from another subject would get the wrong warp and still look plausible.
    subjects = cell_subjects(swc_paths)
    if subjects != {subject_id}:
        logger.error(
            "Reconstructions in %s name subject(s) %s, but the registration is for %s",
            swc_dir,
            ", ".join(sorted(subjects)) or "none",
            subject_id,
        )
        return 1

    record = read_registration_record(client, dataset)
    if record is None:
        return 1
    level = recorded_level(record)
    if level != REGISTRATION_LEVEL:
        logger.error(
            "The registration record states level %d; this stage supports only level %d",
            level,
            REGISTRATION_LEVEL,
        )
        return 1
    try:
        template = recorded_template_to_ccf(record)
    except RegistrationRecordError as error:
        logger.error("s3://%s/%s: %s", dataset.bucket, dataset.name, error)
        return 1
    logger.info("Template-to-CCF from the registration record: %s", template)

    try:
        displacement = fetch_displacement_field(
            client, dataset.bucket, dataset.name, SCRATCH_DIR / "displacement" / dataset.name
        )
    except AmbiguousDisplacementFieldError as error:
        logger.error("%s", error)
        return 1
    if displacement is None and not allow_no_field:
        logger.error(
            'No displacement field for %s; set allow_no_displacement_field to "True" '
            "to transform without one",
            dataset.name,
        )
        return 1

    template_dir = DATA_DIR / template
    template_to_ccf = [
        existing(template_dir / "0GenericAffine.mat", "template-to-CCF affine"),
        existing(template_dir / "1InverseWarp.nii.gz", "template-to-CCF inverse warp"),
    ]
    ccf_path = existing(CCF_TEMPLATE, "CCF template")
    exaspim_template_path = existing(EXASPIM_TEMPLATE, "exaSPIM template")
    if None in (*template_to_ccf, ccf_path, exaspim_template_path):
        logger.error("Are the template assets, including %s, connected to this process?", template)
        return 1

    try:
        acquisition_path, acquisition_source = resolve_acquisition(
            dataset.name, SCRATCH_DIR / "acquisition.json"
        )
    except AcquisitionNotFoundError as error:
        logger.error("%s", error)
        return 1
    logger.info("Acquisition from %s", acquisition_source.value)

    carried = carry_forward(asset_root, RESULTS_DIR, CARRIED_STAGES)
    logger.info("Carried forward: %s", ", ".join(carried) or "nothing")

    # RegistrationPipeline requires an output directory; nothing used is written there.
    scratch_output = SCRATCH_DIR / "registration_pipeline"
    scratch_output.mkdir(parents=True, exist_ok=True)
    pipeline = RegistrationPipeline(
        dataset_id=subject_id,
        output_dir=str(scratch_output),
        acquisition_file=str(acquisition_path),
        brain_path="",
        resampled_brain_path="",
        brain_to_exaspim_transform_path=stage_transforms(
            client,
            dataset.bucket,
            dataset.name,
            subject_id,
            SCRATCH_DIR / "transforms" / dataset.name,
        ),
        exaspim_to_ccf_transform_path=template_to_ccf,
        ccf_path=ccf_path,
        exaspim_template_path=exaspim_template_path,
        transform_res=[CCF_RESOLUTION_UM] * 3,
        level=level,
        manual_transform_path=[displacement.local_path] if displacement else [],
    )
    disable_overlay_normalization()
    loaded, resampled = resolve_geometry(client, dataset.bucket, dataset.name, subject_id, record)
    images = build_reference_images(ccf_path, exaspim_template_path, loaded, resampled)

    output_root = RESULTS_DIR / OUTPUT_STAGE
    failures = transform_all(
        swc_paths, pipeline, images, output_root / "aligned_swcs", args.fail_fast
    )
    shutil.rmtree(scratch_output, ignore_errors=True)

    field = displacement.source if displacement else None
    applied = [f"s3://{dataset.bucket}/{dataset.name}/ccf_alignment/", str(template_dir)]
    if field:
        applied.append(field)
    write_stage_process(
        build_stage_process(
            STEP_NAME,
            resolve_code(url="https://github.com/peter-grotz/exaspim-swc-transform-capsule"),
            started,
            experimenters=[e.strip() for e in args.experimenters.split(",") if e.strip()],
            input_data=applied,
            parameters={"displacement_field": field},
        ),
        output_root,
    )
    # Not metadata: tells packaging which dataset is the parent.
    (output_root / PROCESSED_DATASET_FILE).write_text(dataset.name, encoding="utf-8")

    logger.info("Transformed %d of %d", len(swc_paths) - len(failures), len(swc_paths))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())

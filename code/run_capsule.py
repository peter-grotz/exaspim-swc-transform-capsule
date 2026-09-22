"""Transform exaSPIM SWC reconstructions from specimen space into CCF space."""

import argparse
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from allensdk.core.swc import Compartment, Morphology
from aind_exaspim_register_cells import RegistrationPipeline
from exaspim_swc_processing.stage import (
    UPSTREAM_STAGES,
    build_stage_process,
    carry_forward,
    resolve_code,
    write_stage_process,
)
from exaspim_swc_transform.io_swc import read_swc
from exaspim_swc_transform.reference import (
    build_reference_images,
    disable_overlay_normalization,
    resolve_geometry,
)
from exaspim_swc_processing.registration import dataset_name
from exaspim_swc_transform.s3_stage import BUCKET_DEFAULT as BUCKET
from exaspim_swc_transform.s3_stage import resolve_dataset, s3_client
from exaspim_swc_transform.transform_resolution import resolve_inputs

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/results"))
OUTPUT_STAGE = "alignment"
STEP_NAME = "exaspim_swc_transform"
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
    images: tuple,
    destination: Path,
) -> None:
    """Transform a single reconstruction into CCF space and write it.

    Parameters
    ----------
    swc_path : Path
        Reconstruction in specimen space.
    pipeline : RegistrationPipeline
        The registration pipeline, already holding the loaded transforms.
    images : tuple
        ``(ccf, ants_exaspim, resampled_img, brain_np, resampled_np)``.
    destination : Path
        Where to write the transformed reconstruction.
    """
    ccf, ants_exaspim, resampled_img, brain_np, resampled_np = images
    morph = read_swc(swc_path, add_offset=True)
    coords = np.array([[c["x"], c["y"], c["z"]] for c in morph.compartment_list])
    # cell_filename=None skips ImageVisualizer overlay rendering, which is the only
    # consumer of voxel data in this stage.
    prepped = pipeline.preprocess_coords(coords, brain_np, resampled_np, None)
    idx_pts, _ = pipeline.apply_transforms_to_points(
        prepped, resampled_img, ants_exaspim, ccf, None
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

    transform_dir = args.processed_dataset
    if transform_dir and not Path(transform_dir).is_dir():
        from exaspim_swc_transform.s3_stage import stage_registration_bundle

        transform_dir = stage_registration_bundle(args.processed_dataset)

    # Nextflow hands the next stage only this stage's results, so the upstream outputs
    # have to be republished or they leave the chain.
    carried = carry_forward(DATA_DIR, RESULTS_DIR, UPSTREAM_STAGES)
    logger.info("Carried forward: %s", ", ".join(carried) or "nothing")

    resolved = resolve_inputs(Path(transform_dir), args.df_asset, "")
    output_root = RESULTS_DIR / OUTPUT_STAGE
    swc_out_dir = output_root / "aligned_swcs"

    # RegistrationPipeline requires an output_dir but writes there only for the overlays,
    # which are disabled below.
    scratch = Path("/scratch") if Path("/scratch").is_dir() else Path("/tmp")
    debug_dir = scratch / "exaspim_swc_transform"
    debug_dir.mkdir(parents=True, exist_ok=True)

    pipeline = RegistrationPipeline(
        dataset_id=resolved.dataset_id,
        output_dir=str(debug_dir),
        acquisition_file=resolved.acquisition_file,
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

    # load_images() would read two reference volumes of 1.4-1.8 GB each. Only their shape
    # and geometry are ever read, and 20 of 60 processed assets never published them, so
    # both are reconstructed from the registration's own record instead.
    disable_overlay_normalization()
    # The bundle may be an S3 URI, a dataset name, a sample id, or a Code Ocean mount.
    # All but the sample id carry the dataset name; that one needs an S3 lookup, which
    # cannot parse the other forms.
    bucket = urlparse(args.processed_dataset).netloc or BUCKET
    dataset = dataset_name(args.processed_dataset, str(transform_dir))
    if dataset is None:
        try:
            bucket, dataset = resolve_dataset(args.processed_dataset)
        except (ValueError, FileNotFoundError) as error:
            logger.error(
                "Could not determine the processed dataset from %r: %s",
                args.processed_dataset,
                error,
            )
            return 1
    loaded_geom, resampled_geom = resolve_geometry(
        s3_client(), bucket, dataset, resolved.dataset_id
    )
    reference = build_reference_images(
        resolved.ccf_path, resolved.exaspim_template_path, loaded_geom, resampled_geom
    )
    images = (
        reference.ccf,
        reference.exaspim_template,
        reference.resampled_image,
        reference.brain,
        reference.resampled,
    )

    swc_paths = sorted(Path(args.swc_dir).rglob("*.swc"))
    logger.info("Transforming %d reconstruction(s)", len(swc_paths))
    failures: list[str] = []
    for swc_path in swc_paths:
        try:
            transform_one(swc_path, pipeline, images, swc_out_dir / f"{swc_path.stem}.swc")
        except Exception as error:  # noqa: BLE001 - one bad cell must not lose the run
            if args.fail_fast:
                raise
            logger.warning("Failed to transform %s: %s", swc_path.name, error)
            failures.append(swc_path.stem)

    shutil.rmtree(debug_dir, ignore_errors=True)
    transformed = len(swc_paths) - len(failures)

    # Carry the acquisition forward. The resample stage needs its
    # coordinate_transformations to convert specimen-space reconstructions between voxel
    # and physical units, and this is the only stage that has the file.
    carried_acquisition = None
    acquisition_source = Path(resolved.acquisition_file)
    if acquisition_source.is_file():
        carried_acquisition = output_root / "acquisition.json"
        carried_acquisition.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(acquisition_source, carried_acquisition)
        logger.info("Carried acquisition forward to %s", carried_acquisition)
    else:
        logger.warning("No acquisition file at %s; downstream scaling will be derived",
                       acquisition_source)

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
                "input_swc_count": len(swc_paths),
                "transformed_swc_count": transformed,
                "failed": failures,
                "dataset_id": resolved.dataset_id,
                "acquisition_carried_forward": bool(carried_acquisition),
                "stages_carried_forward": carried,
            },
            experimenters=[e.strip() for e in args.experimenters.split(",") if e.strip()],
            notes=(
                f"Transformed {transformed} of {len(swc_paths)} reconstructions into CCF space."
            ),
        ),
        output_root,
    )

    logger.info("Transformed %d of %d", transformed, len(swc_paths))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())

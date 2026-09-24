"""Locate the registration inputs the transform needs, wherever they were mounted.

The same bundle arrives in several shapes -- staged from S3 into scratch, attached as a
Code Ocean data asset, or laid out under ``ccf_alignment/`` -- and filenames carry the
subject id, so each input is found by trying a short list of candidate paths.

Two inputs are deliberately optional. The reference volumes under
``registration_metadata/`` are published by only some datasets and nothing reads their
voxels; :mod:`exaspim_swc_transform.reference` reconstructs their geometry instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TEMPLATE_TO_CCF_ASSET = "reg_exaspim_template_to_ccf_25um_v1.5"
"""Default template-to-CCF asset; see :mod:`exaspim_swc_transform.template_selection`."""
DEFAULT_CCF_TEMPLATE = "/data/allen_mouse_ccf/average_template/average_template_10.nii.gz"
DEFAULT_EXASPIM_TEMPLATE = (
    "/data/exaspim_template_7subjects_nomask_10um_round6_template_only/fixed_median.nii.gz"
)


class MissingInput(FileNotFoundError):
    """Raised when a required registration input is not present."""


@dataclass(frozen=True)
class ResolvedInputs:
    """Absolute paths to everything ``RegistrationPipeline`` is constructed from.

    Attributes
    ----------
    dataset_id : str
        Subject id, e.g. ``"794492"``.
    acquisition_file : str
        ``acquisition.json``, which supplies the axis orientation.
    brain_path : str
        Loaded reference volume, or ``""`` when the dataset published none.
    resampled_brain_path : str
        Resampled reference volume, or ``""`` when the dataset published none.
    brain_to_exaspim_transform_path : list[str]
        Sample-to-template affine and inverse warp.
    exaspim_to_ccf_transform_path : list[str]
        Template-to-CCF affine and inverse warp.
    manual_transform_path : list[str]
        Manual CCF refinement displacement fields, empty when none was given.
    ccf_path : str
        CCF average template.
    exaspim_template_path : str
        exaSPIM template the sample was registered to.
    """

    dataset_id: str
    acquisition_file: str
    brain_path: str
    resampled_brain_path: str
    brain_to_exaspim_transform_path: list[str]
    exaspim_to_ccf_transform_path: list[str]
    manual_transform_path: list[str]
    ccf_path: str
    exaspim_template_path: str


def _resolve(explicit: str, candidates: list[Path], what: str, *, required: bool = True) -> str:
    """Return an explicit path if given, else the first candidate that exists.

    Parameters
    ----------
    explicit : str
        A path supplied by the caller. When set it must exist.
    candidates : list[Path]
        Paths to try in order.
    what : str
        Name of the input, for error messages.
    required : bool, optional
        When false, absence yields ``""`` instead of raising.

    Returns
    -------
    str
        Absolute path, or ``""`` for an absent optional input.

    Raises
    ------
    MissingInput
        If a required input is nowhere to be found.
    """
    if explicit:
        path = Path(explicit.strip().strip("'\""))
        if not path.is_file():
            raise MissingInput(f"Missing {what}: {path.resolve()}")
        return str(path.resolve())
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    if not required:
        return ""
    tried = "\n".join(f"  - {c}" for c in candidates)
    raise MissingInput(f"Missing {what}. Tried:\n{tried}")


def _bundle_root(transform_dir: Path) -> Path:
    """Return the directory the registration outputs sit directly under.

    Parameters
    ----------
    transform_dir : Path
        Directory the bundle was staged or mounted at.

    Returns
    -------
    Path
        ``<transform_dir>/ccf_alignment`` when present, else ``transform_dir``.
    """
    nested = transform_dir / "ccf_alignment"
    return nested if nested.is_dir() else transform_dir


def resolve_inputs(
    transform_dir: Path,
    dataset_id: str,
    *,
    ccf_template_path: str = DEFAULT_CCF_TEMPLATE,
    exaspim_template_path: str = DEFAULT_EXASPIM_TEMPLATE,
    template_to_ccf_asset: str = TEMPLATE_TO_CCF_ASSET,
    displacement_field: str = "",
) -> ResolvedInputs:
    """Locate every input the transform needs.

    Parameters
    ----------
    transform_dir : Path
        Directory the registration bundle was staged or mounted at.
    dataset_id : str
        Subject id, from the registry record. It names the registration's files.
    ccf_template_path : str, optional
        CCF average template.
    exaspim_template_path : str, optional
        exaSPIM template.
    template_to_ccf_asset : str, optional
        Mount name of the template-to-CCF data asset; it must be connected to this
        process. Chosen per sample by :mod:`exaspim_swc_transform.template_selection`.
    displacement_field : str, optional
        Local path of the manual CCF refinement field, or an empty string for none. Located by
        :mod:`exaspim_swc_transform.displacement`.

    Returns
    -------
    ResolvedInputs
        Absolute paths to every input.

    Raises
    ------
    MissingInput
        If a required input is absent. Raised by the helpers below; documented here
        because this is the function callers hold.
    """  # noqa: DOC502 - propagated, and callers need it documented
    root = _bundle_root(transform_dir)
    meta = root / "registration_metadata"

    return ResolvedInputs(
        dataset_id=dataset_id,
        # Optional: the acquisition is resolved from DocDB first, and a staged copy is
        # only the last resort. See exaspim_swc_processing.acquisition.
        acquisition_file=_resolve(
            "",
            [
                meta / f"acquisition_{dataset_id}.json",
                meta / "acquisition.json",
                root / "acquisition.json",
                root.parent / "acquisition.json",
            ],
            "acquisition file",
            required=False,
        ),
        brain_path=_resolve(
            "",
            [meta / f"{dataset_id}_10um_loaded_zarr_img.nii.gz"],
            "10 um loaded reference volume",
            required=False,
        ),
        resampled_brain_path=_resolve(
            "",
            [meta / f"{dataset_id}_10um_resampled_zarr_img.nii.gz"],
            "10 um resampled reference volume",
            required=False,
        ),
        brain_to_exaspim_transform_path=[
            _resolve(
                "",
                [root / f"{dataset_id}_to_exaSPIM_SyN_0GenericAffine.mat"],
                "sample->exaSPIM affine",
            ),
            _resolve(
                "",
                [root / f"{dataset_id}_to_exaSPIM_SyN_1InverseWarp.nii.gz"],
                "sample->exaSPIM inverse warp",
            ),
        ],
        exaspim_to_ccf_transform_path=[
            _resolve(
                f"/data/{template_to_ccf_asset}/0GenericAffine.mat",
                [],
                f"exaSPIM->CCF affine from {template_to_ccf_asset}",
            ),
            _resolve(
                f"/data/{template_to_ccf_asset}/1InverseWarp.nii.gz",
                [],
                f"exaSPIM->CCF inverse warp from {template_to_ccf_asset}",
            ),
        ],
        manual_transform_path=[displacement_field] if displacement_field else [],
        ccf_path=_resolve(ccf_template_path, [], "CCF template"),
        exaspim_template_path=_resolve(exaspim_template_path, [], "exaSPIM template"),
    )

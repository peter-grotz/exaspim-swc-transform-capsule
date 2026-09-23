"""Locate the registration inputs the transform needs, wherever they were mounted.

The same bundle arrives in several shapes -- staged from S3 into scratch, attached as a
Code Ocean data asset, or laid out under ``ccf_alignment/`` -- and filenames carry the
subject id, so each input is found by trying a short list of candidate paths.

Two inputs are deliberately optional. The reference volumes under
``registration_metadata/`` are published by only some datasets and nothing reads their
voxels; :mod:`exaspim_swc_transform.reference` reconstructs their geometry instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

TEMPLATE_TO_CCF_ASSET = "reg_exaspim_template_to_ccf_25um_v1.5"
"""Mounted template-to-CCF registration the pipeline standardises on."""

DEFAULT_EXASPIM_TO_CCF_AFFINE = f"/data/{TEMPLATE_TO_CCF_ASSET}/0GenericAffine.mat"
DEFAULT_EXASPIM_TO_CCF_INVERSE_WARP = f"/data/{TEMPLATE_TO_CCF_ASSET}/1InverseWarp.nii.gz"
DEFAULT_CCF_TEMPLATE = "/data/allen_mouse_ccf/average_template/average_template_10.nii.gz"
DEFAULT_EXASPIM_TEMPLATE = (
    "/data/exaspim_template_7subjects_nomask_10um_round6_template_only/fixed_median.nii.gz"
)

SUBJECT_ID = re.compile(r"\d{6}")
"""Subject ids are six digits; they appear in directory and file names."""


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


def _infer_dataset_id(bundle_root: Path, transform_dir: Path) -> str:
    """Recover the subject id from the directory and file names around the bundle.

    Parameters
    ----------
    bundle_root : Path
        Directory holding the registration outputs.
    transform_dir : Path
        Directory the bundle was staged or mounted at.

    Returns
    -------
    str
        The six-digit subject id.

    Raises
    ------
    ValueError
        If no id appears in any directory or filename.
    """
    names = [
        bundle_root.name,
        bundle_root.parent.name,
        transform_dir.name,
        transform_dir.parent.name,
    ]
    for root in (bundle_root / "registration_metadata", bundle_root, transform_dir):
        if root.is_dir():
            names.extend(path.name for path in sorted(root.glob("*")))
    for name in names:
        match = SUBJECT_ID.search(name)
        if match:
            return match.group(0)
    raise ValueError(
        f"Could not infer the subject id from {transform_dir}; pass dataset_id explicitly"
    )


DISPLACEMENT_FIELD_SUFFIXES = (".nrrd", ".nii.gz")
"""Extensions a manual CCF refinement displacement field may be published with."""


def _resolve_manual_df(manual_df_path: str, dataset_id: str) -> list[str]:
    """Locate the manual CCF refinement displacement field, if one was given.

    No filename convention is assumed. A directory -- typically a data asset mounted
    straight onto the pipeline -- is searched for a single field by extension, and an
    ambiguous directory is an error rather than a guess. Once fields are published under
    a settled name, matching it explicitly would be cheaper than this scan.

    Parameters
    ----------
    manual_df_path : str
        A file, a directory holding exactly one field, or empty for none.
    dataset_id : str
        Subject id. Unused while no naming convention is assumed; kept because the
        commented-out candidates below need it.

    Returns
    -------
    list[str]
        One path, or empty when no field was requested.

    Raises
    ------
    MissingInput
        If a path was given but no single displacement field can be identified there.
    """
    del dataset_id  # only the commented-out name candidates below would use it
    given = manual_df_path.strip().strip("'\"")
    if not given:
        return []
    path = Path(given)
    if path.is_file():
        return [str(path.resolve())]
    if not path.is_dir():
        raise MissingInput(f"--df-asset is neither a file nor a directory: {given}")

    # Guessing filenames was wrong: no displacement field has been published yet, so
    # every candidate below is speculation. Restore this once a convention exists.
    #
    # candidates = [
    #     path / f"{dataset_id}_displacement_field_vector_volume.nrrd",
    #     path / f"{dataset_id}_displacement_field.nrrd",
    #     path / "displacement_field_vector_volume.nrrd",
    #     path / "displacement_field.nrrd",
    # ]
    # return [_resolve("", candidates, "manual displacement field")]

    found = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.name.endswith(DISPLACEMENT_FIELD_SUFFIXES)
    )
    if len(found) == 1:
        return [str(found[0].resolve())]
    if not found:
        present = "\n".join(f"  - {c.name}" for c in sorted(path.iterdir())[:20]) or "  (empty)"
        raise MissingInput(
            f"No displacement field ({', '.join(DISPLACEMENT_FIELD_SUFFIXES)}) under "
            f"{path}. Contents:\n{present}"
        )
    listing = "\n".join(f"  - {c}" for c in found)
    raise MissingInput(
        f"{len(found)} displacement fields under {path}; point --df-asset at one.\n{listing}"
    )


def resolve_inputs(
    transform_dir: Path,
    manual_df_path: str = "",
    dataset_id: str = "",
    *,
    ccf_template_path: str = DEFAULT_CCF_TEMPLATE,
    exaspim_template_path: str = DEFAULT_EXASPIM_TEMPLATE,
    exaspim_to_ccf_affine_path: str = DEFAULT_EXASPIM_TO_CCF_AFFINE,
    exaspim_to_ccf_inverse_warp_path: str = DEFAULT_EXASPIM_TO_CCF_INVERSE_WARP,
) -> ResolvedInputs:
    """Locate every input the transform needs.

    Parameters
    ----------
    transform_dir : Path
        Directory the registration bundle was staged or mounted at.
    manual_df_path : str, optional
        Manual CCF refinement displacement field, or a directory holding one.
    dataset_id : str, optional
        Subject id. Inferred from the bundle when omitted.
    ccf_template_path : str, optional
        CCF average template.
    exaspim_template_path : str, optional
        exaSPIM template.
    exaspim_to_ccf_affine_path : str, optional
        Template-to-CCF affine.
    exaspim_to_ccf_inverse_warp_path : str, optional
        Template-to-CCF inverse warp.

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
    dataset_id = dataset_id or _infer_dataset_id(root, transform_dir)
    meta = root / "registration_metadata"

    return ResolvedInputs(
        dataset_id=dataset_id,
        # Older datasets file the acquisition under registration_metadata/; newer ones
        # publish it only at the dataset root.
        acquisition_file=_resolve(
            "",
            [
                meta / f"acquisition_{dataset_id}.json",
                meta / "acquisition.json",
                root / "acquisition.json",
                root.parent / "acquisition.json",
            ],
            "acquisition file",
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
            _resolve(exaspim_to_ccf_affine_path, [], "exaSPIM->CCF affine"),
            _resolve(exaspim_to_ccf_inverse_warp_path, [], "exaSPIM->CCF inverse warp"),
        ],
        manual_transform_path=_resolve_manual_df(manual_df_path, dataset_id),
        ccf_path=_resolve(ccf_template_path, [], "CCF template"),
        exaspim_template_path=_resolve(exaspim_template_path, [], "exaSPIM template"),
    )

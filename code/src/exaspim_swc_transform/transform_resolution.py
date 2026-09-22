"""Resolve registration assets and transform paths from bundle layout."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_EXASPIM_TO_CCF_AFFINE = "/data/reg_exaspim_template_to_ccf_25um_v1.5/0GenericAffine.mat"
DEFAULT_EXASPIM_TO_CCF_INVERSE_WARP = "/data/reg_exaspim_template_to_ccf_25um_v1.5/1InverseWarp.nii.gz"
DEFAULT_CCF_TEMPLATE = "/data/allen_mouse_ccf/average_template/average_template_10.nii.gz"
DEFAULT_EXASPIM_TEMPLATE = (
    "/data/exaspim_template_7subjects_nomask_10um_round6_template_only/fixed_median.nii.gz"
)


@dataclass(frozen=True)
class ResolvedInputs:
    dataset_id: str
    acquisition_file: str
    brain_path: str
    resampled_brain_path: str
    brain_to_exaspim_transform_path: list[str]
    exaspim_to_ccf_transform_path: list[str]
    manual_transform_path: list[str]
    ccf_path: str
    exaspim_template_path: str


def _clean_path_str(path_str: str) -> str:
    cleaned = path_str.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _require_file(path: str, what: str) -> str:
    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Missing {what}: {abs_path}")
    return abs_path


def _candidate_paths(root: Path, rels: list[str]) -> list[Path]:
    return [root / rel for rel in rels]


def _pick_existing(candidates: list[Path], what: str) -> str:
    hits = [c for c in candidates if c.is_file()]
    if len(hits) != 1:
        debug = "\n".join(f"  - {p}" for p in candidates)
        raise FileNotFoundError(f"Expected exactly one {what}. Found {len(hits)}.\nCandidates:\n{debug}")
    return str(hits[0].resolve())


def _resolve_bundle_root(transform_dir: Path) -> Path:
    bundle_root = transform_dir
    align_root = bundle_root / "ccf_alignment"
    if align_root.exists():
        return align_root
    return bundle_root


def _extract_id(text: str) -> str | None:
    match = re.search(r"\d{6}", text)
    if match:
        return match.group(0)
    return None


def _infer_dataset_id(bundle_root: Path, transform_dir: Path, dataset_id: str) -> str:
    if dataset_id:
        return dataset_id

    # 1) Try common directory names first.
    dir_candidates = [
        bundle_root.name,
        bundle_root.parent.name,
        transform_dir.name,
        transform_dir.parent.name,
    ]
    for candidate in dir_candidates:
        inferred = _extract_id(candidate)
        if inferred:
            return inferred

    # 2) Try registration metadata filenames.
    metadata_roots = [
        bundle_root / "registration_metadata",
        transform_dir / "registration_metadata",
    ]
    for meta_root in metadata_roots:
        if not meta_root.exists():
            continue

        for pattern in (
            "acquisition_*.json",
            "*_10um_loaded_zarr_img.nii.gz",
            "*_10um_resampled_zarr_img.nii.gz",
        ):
            for path in sorted(meta_root.glob(pattern)):
                inferred = _extract_id(path.name)
                if inferred:
                    return inferred

    # 3) Try top-level transform filenames.
    for root in (bundle_root, transform_dir):
        for pattern in (
            "*_to_exaSPIM_SyN_0GenericAffine.mat",
            "*_to_exaSPIM_SyN_1InverseWarp.nii.gz",
        ):
            for path in sorted(root.glob(pattern)):
                inferred = _extract_id(path.name)
                if inferred:
                    return inferred

    raise ValueError(
        "Could not infer dataset id from transform inputs. "
        "Please pass --dataset-id explicitly."
    )


def _resolve_manual_df(manual_df_path: str, dataset_id: str, manual_df_filename: str) -> list[str]:
    manual_df_path = _clean_path_str(manual_df_path)
    manual_df_filename = _clean_path_str(manual_df_filename)
    if not manual_df_path:
        return []

    p = Path(manual_df_path)
    if p.is_file():
        return [str(p.resolve())]

    if not p.is_dir():
        suggestions: list[Path] = []
        data_root = Path("/data")
        basename = p.name.lower()
        dataset_hint = dataset_id.lower() if dataset_id else ""
        if data_root.is_dir():
            for candidate in data_root.rglob("*"):
                if not candidate.is_file():
                    continue
                name = candidate.name.lower()
                if basename and name == basename:
                    suggestions.append(candidate)
                    continue
                if "displacement" in name and "field" in name:
                    if not dataset_hint or dataset_hint in name:
                        suggestions.append(candidate)

        suggestion_text = ""
        if suggestions:
            sample = "\n".join(f"  - {s}" for s in sorted({s.resolve() for s in suggestions})[:10])
            suggestion_text = (
                "\nPotential displacement-field files found under /data:\n"
                f"{sample}\n"
                "Set --manual-df-path to one of the exact paths above."
            )
        raise FileNotFoundError(
            f"--manual-df-path is neither file nor directory: {manual_df_path}{suggestion_text}"
        )

    if manual_df_filename:
        return [_pick_existing([p / manual_df_filename], "manual displacement field")]

    candidates = _candidate_paths(
        p,
        [
            f"{dataset_id}_displacement_field_vector_volume.nrrd",
            f"{dataset_id}_displacement_field.nrrd",
            f"{dataset_id}_displacement_field_vector_volume.nii.gz",
            f"{dataset_id}_displacement_field.nii.gz",
            f"{dataset_id} Displacement Field.nrrd",
            "displacement_field_vector_volume.nrrd",
            "displacement_field.nrrd",
            "manual_displacement_field.nrrd",
        ],
    )
    hits = [c for c in candidates if c.is_file()]
    if len(hits) == 1:
        return [str(hits[0].resolve())]
    if len(hits) > 1:
        debug = "\n".join(f"  - {p}" for p in hits)
        raise FileNotFoundError(
            "Multiple manual displacement field candidates found. "
            "Pass --manual-df-filename or --manual-df-path to exact file.\n"
            f"Matches:\n{debug}"
        )
    raise FileNotFoundError(
        "No manual displacement field found in directory. "
        "Pass --manual-df-filename or set --manual-df-path to the exact file."
    )


def resolve_inputs(
    transform_dir: Path,
    manual_df_path: str = "",
    dataset_id: str = "",
    *,
    acquisition_file_path: str = "",
    loaded_zarr_image_path: str = "",
    resampled_zarr_image_path: str = "",
    sample_to_exaspim_affine_path: str = "",
    sample_to_exaspim_inverse_warp_path: str = "",
    exaspim_to_ccf_affine_path: str = DEFAULT_EXASPIM_TO_CCF_AFFINE,
    exaspim_to_ccf_inverse_warp_path: str = DEFAULT_EXASPIM_TO_CCF_INVERSE_WARP,
    ccf_template_path: str = DEFAULT_CCF_TEMPLATE,
    exaspim_template_path: str = DEFAULT_EXASPIM_TEMPLATE,
    manual_df_filename: str = "",
) -> ResolvedInputs:
    bundle_root = _resolve_bundle_root(transform_dir)
    dataset_id = _infer_dataset_id(bundle_root, transform_dir, dataset_id)
    meta_root = bundle_root / "registration_metadata"

    if acquisition_file_path:
        acquisition_file = _require_file(acquisition_file_path, "acquisition file")
    else:
        acquisition_candidates = _candidate_paths(
            meta_root,
            [
                f"acquisition_{dataset_id}.json",
                "acquisition.json",
            ],
        )
        acquisition_file = _pick_existing(acquisition_candidates, "acquisition file")

    if loaded_zarr_image_path:
        brain_path = _require_file(loaded_zarr_image_path, "10um loaded zarr image")
    else:
        brain_path = _pick_existing(
            _candidate_paths(
                meta_root,
                [
                    f"{dataset_id}_10um_loaded_zarr_img.nii.gz",
                ],
            ),
            "10um loaded zarr image",
        )

    if resampled_zarr_image_path:
        resampled_brain_path = _require_file(resampled_zarr_image_path, "10um resampled zarr image")
    else:
        resampled_brain_path = _pick_existing(
            _candidate_paths(
                meta_root,
                [
                    f"{dataset_id}_10um_resampled_zarr_img.nii.gz",
                ],
            ),
            "10um resampled zarr image",
        )

    if sample_to_exaspim_affine_path:
        affine_to_exaspim = _require_file(sample_to_exaspim_affine_path, "sample->exaSPIM affine")
    else:
        affine_to_exaspim = _pick_existing(
            _candidate_paths(
                bundle_root,
                [
                    f"{dataset_id}_to_exaSPIM_SyN_0GenericAffine.mat",
                ],
            ),
            "sample->exaSPIM affine",
        )

    if sample_to_exaspim_inverse_warp_path:
        invwarp_to_exaspim = _require_file(
            sample_to_exaspim_inverse_warp_path, "sample->exaSPIM inverse warp"
        )
    else:
        invwarp_to_exaspim = _pick_existing(
            _candidate_paths(
                bundle_root,
                [
                    f"{dataset_id}_to_exaSPIM_SyN_1InverseWarp.nii.gz",
                ],
            ),
            "sample->exaSPIM inverse warp",
        )

    exaspim_to_ccf_affine = _require_file(exaspim_to_ccf_affine_path, "exaSPIM->CCF affine")
    exaspim_to_ccf_invwarp = _require_file(exaspim_to_ccf_inverse_warp_path, "exaSPIM->CCF inverse warp")
    ccf_path = _require_file(ccf_template_path, "CCF template")
    exaspim_template = _require_file(exaspim_template_path, "ExaSPIM template")
    manual_transform_path = _resolve_manual_df(manual_df_path, dataset_id, manual_df_filename)

    return ResolvedInputs(
        dataset_id=dataset_id,
        acquisition_file=acquisition_file,
        brain_path=brain_path,
        resampled_brain_path=resampled_brain_path,
        brain_to_exaspim_transform_path=[affine_to_exaspim, invwarp_to_exaspim],
        exaspim_to_ccf_transform_path=[exaspim_to_ccf_affine, exaspim_to_ccf_invwarp],
        manual_transform_path=manual_transform_path,
        ccf_path=ccf_path,
        exaspim_template_path=exaspim_template,
    )

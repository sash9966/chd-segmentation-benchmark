#!/usr/bin/env python3
"""
data_qa.py — NIfTI QA for ImageCHD (raw, pre-nnUNet)

Usage:
    python data_qa.py \
        --images-dir /path/to/imagesTr \
        --labels-dir /path/to/labelsTr \
        --output-csv qa_report.csv \
        [--fix-labels] \
        [--fix-output-dir /path/to/corrected_labels]

Checks per case:
  - NIfTI header: voxel spacing, orientation, shape
  - Label IDs outside {0..7} — optionally zeroed out (--fix-labels)
  - Aorta (label 6) and PA (label 7): largest connected component as % of total
    class volume; flagged if < 90%

Output: one row per case CSV with flag column.
"""

import argparse
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

VALID_LABEL_IDS = set(range(8))  # {0, 1, 2, 3, 4, 5, 6, 7}
CC_THRESHOLD = 0.90
ANISOTROPY_THRESHOLD = 3.0


def voxel_spacing(img: nib.Nifti1Image) -> tuple[float, float, float]:
    zooms = img.header.get_zooms()
    return float(zooms[0]), float(zooms[1]), float(zooms[2])


def orientation_code(img: nib.Nifti1Image) -> str:
    return "".join(nib.aff2axcodes(img.affine))


def largest_cc_fraction(mask: np.ndarray) -> float:
    """Return fraction of mask voxels that belong to the largest connected component."""
    labeled, n_components = ndimage.label(mask)
    if n_components == 0:
        return 0.0
    component_sizes = np.bincount(labeled.ravel())[1:]  # exclude background (0)
    return float(component_sizes.max()) / float(mask.sum())


def check_case(
    case_id: str,
    label_path: Path,
    fix_labels: bool,
    fix_output_dir: Path | None,
    image_path: Path | None,
) -> dict:
    row: dict = {"case_id": case_id}
    flags: list[str] = []

    # --- Image header (if image provided) ---
    if image_path is not None and image_path.exists():
        img = nib.load(str(image_path))
        sx, sy, sz = voxel_spacing(img)
        row["spacing_x"] = sx
        row["spacing_y"] = sy
        row["spacing_z"] = sz
        row["orientation"] = orientation_code(img)
        shape = img.shape
        row["shape_x"] = shape[0]
        row["shape_y"] = shape[1]
        row["shape_z"] = shape[2]

        spacings = [sx, sy, sz]
        ratio = max(spacings) / (min(spacings) + 1e-6)
        if ratio > ANISOTROPY_THRESHOLD:
            flags.append(f"anisotropic_spacing_ratio_{ratio:.1f}")

        canonical = {"RAS", "LAS", "RPS", "LPS"}
        if row["orientation"][:3] not in {o[:3] for o in canonical}:
            flags.append(f"unusual_orientation_{row['orientation']}")
    else:
        for k in ("spacing_x", "spacing_y", "spacing_z", "orientation", "shape_x", "shape_y", "shape_z"):
            row[k] = None

    # --- Label checks ---
    lbl_img = nib.load(str(label_path))
    lbl_data = np.asarray(lbl_img.dataobj, dtype=np.int16)

    unique_ids = sorted(np.unique(lbl_data).tolist())
    row["label_unique_ids"] = ";".join(map(str, unique_ids))

    oob_ids = sorted(set(unique_ids) - VALID_LABEL_IDS)
    row["oob_ids"] = ";".join(map(str, oob_ids)) if oob_ids else ""
    if oob_ids:
        flags.append(f"oob_label_ids_{oob_ids}")

    if fix_labels and oob_ids:
        for bad_id in oob_ids:
            lbl_data[lbl_data == bad_id] = 0
        if fix_output_dir is not None:
            fix_output_dir.mkdir(parents=True, exist_ok=True)
            out_path = fix_output_dir / label_path.name
        else:
            out_path = label_path  # overwrite in-place
        fixed_img = nib.Nifti1Image(lbl_data.astype(np.int16), lbl_img.affine, lbl_img.header)
        nib.save(fixed_img, str(out_path))

    # --- Connectivity checks for aorta (6) and PA (7) ---
    for cls_id, cls_name in [(6, "aorta"), (7, "pa")]:
        mask = lbl_data == cls_id
        present = bool(mask.any())
        row[f"{cls_name}_present"] = present
        if not present:
            row[f"{cls_name}_cc_pct"] = None
        else:
            frac = largest_cc_fraction(mask)
            row[f"{cls_name}_cc_pct"] = round(frac * 100, 2)
            if frac < CC_THRESHOLD:
                flags.append(f"{cls_name}_cc_low_{frac*100:.1f}pct")

    row["flags"] = "; ".join(flags) if flags else ""
    return row


def strip_modality_suffix(name: str) -> str:
    return re.sub(r"_\d{4}\.nii\.gz$", "", name)


def main() -> None:
    parser = argparse.ArgumentParser(description="NIfTI QA for ImageCHD raw data")
    parser.add_argument("--images-dir", type=Path, required=True, help="imagesTr directory")
    parser.add_argument("--labels-dir", type=Path, required=True, help="labelsTr directory")
    parser.add_argument("--output-csv", type=Path, default=Path("qa_report.csv"))
    parser.add_argument(
        "--fix-labels",
        action="store_true",
        help="Zero out label IDs outside {0..7} and save corrected files",
    )
    parser.add_argument(
        "--fix-output-dir",
        type=Path,
        default=None,
        help="Directory for corrected label files. If omitted with --fix-labels, files are overwritten in-place.",
    )
    args = parser.parse_args()

    label_files = sorted(args.labels_dir.glob("*.nii.gz"))
    if not label_files:
        print(f"ERROR: no .nii.gz files found in {args.labels_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for label_path in label_files:
        raw_stem = label_path.name.replace(".nii.gz", "")
        case_id = strip_modality_suffix(label_path.name) or raw_stem

        # Match image: try exact name with _0000 suffix first, then bare name
        candidates = [
            args.images_dir / label_path.name.replace(".nii.gz", "_0000.nii.gz"),
            args.images_dir / label_path.name,
        ]
        image_path = next((p for p in candidates if p.exists()), None)

        print(f"  QA: {case_id}", end="", flush=True)
        row = check_case(
            case_id=case_id,
            label_path=label_path,
            fix_labels=args.fix_labels,
            fix_output_dir=args.fix_output_dir,
            image_path=image_path,
        )
        flagged = bool(row["flags"])
        print(f"  {'[FLAGGED]' if flagged else '[OK]'}")
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)

    # --- Summary ---
    n = len(df)
    n_flagged = (df["flags"] != "").sum()
    n_oob = (df["oob_ids"] != "").sum()
    n_aorta = df["aorta_cc_pct"].notna() & (df["aorta_cc_pct"] < CC_THRESHOLD * 100)
    n_pa = df["pa_cc_pct"].notna() & (df["pa_cc_pct"] < CC_THRESHOLD * 100)

    print("\n=== QA Summary ===")
    print(f"  Total cases:            {n}")
    print(f"  Flagged (any issue):    {n_flagged}")
    print(f"  OOB label IDs:          {n_oob}")
    print(f"  Aorta CC < 90%:         {n_aorta.sum()}")
    print(f"  PA CC < 90%:            {n_pa.sum()}")
    print(f"\nReport written to: {args.output_csv}")


if __name__ == "__main__":
    main()

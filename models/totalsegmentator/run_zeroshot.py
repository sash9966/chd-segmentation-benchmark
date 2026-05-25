#!/usr/bin/env python3
"""
TotalSegmentator zero-shot inference on the CHD benchmark test set.

TotalSegmentator ships nnUNet weights. We run it out-of-the-box and then
remap its output labels to our CHD label space.

Usage:
    python models/totalsegmentator/run_zeroshot.py \
        --input-dir data/imagesTs \
        --output-dir results/totalsegmentator/zeroshot \
        [--fast]   # use fast mode (3mm) for a quick sanity check

Environment: environments/totalsegmentator.yml
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

# Map TotalSegmentator structure names → our CHD label IDs (0=bg, 1-7)
# TotalSegmentator uses structure names; we pick the ones relevant to CHD.
# Adjust based on which TS task you're running (e.g., 'heartchambers_highres').
TS_TO_CHD: dict[str, int] = {
    "heart_ventricle_left": 1,
    "heart_ventricle_right": 2,
    "heart_atrium_left": 3,
    "heart_atrium_right": 4,
    "heart_myocardium": 5,
    "aorta": 6,
    "pulmonary_artery": 7,
}


def merge_ts_output(ts_dir: Path, out_path: Path, shape: tuple, affine: np.ndarray, header) -> None:
    """Merge per-structure NIfTI files from TotalSegmentator into one label map."""
    merged = np.zeros(shape, dtype=np.uint8)
    for ts_name, chd_id in TS_TO_CHD.items():
        seg_file = ts_dir / f"{ts_name}.nii.gz"
        if seg_file.exists():
            seg = np.asarray(nib.load(str(seg_file)).dataobj, dtype=np.uint8)
            merged[seg > 0] = chd_id
    nib.save(nib.Nifti1Image(merged, affine, header), str(out_path))


def run_case(image_path: Path, out_path: Path, fast: bool) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        ts_out = Path(tmpdir) / "ts_segs"
        cmd = ["TotalSegmentator", "-i", str(image_path), "-o", str(ts_out), "--task", "heartchambers_highres"]
        if fast:
            cmd.append("--fast")
        subprocess.run(cmd, check=True)

        img = nib.load(str(image_path))
        merge_ts_output(ts_out, out_path, img.shape[:3], img.affine, img.header)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/imagesTs"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/totalsegmentator/zeroshot"))
    parser.add_argument("--fast", action="store_true", help="TotalSegmentator --fast flag (3mm)")
    parser.add_argument("--cases", nargs="*", help="Specific case IDs to run (default: all)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(args.input_dir.glob("*_0000.nii.gz"))
    if not images:
        images = sorted(args.input_dir.glob("*.nii.gz"))
    if args.cases:
        images = [p for p in images if any(c in p.name for c in args.cases)]

    print(f"Running TotalSegmentator zero-shot on {len(images)} cases...")
    for img_path in images:
        case_id = img_path.name.replace("_0000.nii.gz", "").replace(".nii.gz", "")
        out_path = args.output_dir / f"{case_id}.nii.gz"
        if out_path.exists():
            print(f"  [skip] {case_id}")
            continue
        print(f"  {case_id}...", end=" ", flush=True)
        try:
            run_case(img_path, out_path, args.fast)
            print("done")
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\nPredictions written to {args.output_dir}")


if __name__ == "__main__":
    main()

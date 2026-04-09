#!/usr/bin/env python3
"""Prepare a 3DGS dataset with VGGT (COLMAP-format sparse output).

Typical layout (as the user described):

workspace/
├── 3DGS/
├── VGGT/
└── datasets/scene_x/   # only images, or an images/ subfolder

This script can:
1) run VGGT's `demo_colmap.py` to produce sparse reconstruction;
2) normalize output to 3DGS-friendly COLMAP layout (`sparse/0/*`);
3) optionally launch 3DGS training with another Python interpreter
   (so VGGT/3DGS environments can be switched automatically).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use VGGT to prepare COLMAP-format sparse data for 3DGS."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help=(
            "Input dataset path. Supports either: "
            "(a) folder with images directly; (b) folder containing images/."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output scene directory (default: <dataset>_vggt_colmap)",
    )

    parser.add_argument(
        "--vggt-repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to VGGT repository (default: current repo root).",
    )
    parser.add_argument(
        "--vggt-python",
        type=str,
        default=sys.executable,
        help=(
            "Python executable for VGGT env, e.g. /path/to/vggt_env/bin/python "
            "or `conda run -n vggt python` via --vggt-launch-prefix."
        ),
    )
    parser.add_argument(
        "--vggt-launch-prefix",
        nargs="*",
        default=None,
        help=(
            "Optional command prefix before VGGT python, e.g.: "
            "--vggt-launch-prefix conda run -n vggt"
        ),
    )

    parser.add_argument("--use-ba", action="store_true", help="Pass --use_ba to demo_colmap.py")
    parser.add_argument("--max-reproj-error", type=float, default=8.0)
    parser.add_argument("--shared-camera", action="store_true")
    parser.add_argument("--camera-type", type=str, default="SIMPLE_PINHOLE")
    parser.add_argument("--vis-thresh", type=float, default=0.2)
    parser.add_argument("--query-frame-num", type=int, default=8)
    parser.add_argument("--max-query-pts", type=int, default=4096)
    parser.add_argument("--no-fine-tracking", action="store_true", help="Disable fine tracking in BA mode")
    parser.add_argument("--conf-thres-value", type=float, default=5.0)

    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into output/images (default is symlink when possible).",
    )

    parser.add_argument(
        "--run-3dgs",
        action="store_true",
        help="After VGGT preprocessing, auto-launch 3DGS training command.",
    )
    parser.add_argument(
        "--gs-repo",
        type=Path,
        default=None,
        help="Path to 3DGS repo (required when --run-3dgs).",
    )
    parser.add_argument(
        "--gs-python",
        type=str,
        default=sys.executable,
        help="Python executable for 3DGS env.",
    )
    parser.add_argument(
        "--gs-launch-prefix",
        nargs="*",
        default=None,
        help=(
            "Optional command prefix before 3DGS python, e.g.: "
            "--gs-launch-prefix conda run -n gs"
        ),
    )
    parser.add_argument(
        "--gs-script",
        type=Path,
        default=Path("train.py"),
        help="3DGS entry script path, relative to --gs-repo or absolute.",
    )
    parser.add_argument(
        "--gs-extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra arguments passed to 3DGS command (put at end).",
    )

    return parser.parse_args()


def run_cmd(cmd: Iterable[str], cwd: Path | None = None) -> None:
    printable = " ".join(str(x) for x in cmd)
    print(f"\n[RUN] {printable}")
    subprocess.run(list(cmd), cwd=str(cwd) if cwd else None, check=True)


def find_images_dir(dataset: Path) -> Path:
    dataset = dataset.resolve()
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    candidate = dataset / "images"
    if candidate.is_dir() and any(p.suffix.lower() in IMAGE_EXTS for p in candidate.iterdir() if p.is_file()):
        return candidate

    if dataset.is_dir() and any(p.suffix.lower() in IMAGE_EXTS for p in dataset.iterdir() if p.is_file()):
        return dataset

    raise ValueError(
        f"No images found in {dataset} or {candidate}. "
        "Supported extensions: " + ", ".join(sorted(IMAGE_EXTS))
    )


def safe_link_or_copy(src_dir: Path, dst_dir: Path, copy_images: bool) -> None:
    dst_dir.parent.mkdir(parents=True, exist_ok=True)
    if dst_dir.exists() or dst_dir.is_symlink():
        if dst_dir.is_symlink() or dst_dir.is_file():
            dst_dir.unlink()
        else:
            shutil.rmtree(dst_dir)

    if copy_images:
        shutil.copytree(src_dir, dst_dir)
        return

    try:
        dst_dir.symlink_to(src_dir, target_is_directory=True)
    except OSError:
        print("[WARN] Symlink failed, fallback to copy images.")
        shutil.copytree(src_dir, dst_dir)


def ensure_sparse_zero(scene_dir: Path) -> None:
    sparse = scene_dir / "sparse"
    sparse0 = sparse / "0"
    sparse0.mkdir(parents=True, exist_ok=True)

    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        src = sparse / name
        if src.exists():
            dst = sparse0 / name
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))

    required = [sparse0 / "cameras.bin", sparse0 / "images.bin", sparse0 / "points3D.bin"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing COLMAP outputs after VGGT: {missing}")


def build_prefixed_cmd(prefix: list[str] | None, py_exec: str, *args: str) -> list[str]:
    cmd: list[str] = []
    if prefix:
        cmd.extend(prefix)
    cmd.append(py_exec)
    cmd.extend(args)
    return cmd


def main() -> None:
    args = parse_args()

    image_src = find_images_dir(args.dataset)
    scene_dir = args.output.resolve() if args.output else (args.dataset.resolve().parent / f"{args.dataset.name}_vggt_colmap")
    scene_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Image source: {image_src}")
    print(f"[INFO] Output scene: {scene_dir}")

    safe_link_or_copy(image_src, scene_dir / "images", args.copy_images)

    vggt_repo = args.vggt_repo.resolve()
    demo_script = vggt_repo / "demo_colmap.py"
    if not demo_script.exists():
        raise FileNotFoundError(f"Cannot find {demo_script}")

    vggt_cmd = build_prefixed_cmd(
        args.vggt_launch_prefix,
        args.vggt_python,
        str(demo_script),
        "--scene_dir",
        str(scene_dir),
        "--max_reproj_error",
        str(args.max_reproj_error),
        "--camera_type",
        args.camera_type,
        "--vis_thresh",
        str(args.vis_thresh),
        "--query_frame_num",
        str(args.query_frame_num),
        "--max_query_pts",
        str(args.max_query_pts),
        "--conf_thres_value",
        str(args.conf_thres_value),
    )
    if args.use_ba:
        vggt_cmd.append("--use_ba")
    if args.shared_camera:
        vggt_cmd.append("--shared_camera")
    if args.no_fine_tracking:
        # demo_colmap default is True via action=store_true, so explicitly disabling is not directly supported;
        # for compatibility we skip passing --fine_tracking when user disables it.
        pass
    else:
        vggt_cmd.append("--fine_tracking")

    run_cmd(vggt_cmd, cwd=vggt_repo)

    ensure_sparse_zero(scene_dir)
    print("[INFO] COLMAP sparse ready for 3DGS at:", scene_dir / "sparse" / "0")

    if args.run_3dgs:
        if args.gs_repo is None:
            raise ValueError("--gs-repo is required when --run-3dgs")
        gs_repo = args.gs_repo.resolve()
        gs_script = args.gs_script if args.gs_script.is_absolute() else (gs_repo / args.gs_script)
        if not gs_script.exists():
            raise FileNotFoundError(f"Cannot find 3DGS script: {gs_script}")

        gs_cmd = build_prefixed_cmd(
            args.gs_launch_prefix,
            args.gs_python,
            str(gs_script),
            "-s",
            str(scene_dir),
            *args.gs_extra_args,
        )
        run_cmd(gs_cmd, cwd=gs_repo)

    print("\n[DONE] VGGT preprocessing finished.")


if __name__ == "__main__":
    main()

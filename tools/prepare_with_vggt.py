#!/usr/bin/env python3
"""Prepare a COLMAP-style dataset for 3D Gaussian Splatting using VGGT.

This script is designed for a workspace layout like:

workspace/
├── 3DGS/
└── VGGT/

Given an image-only dataset, it can:
1. normalize the scene layout to `scene/images/*`
2. run VGGT's `demo_colmap.py` to export COLMAP binaries
3. reshape the sparse output for 3DGS (`sparse/0/*.bin`)
4. optionally launch a 3DGS command using a different Python/Conda env

Environment switching support:
- preferred: pass `--vggt-python` and `--threedgs-python`
- optional: pass `--vggt-conda-env` and/or `--threedgs-conda-env`
"""

from __future__ import annotations

import argparse
import struct
import os
import shutil
import subprocess
import sys
import numpy as np
from pathlib import Path
from typing import List, Optional

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use VGGT to prepare COLMAP files for 3DGS.")
    parser.add_argument("--dataset", type=Path, required=True, help="Image-only dataset dir, or dir containing images/.")
    parser.add_argument(
        "--output-scene",
        type=Path,
        default=None,
        help="Output scene dir. Default: in-place when dataset has images/, otherwise <dataset>_vggt_scene.",
    )
    parser.add_argument("--vggt-repo", type=Path, default=Path(__file__).resolve().parents[1], help="Path to VGGT repo.")
    parser.add_argument("--threedgs-repo", type=Path, default=None, help="Path to 3DGS repo (optional).")
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlink/hardlink fallback.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output-scene directory if needed.")

    # VGGT demo_colmap options
    parser.add_argument("--use-ba", action="store_true", help="Enable bundle adjustment in VGGT demo_colmap.")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Local VGGT checkpoint path (.pt). Forwarded to demo_colmap.py.",
    )
    parser.add_argument(
        "--tracker-checkpoint-path",
        type=Path,
        default=None,
        help="Local VGGSfM tracker checkpoint path (.pt). Used for --use-ba, forwarded to demo_colmap.py.",
    )
    parser.add_argument("--max-query-pts", type=int, default=None, help="Forwarded to demo_colmap.py.")
    parser.add_argument("--query-frame-num", type=int, default=None, help="Forwarded to demo_colmap.py.")
    parser.add_argument("--vis-thresh", type=float, default=None, help="Forwarded to demo_colmap.py BA mode.")
    parser.add_argument(
        "--keypoint-extractor",
        type=str,
        default=None,
        help="Forwarded to demo_colmap.py BA mode (e.g. 'sift' avoids extra model downloads).",
    )
    parser.add_argument(
        "--conf-thres-value",
        type=float,
        default=None,
        help="Forwarded to demo_colmap.py (non-BA mode). Lower it if points are too sparse.",
    )
    parser.add_argument(
        "--img-load-resolution",
        type=int,
        default=None,
        help="Forwarded to demo_colmap.py. Lower this (e.g. 768) to reduce GPU memory usage.",
    )
    parser.add_argument(
        "--vggt-resolution",
        type=int,
        default=None,
        help="Forwarded to demo_colmap.py. Lower this (e.g. 448) to reduce GPU memory usage.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Forwarded to demo_colmap.py.")

    # env switching options
    parser.add_argument("--vggt-python", type=Path, default=None, help="Python executable for VGGT env.")
    parser.add_argument("--threedgs-python", type=Path, default=None, help="Python executable for 3DGS env.")
    parser.add_argument("--vggt-conda-env", type=str, default=None, help="Conda env name for VGGT (if python path is not provided).")
    parser.add_argument("--threedgs-conda-env", type=str, default=None, help="Conda env name for 3DGS (if python path is not provided).")

    # optional 3DGS launch
    parser.add_argument(
        "--run-3dgs",
        choices=["none", "train", "custom"],
        default="none",
        help="Optionally launch 3DGS after VGGT preprocessing.",
    )
    parser.add_argument(
        "--custom-3dgs-cmd",
        type=str,
        default=None,
        help="Custom command for --run-3dgs custom. Use {scene} placeholder for scene path.",
    )
    parser.add_argument(
        "--train-args",
        type=str,
        default="",
        help="Extra args for 3DGS train.py when --run-3dgs train is used.",
    )

    return parser.parse_args()


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in _IMAGE_EXTS


def _list_images(folder: Path) -> List[Path]:
    images = [p for p in folder.iterdir() if _is_image(p)]
    return sorted(images)


def resolve_scene_layout(dataset: Path, output_scene: Optional[Path], force: bool) -> Path:
    dataset = dataset.resolve()
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset does not exist: {dataset}")

    images_dir = dataset / "images"
    if images_dir.is_dir() and _list_images(images_dir):
        scene_dir = output_scene.resolve() if output_scene else dataset
        if scene_dir != dataset:
            if scene_dir.exists() and force:
                shutil.rmtree(scene_dir)
            scene_dir.mkdir(parents=True, exist_ok=True)
            dst_images = scene_dir / "images"
            if dst_images.exists() and force:
                shutil.rmtree(dst_images)
            if not dst_images.exists():
                os.symlink(images_dir, dst_images, target_is_directory=True)
        return scene_dir

    root_images = _list_images(dataset)
    if not root_images:
        raise RuntimeError(
            f"No images found in {dataset}. Put images in dataset root or dataset/images/."
        )

    scene_dir = output_scene.resolve() if output_scene else dataset.with_name(dataset.name + "_vggt_scene")
    if scene_dir.exists() and force:
        shutil.rmtree(scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "images").mkdir(parents=True, exist_ok=True)
    return scene_dir


def populate_images(dataset: Path, scene_dir: Path, copy_images: bool) -> None:
    target_images = scene_dir / "images"
    if any(target_images.iterdir()):
        return

    source_images = _list_images(dataset / "images") if (dataset / "images").is_dir() else _list_images(dataset)
    if not source_images:
        raise RuntimeError(f"No images to populate from {dataset}")

    for src in source_images:
        dst = target_images / src.name
        if copy_images:
            shutil.copy2(src, dst)
            continue

        try:
            os.link(src, dst)
        except OSError:
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)


def build_runner_cmd(script: Path, python_path: Optional[Path], conda_env: Optional[str], args: List[str]) -> List[str]:
    if python_path:
        return [str(python_path), str(script), *args]
    if conda_env:
        return ["conda", "run", "-n", conda_env, "python", str(script), *args]
    return [sys.executable, str(script), *args]


def run_command(cmd: List[str], cwd: Optional[Path] = None) -> None:
    shown = " ".join(cmd)
    print(f"\n[RUN] {shown}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def run_vggt_colmap(args: argparse.Namespace, scene_dir: Path) -> None:
    demo_script = args.vggt_repo.resolve() / "demo_colmap.py"
    if not demo_script.exists():
        raise FileNotFoundError(f"Cannot find demo_colmap.py at: {demo_script}")

    demo_args = ["--scene_dir", str(scene_dir), "--seed", str(args.seed)]
    if args.checkpoint_path is not None:
        demo_args += ["--checkpoint_path", str(args.checkpoint_path.resolve())]
    if args.tracker_checkpoint_path is not None:
        demo_args += ["--tracker_checkpoint_path", str(args.tracker_checkpoint_path.resolve())]
    if args.use_ba:
        demo_args.append("--use_ba")
    if args.max_query_pts is not None:
        demo_args += ["--max_query_pts", str(args.max_query_pts)]
    if args.query_frame_num is not None:
        demo_args += ["--query_frame_num", str(args.query_frame_num)]
    if args.vis_thresh is not None:
        demo_args += ["--vis_thresh", str(args.vis_thresh)]
    if args.keypoint_extractor is not None:
        demo_args += ["--keypoint_extractor", args.keypoint_extractor]
    if args.conf_thres_value is not None:
        demo_args += ["--conf_thres_value", str(args.conf_thres_value)]
    if args.img_load_resolution is not None:
        demo_args += ["--img_load_resolution", str(args.img_load_resolution)]
    if args.vggt_resolution is not None:
        demo_args += ["--vggt_resolution", str(args.vggt_resolution)]

    cmd = build_runner_cmd(
        script=demo_script,
        python_path=args.vggt_python.resolve() if args.vggt_python else None,
        conda_env=args.vggt_conda_env,
        args=demo_args,
    )
    run_command(cmd, cwd=args.vggt_repo.resolve())


def ensure_3dgs_sparse_layout(scene_dir: Path) -> None:
    sparse = scene_dir / "sparse"
    if not sparse.exists():
        raise RuntimeError(f"Expected VGGT sparse output not found: {sparse}")

    model0 = sparse / "0"
    model0.mkdir(exist_ok=True)

    for name in ["cameras.bin", "images.bin", "points3D.bin"]:
        src = sparse / name
        if not src.exists():
            raise RuntimeError(f"Missing required COLMAP file: {src}")
        dst = model0 / name
        if dst.exists():
            dst.unlink()
        shutil.copy2(src, dst)

    # If VGGT exported a PLY preview, convert it into a 3DGS-compatible PLY
    # with explicit normal fields (nx, ny, nz).
    src_ply = sparse / "points.ply"
    dst_ply = model0 / "points3D.ply"
    if src_ply.exists():
        _write_3dgs_compatible_ply_from_vggt(src_ply, dst_ply)


def _write_3dgs_compatible_ply_from_vggt(src_ply: Path, dst_ply: Path) -> None:
    import trimesh

    cloud = trimesh.load(str(src_ply), process=False)
    points = np.asarray(cloud.vertices)
    colors = np.asarray(cloud.colors) if hasattr(cloud, "colors") else None
    if colors is None or colors.size == 0:
        colors = np.full((points.shape[0], 3), 255, dtype=np.uint8)
    else:
        colors = colors[:, :3].astype(np.uint8)

    normals = np.zeros_like(points, dtype=np.float32)
    dst_ply.parent.mkdir(parents=True, exist_ok=True)
    with dst_ply.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, n, c in zip(points, normals, colors):
            f.write(
                f"{float(p[0])} {float(p[1])} {float(p[2])} "
                f"{float(n[0])} {float(n[1])} {float(n[2])} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )


def run_3dgs(args: argparse.Namespace, scene_dir: Path) -> None:
    if args.run_3dgs == "none":
        return
    if not args.threedgs_repo:
        raise ValueError("--threedgs-repo is required when --run-3dgs is not 'none'.")

    repo = args.threedgs_repo.resolve()
    if not repo.exists():
        raise FileNotFoundError(f"3DGS repo does not exist: {repo}")

    if args.run_3dgs == "train":
        train_script = repo / "train.py"
        if not train_script.exists():
            raise FileNotFoundError(f"Cannot find 3DGS train.py at: {train_script}")
        extra = args.train_args.split() if args.train_args else []
        run_args = [str(train_script), "-s", str(scene_dir), *extra]
    else:
        if not args.custom_3dgs_cmd:
            raise ValueError("--custom-3dgs-cmd is required for --run-3dgs custom.")
        formatted = args.custom_3dgs_cmd.format(scene=str(scene_dir))
        run_args = ["-c", formatted]

    if args.run_3dgs == "custom":
        if args.threedgs_python:
            cmd = [str(args.threedgs_python.resolve()), *run_args]
        elif args.threedgs_conda_env:
            cmd = ["conda", "run", "-n", args.threedgs_conda_env, "bash", *run_args]
        else:
            cmd = ["bash", *run_args]
    else:
        if args.threedgs_python:
            cmd = [str(args.threedgs_python.resolve()), *run_args]
        elif args.threedgs_conda_env:
            cmd = ["conda", "run", "-n", args.threedgs_conda_env, "python", *run_args]
        else:
            cmd = [sys.executable, *run_args]

    run_command(cmd, cwd=repo)


def count_colmap_points(scene_dir: Path) -> int:
    candidates = [scene_dir / "sparse" / "0" / "points3D.bin", scene_dir / "sparse" / "points3D.bin"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return -1
    with path.open("rb") as f:
        header = f.read(8)
        if len(header) < 8:
            return -1
        return struct.unpack("<Q", header)[0]


def main() -> None:
    args = parse_args()

    dataset = args.dataset.resolve()
    scene_dir = resolve_scene_layout(dataset, args.output_scene, args.force)
    populate_images(dataset, scene_dir, copy_images=args.copy_images)

    print(f"Dataset: {dataset}")
    print(f"Prepared scene: {scene_dir}")

    run_vggt_colmap(args, scene_dir)
    ensure_3dgs_sparse_layout(scene_dir)

    num_points = count_colmap_points(scene_dir)
    if num_points >= 0:
        print(f"Sparse initialization points in COLMAP: {num_points}")
    else:
        print("Sparse initialization points in COLMAP: unknown (unable to read points3D.bin header)")
    if args.run_3dgs != "none" and num_points == 0:
        raise RuntimeError(
            "VGGT produced 0 sparse points in points3D.bin, so 3DGS cannot initialize. "
            "Try BA with stronger tracking settings, e.g. --max-query-pts 4096 --vis-thresh 0.05 "
            "and avoid weak extractors."
        )

    run_3dgs(args, scene_dir)

    print("\nDone. 3DGS-ready scene:")
    print(f"  images: {scene_dir / 'images'}")
    print(f"  sparse: {scene_dir / 'sparse'} (and sparse/0)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render retargeted robot motion to an MP4 without opening a viewer.

The script reuses the scene and motion-loading contract from
``multi_robot_visualize.py``. Run it with ``MUJOCO_GL=egl`` on a headless
machine so MuJoCo can render through EGL.

Example:
    MUJOCO_GL=egl python scripts/render_robot_motion.py \
        --motion dance1_subject2_from_g1 \
        --robots kengo \
        --source-fps 30 \
        --render-fps 30 \
        --output output_data/videos/dance1_subject2_from_g1_kengo.mp4
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from tqdm import tqdm

from multi_robot_visualize import (
    MOTION_DIR,
    available_robots,
    build_combined_spec,
    get_qpos_start,
    load_motion,
)


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def positive_even_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0 or parsed % 2:
        raise argparse.ArgumentTypeError("value must be a positive even integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one or more retargeted robot motions to H.264 MP4."
    )
    parser.add_argument(
        "--motion",
        required=True,
        help="Motion stem; files must be named <motion>_<robot>.csv.",
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        required=True,
        choices=available_robots(),
        help="Robot names to render.",
    )
    parser.add_argument(
        "--motion-dir",
        type=Path,
        default=Path(MOTION_DIR),
        help="Directory containing retargeted motion CSV files.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output MP4 path.")
    parser.add_argument("--source-fps", type=positive_float, default=30.0)
    parser.add_argument("--render-fps", type=positive_float, default=30.0)
    parser.add_argument("--width", type=positive_even_int, default=960)
    parser.add_argument("--height", type=positive_even_int, default=540)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--end-frame",
        type=int,
        default=-1,
        help="Inclusive source-frame index; -1 renders through the end.",
    )
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-15.0)
    parser.add_argument("--camera-distance", type=positive_float, default=2.5)
    parser.add_argument(
        "--camera-track-alpha",
        type=float,
        default=0.18,
        help="Root tracking smoothing in (0, 1]; 1 follows immediately.",
    )
    parser.add_argument("--crf", type=int, default=24, help="libx264 quality, 0-51.")
    parser.add_argument("--preset", default="medium", help="libx264 preset.")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable.")
    args = parser.parse_args()

    if not 0.0 < args.camera_track_alpha <= 1.0:
        parser.error("--camera-track-alpha must be in (0, 1]")
    if not 0 <= args.crf <= 51:
        parser.error("--crf must be between 0 and 51")
    return args


def main() -> None:
    args = parse_args()
    ffmpeg_path = shutil.which(args.ffmpeg)
    if ffmpeg_path is None:
        raise FileNotFoundError(f"ffmpeg executable not found: {args.ffmpeg}")

    robot_qpos: dict[str, np.ndarray] = {}
    for robot in args.robots:
        csv_path = args.motion_dir / f"{args.motion}_{robot}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Motion file not found: {csv_path}")
        robot_qpos[robot] = load_motion(str(csv_path))

    n_frames = min(qpos.shape[0] for qpos in robot_qpos.values())
    if n_frames <= 0:
        raise ValueError("Motion contains no frames")
    start_frame = max(0, args.start_frame)
    end_frame = n_frames - 1 if args.end_frame < 0 else min(args.end_frame, n_frames - 1)
    if start_frame > end_frame:
        raise ValueError(
            f"Invalid frame range [{args.start_frame}, {args.end_frame}] for {n_frames} frames"
        )

    frame_step = max(1, round(args.source_fps / args.render_fps))
    encoded_fps = args.source_fps / frame_step
    frame_ids = range(start_frame, end_frame + 1, frame_step)
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spec = build_combined_spec(args.robots)
    spec.visual.global_.offwidth = args.width
    spec.visual.global_.offheight = args.height
    model = spec.compile()
    data = mujoco.MjData(model)

    robot_start: dict[str, int] = {}
    robot_dim: dict[str, int] = {}
    for robot in args.robots:
        start = get_qpos_start(model, f"{robot}_floating_base_joint")
        dim = robot_qpos[robot].shape[1]
        if start + dim > model.nq:
            raise ValueError(
                f"qpos dimension mismatch for {robot}: [{start}:{start + dim}] exceeds nq={model.nq}"
            )
        robot_start[robot] = start
        robot_dim[robot] = dim

    cols = max(1, math.ceil(math.sqrt(len(args.robots))))
    offsets: dict[str, tuple[float, float]] = {}
    for index, robot in enumerate(args.robots):
        col = index % cols
        row = index // cols
        rows = math.ceil(len(args.robots) / cols)
        offsets[robot] = (
            (col - (cols - 1) / 2.0) * 2.0,
            (row - (rows - 1) / 2.0) * 2.0,
        )

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = args.camera_azimuth
    camera.elevation = args.camera_elevation
    camera.distance = args.camera_distance

    ffmpeg_cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{args.width}x{args.height}",
        "-r",
        f"{encoded_fps:.12g}",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        args.preset,
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    renderer: mujoco.Renderer | None = None
    tracked_lookat: np.ndarray | None = None
    try:
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        if process.stdin is None:
            raise RuntimeError("ffmpeg stdin was not created")

        for frame_idx in tqdm(frame_ids, desc="Rendering MP4", unit="frame"):
            root_positions = []
            for robot in args.robots:
                start = robot_start[robot]
                dim = robot_dim[robot]
                data.qpos[start : start + dim] = robot_qpos[robot][frame_idx]
                dx, dy = offsets[robot]
                data.qpos[start] += dx
                data.qpos[start + 1] += dy
                root_positions.append(data.qpos[start : start + 3].copy())

            mujoco.mj_forward(model, data)
            target_lookat = np.mean(root_positions, axis=0)
            target_lookat[2] -= 0.15
            if tracked_lookat is None:
                tracked_lookat = target_lookat
            else:
                tracked_lookat += args.camera_track_alpha * (target_lookat - tracked_lookat)
            camera.lookat[:] = tracked_lookat

            renderer.update_scene(data, camera=camera)
            process.stdin.write(renderer.render().tobytes())

        process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")
    except BaseException:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        process.wait()
        raise
    finally:
        if renderer is not None:
            renderer.close()

    print(
        f"Saved {len(frame_ids)} frames at {encoded_fps:g} FPS "
        f"({args.width}x{args.height}) to: {output_path}"
    )


if __name__ == "__main__":
    main()

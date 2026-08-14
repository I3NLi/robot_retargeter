#!/usr/bin/env python3
"""Batch-render Kengo retargeted NPZ motions to H.264 MP4.

The output directory mirrors the source hierarchy. Existing valid MP4 files
are skipped by default, which makes interrupted batches resumable.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from audit_retargeted_npz import DEFAULT_ROBOT_XML, load_motion, model_joint_contract


def positive_even_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0 or parsed % 2:
        raise argparse.ArgumentTypeError("value must be a positive even integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-render Kengo NPZ motions to MP4.")
    parser.add_argument("input_root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_ROBOT_XML)
    parser.add_argument("--width", type=positive_even_int, default=640)
    parser.add_argument("--height", type=positive_even_int, default=360)
    parser.add_argument("--max-render-fps", type=float, default=30.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-15.0)
    parser.add_argument("--camera-distance", type=float, default=2.4)
    parser.add_argument("--camera-track-alpha", type=float, default=0.18)
    parser.add_argument("--crf", type=int, default=26)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--ffmpeg", default=None)
    parser.add_argument("--ffprobe", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--match",
        default="",
        help="Only render relative paths containing this case-insensitive text.",
    )
    parser.add_argument("--progress-frames", type=int, default=1000)
    args = parser.parse_args()
    if args.max_render_fps <= 0.0:
        parser.error("--max-render-fps must be positive")
    if args.camera_distance <= 0.0:
        parser.error("--camera-distance must be positive")
    if not 0.0 < args.camera_track_alpha <= 1.0:
        parser.error("--camera-track-alpha must be in (0, 1]")
    if not 0 <= args.crf <= 51:
        parser.error("--crf must be between 0 and 51")
    if args.shard_count <= 0:
        parser.error("--shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, shard-count)")
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    return args


def bundled_binary(name: str) -> str | None:
    executable_dir = Path(sys.executable).resolve().parent
    suffix = ".exe" if os.name == "nt" else ""
    candidates = (
        executable_dir / "Library" / "bin" / f"{name}{suffix}",
        executable_dir / "bin" / f"{name}{suffix}",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return shutil.which(name)


def build_scene_model(robot_xml: Path, width: int, height: int) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(robot_xml))
    spec.modelname = "kengo_npz_recording"
    spec.visual.global_.offwidth = width
    spec.visual.global_.offheight = height
    spec.visual.headlight.ambient = [0.5, 0.5, 0.5]
    spec.visual.headlight.diffuse = [0.65, 0.65, 0.65]
    spec.visual.headlight.specular = [0.25, 0.25, 0.25]

    spec.add_texture(
        name="recording_skybox",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.34, 0.48, 0.62],
        rgb2=[0.03, 0.05, 0.08],
        width=512,
        height=512,
    )
    grid_res = 256
    line_px = 3
    grid_rgba = np.zeros((grid_res, grid_res, 4), dtype=np.uint8)
    grid_rgba[:, :, :3] = (118, 122, 126)
    grid_rgba[:, :, 3] = 255
    grid_rgba[:line_px, :, :3] = (58, 62, 66)
    grid_rgba[-line_px:, :, :3] = (58, 62, 66)
    grid_rgba[:, :line_px, :3] = (58, 62, 66)
    grid_rgba[:, -line_px:, :3] = (58, 62, 66)
    grid_texture = spec.add_texture(
        name="recording_ground_texture",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        width=grid_res,
        height=grid_res,
        nchannel=4,
    )
    grid_texture.data = grid_rgba.reshape(-1).tobytes()
    spec.add_material(
        name="recording_ground_material",
        textures=["", "recording_ground_texture"],
        texrepeat=[12, 12],
        texuniform=True,
        reflectance=0.0,
    )
    spec.worldbody.add_light(
        pos=[0.0, 0.0, 8.0],
        dir=[0.0, 0.0, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.72, 0.72, 0.72],
        specular=[0.2, 0.2, 0.2],
    )
    ground = spec.worldbody.add_geom(
        name="recording_ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.0, 0.0, 0.05],
        material="recording_ground_material",
        pos=[0.0, 0.0, 0.0],
    )
    ground.contype = 0
    ground.conaffinity = 0
    for geom in spec.geoms:
        geom.contype = 0
        geom.conaffinity = 0
    return spec.compile()


def is_valid_mp4(path: Path, ffprobe: str | None) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    if ffprobe is None:
        return True
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,nb_frames",
        "-of",
        "csv=p=0",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    return result.returncode == 0 and bool(result.stdout.strip())


def select_frame_ids(
    frame_count: int, source_fps: float, max_render_fps: float
) -> tuple[range | np.ndarray, float]:
    if source_fps <= max_render_fps:
        return range(frame_count), source_fps

    # Sample on an evenly spaced output timeline instead of taking an integer
    # stride. This supports standard review rates such as 24 fps without
    # accidentally turning a 30 fps source into 15 fps. Slightly adjust the
    # encoded rate so output_frames / encoded_fps exactly preserves the source
    # duration (source_frames / source_fps).
    output_frame_count = max(1, int(round(frame_count * max_render_fps / source_fps)))
    frame_ids = np.floor(
        np.arange(output_frame_count, dtype=np.float64) * source_fps / max_render_fps
    ).astype(np.int64)
    np.clip(frame_ids, 0, frame_count - 1, out=frame_ids)
    encoded_fps = output_frame_count * source_fps / frame_count
    return frame_ids, encoded_fps


def ffmpeg_command(
    ffmpeg: str,
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    preset: str,
    crf: int,
    title: str,
) -> list[str]:
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        f"{fps:.12g}",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-metadata",
        f"title={title}",
        str(output_path),
    ]


def render_one(
    source_path: Path,
    relative_path: Path,
    output_path: Path,
    model: mujoco.MjModel,
    renderer: mujoco.Renderer,
    model_joint_names: list[str],
    qpos_addresses: np.ndarray,
    ffmpeg: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    fps, joint_pos, base_pos, base_quat, _ = load_motion(source_path, model_joint_names)
    all_values_finite = (
        np.isfinite(joint_pos).all()
        and np.isfinite(base_pos).all()
        and np.isfinite(base_quat).all()
    )
    if not all_values_finite:
        raise ValueError("motion contains non-finite values")
    quat_norm = np.linalg.norm(base_quat, axis=1)
    if np.any(quat_norm <= 1e-12):
        raise ValueError("motion contains a zero-length base quaternion")
    base_quat = base_quat / quat_norm[:, None]
    frame_ids, encoded_fps = select_frame_ids(
        joint_pos.shape[0], fps, args.max_render_fps
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(f"{output_path.stem}.partial.mp4")
    if partial_path.exists():
        partial_path.unlink()
    command = ffmpeg_command(
        ffmpeg=ffmpeg,
        output_path=partial_path,
        width=args.width,
        height=args.height,
        fps=encoded_fps,
        preset=args.preset,
        crf=args.crf,
        title=relative_path.as_posix(),
    )
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = args.camera_azimuth
    camera.elevation = args.camera_elevation
    camera.distance = args.camera_distance
    tracked_lookat: np.ndarray | None = None
    rendered_frames = 0
    try:
        if process.stdin is None:
            raise RuntimeError("ffmpeg stdin was not created")
        for frame_idx in frame_ids:
            data.qpos[:3] = base_pos[frame_idx]
            data.qpos[3:7] = base_quat[frame_idx]
            data.qpos[qpos_addresses] = joint_pos[frame_idx]
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)

            target = base_pos[frame_idx].copy()
            target[2] -= 0.05
            if tracked_lookat is None:
                tracked_lookat = target
            else:
                tracked_lookat += args.camera_track_alpha * (target - tracked_lookat)
            camera.lookat[:] = tracked_lookat
            renderer.update_scene(data, camera=camera)
            process.stdin.write(renderer.render().tobytes())
            rendered_frames += 1
            if args.progress_frames and rendered_frames % args.progress_frames == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"  {relative_path.as_posix()}: {rendered_frames}/{len(frame_ids)} "
                    f"frames ({rendered_frames / max(elapsed, 1e-9):.1f} fps)",
                    flush=True,
                )
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with {return_code}: {stderr.strip()}")
        partial_path.replace(output_path)
    except BaseException:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        process.wait()
        if partial_path.exists():
            partial_path.unlink()
        raise

    elapsed_s = time.perf_counter() - started
    return {
        "relative_npz": relative_path.as_posix(),
        "relative_mp4": output_path.relative_to(args.output_root.resolve()).as_posix(),
        "status": "rendered",
        "source_frames": int(joint_pos.shape[0]),
        "rendered_frames": rendered_frames,
        "source_fps": float(fps),
        "encoded_fps": float(encoded_fps),
        "duration_s": float(joint_pos.shape[0] / fps),
        "bytes": int(output_path.stat().st_size),
        "elapsed_s": elapsed_s,
        "render_throughput_fps": rendered_frames / max(elapsed_s, 1e-9),
        "error": "",
    }


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "relative_npz",
        "relative_mp4",
        "status",
        "source_frames",
        "rendered_frames",
        "source_fps",
        "encoded_fps",
        "duration_s",
        "bytes",
        "elapsed_s",
        "render_throughput_fps",
        "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.robot_xml = args.robot_xml.expanduser().resolve()
    if not args.input_root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input_root}")
    if not args.robot_xml.is_file():
        raise FileNotFoundError(f"Robot MJCF not found: {args.robot_xml}")

    ffmpeg = args.ffmpeg or bundled_binary("ffmpeg")
    ffprobe = args.ffprobe or bundled_binary("ffprobe")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg executable not found")
    files = sorted(args.input_root.rglob("*.npz"))
    if args.match:
        needle = args.match.casefold()
        files = [
            path
            for path in files
            if needle in path.relative_to(args.input_root).as_posix().casefold()
        ]
    files = [
        path
        for global_index, path in enumerate(files)
        if global_index % args.shard_count == args.shard_index
    ]
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError("No NPZ files matched the selected shard/filter")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        args.output_root
        / "_render_manifests"
        / f"render_manifest_shard_{args.shard_index:03d}_of_{args.shard_count:03d}.csv"
    )
    model = build_scene_model(args.robot_xml, args.width, args.height)
    model_joint_names, qpos_addresses, _lowers, _uppers = model_joint_contract(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    rows: list[dict[str, Any]] = []
    batch_started = time.perf_counter()
    try:
        for local_index, source_path in enumerate(files, start=1):
            relative = source_path.relative_to(args.input_root)
            output_path = (args.output_root / relative).with_suffix(".mp4")
            if not args.overwrite and is_valid_mp4(output_path, ffprobe):
                row = {
                    "relative_npz": relative.as_posix(),
                    "relative_mp4": output_path.relative_to(args.output_root).as_posix(),
                    "status": "skipped_existing",
                    "bytes": int(output_path.stat().st_size),
                    "error": "",
                }
                rows.append(row)
                print(
                    f"[{local_index}/{len(files)}] skip existing: {relative.as_posix()}",
                    flush=True,
                )
                write_manifest(manifest_path, rows)
                continue

            print(
                f"[{local_index}/{len(files)}] render: {relative.as_posix()}",
                flush=True,
            )
            try:
                row = render_one(
                    source_path=source_path,
                    relative_path=relative,
                    output_path=output_path,
                    model=model,
                    renderer=renderer,
                    model_joint_names=model_joint_names,
                    qpos_addresses=qpos_addresses,
                    ffmpeg=ffmpeg,
                    args=args,
                )
                print(
                    f"  saved {row['rendered_frames']} frames, "
                    f"{row['bytes'] / 1048576.0:.2f} MiB, "
                    f"{row['render_throughput_fps']:.1f} fps",
                    flush=True,
                )
            except Exception as exc:
                row = {
                    "relative_npz": relative.as_posix(),
                    "relative_mp4": output_path.relative_to(args.output_root).as_posix(),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"  FAILED: {row['error']}", flush=True)
            rows.append(row)
            write_manifest(manifest_path, rows)
    finally:
        renderer.close()

    elapsed_s = time.perf_counter() - batch_started
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    print(
        f"Shard complete in {elapsed_s / 60.0:.2f} min: {counts}; manifest={manifest_path}",
        flush=True,
    )
    if counts.get("failed", 0):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

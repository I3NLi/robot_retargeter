#!/usr/bin/env python3
"""Validate a completed NPZ-to-MP4 batch and write an aggregate manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate rendered NPZ MP4 outputs.")
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=1)
    parser.add_argument("--target-fps", type=float, default=24.0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--probe-samples", type=int, default=48)
    parser.add_argument("--ffprobe", default=None)
    return parser.parse_args()


def bundled_ffprobe() -> str | None:
    suffix = ".exe" if sys.platform.startswith("win") else ""
    executable_dir = Path(sys.executable).resolve().parent
    for candidate in (
        executable_dir / "Library" / "bin" / f"ffprobe{suffix}",
        executable_dir / "bin" / f"ffprobe{suffix}",
    ):
        if candidate.is_file():
            return str(candidate)
    return shutil.which("ffprobe")


def read_manifests(output_root: Path, expected_shards: int) -> tuple[dict[str, dict[str, str]], list[str]]:
    manifest_dir = output_root / "_render_manifests"
    rows: dict[str, dict[str, str]] = {}
    errors: list[str] = []
    for shard_index in range(expected_shards):
        path = manifest_dir / f"render_manifest_shard_{shard_index:03d}_of_{expected_shards:03d}.csv"
        if not path.is_file():
            errors.append(f"missing manifest: {path}")
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                relative = str(row.get("relative_npz", ""))
                if not relative:
                    errors.append(f"blank relative_npz in {path}")
                    continue
                if relative in rows:
                    errors.append(f"duplicate manifest row: {relative}")
                    continue
                rows[relative] = row
                if row.get("status") == "failed":
                    errors.append(f"failed render: {relative}: {row.get('error', '')}")
    return rows, errors


def evenly_spaced_sample(items: list[Path], count: int) -> list[Path]:
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return items
    indices = np.linspace(0, len(items) - 1, count, dtype=np.int64)
    return [items[int(index)] for index in np.unique(indices)]


def probe_video(ffprobe: str, path: Path) -> dict[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_frames:format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"ffprobe exited {result.returncode}")
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise ValueError("no video stream")
    stream = streams[0]
    rate_text = str(stream.get("avg_frame_rate", "0/1"))
    numerator, denominator = rate_text.split("/", 1)
    rate = float(numerator) / float(denominator)
    return {
        "codec": stream.get("codec_name"),
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "pix_fmt": stream.get("pix_fmt"),
        "fps": rate,
        "frames": int(stream.get("nb_frames", 0)),
        "duration_s": float(payload.get("format", {}).get("duration", 0.0)),
    }


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    ffprobe = args.ffprobe or bundled_ffprobe()
    if ffprobe is None:
        raise FileNotFoundError("ffprobe executable not found")

    source_paths = sorted(input_root.rglob("*.npz"))
    video_paths = sorted(
        path
        for path in output_root.rglob("*.mp4")
        if not path.name.endswith(".partial.mp4")
    )
    partial_paths = sorted(output_root.rglob("*.partial.mp4"))
    source_relatives = {path.relative_to(input_root).as_posix() for path in source_paths}
    video_relatives = {
        path.relative_to(output_root).with_suffix(".npz").as_posix()
        for path in video_paths
    }
    missing = sorted(source_relatives - video_relatives)
    extra = sorted(video_relatives - source_relatives)
    manifest_rows, errors = read_manifests(output_root, args.expected_shards)
    manifest_missing = sorted(source_relatives - set(manifest_rows))
    if manifest_missing:
        errors.append(f"manifest rows missing: {len(manifest_missing)}")

    aggregate_rows: list[dict[str, Any]] = []
    total_source_frames = 0
    total_source_duration = 0.0
    total_video_bytes = 0
    for source_path in source_paths:
        relative = source_path.relative_to(input_root).as_posix()
        video_path = (output_root / Path(relative)).with_suffix(".mp4")
        with np.load(source_path, allow_pickle=False) as payload:
            frames = int(payload["joint_pos"].shape[0])
            fps = float(payload["framerate"])
        duration = frames / fps
        expected_rendered_frames = (
            frames if fps <= args.target_fps else max(1, int(round(frames * args.target_fps / fps)))
        )
        manifest = manifest_rows.get(relative, {})
        row = {
            "relative_npz": relative,
            "relative_mp4": video_path.relative_to(output_root).as_posix(),
            "source_frames": frames,
            "source_fps": fps,
            "duration_s": duration,
            "expected_rendered_frames": expected_rendered_frames,
            "manifest_status": manifest.get("status", "missing"),
            "manifest_rendered_frames": manifest.get("rendered_frames", ""),
            "video_exists": video_path.is_file(),
            "video_bytes": video_path.stat().st_size if video_path.is_file() else 0,
        }
        if manifest.get("rendered_frames"):
            if int(manifest["rendered_frames"]) != expected_rendered_frames:
                errors.append(
                    f"rendered frame mismatch: {relative}: "
                    f"manifest={manifest['rendered_frames']} expected={expected_rendered_frames}"
                )
        aggregate_rows.append(row)
        total_source_frames += frames
        total_source_duration += duration
        total_video_bytes += int(row["video_bytes"])

    probe_errors: list[str] = []
    probe_results: list[dict[str, Any]] = []
    for video_path in evenly_spaced_sample(video_paths, args.probe_samples):
        relative_mp4 = video_path.relative_to(output_root).as_posix()
        source_relative = video_path.relative_to(output_root).with_suffix(".npz").as_posix()
        source_row = next(row for row in aggregate_rows if row["relative_npz"] == source_relative)
        try:
            probe = probe_video(ffprobe, video_path)
            probe["relative_mp4"] = relative_mp4
            probe_results.append(probe)
            if probe["codec"] != "h264":
                probe_errors.append(f"codec is not h264: {relative_mp4}: {probe['codec']}")
            if probe["pix_fmt"] != "yuv420p":
                probe_errors.append(
                    f"pixel format is not yuv420p: {relative_mp4}: {probe['pix_fmt']}"
                )
            if (probe["width"], probe["height"]) != (args.width, args.height):
                probe_errors.append(
                    f"resolution mismatch: {relative_mp4}: {probe['width']}x{probe['height']}"
                )
            if probe["frames"] != int(source_row["expected_rendered_frames"]):
                probe_errors.append(
                    f"probed frame mismatch: {relative_mp4}: "
                    f"{probe['frames']} != {source_row['expected_rendered_frames']}"
                )
            if abs(probe["duration_s"] - float(source_row["duration_s"])) > 0.05:
                probe_errors.append(
                    f"duration mismatch: {relative_mp4}: "
                    f"{probe['duration_s']:.6f} != {source_row['duration_s']:.6f}"
                )
        except Exception as exc:
            probe_errors.append(f"probe failed: {relative_mp4}: {type(exc).__name__}: {exc}")
    errors.extend(probe_errors)

    manifest_path = report_dir / "recording_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "source_files": len(source_paths),
        "video_files": len(video_paths),
        "missing_videos": missing,
        "extra_videos": extra,
        "partial_files": [path.relative_to(output_root).as_posix() for path in partial_paths],
        "manifest_rows": len(manifest_rows),
        "manifest_missing_count": len(manifest_missing),
        "total_source_frames": total_source_frames,
        "total_source_duration_s": total_source_duration,
        "total_video_bytes": total_video_bytes,
        "target_fps": args.target_fps,
        "resolution": [args.width, args.height],
        "probe_sample_count": len(probe_results),
        "probe_results": probe_results,
        "errors": errors,
        "valid": not missing and not extra and not partial_paths and not errors,
    }
    summary_path = report_dir / "recording_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in (
        "source_files", "video_files", "manifest_rows", "total_video_bytes",
        "probe_sample_count", "errors", "valid"
    )}, ensure_ascii=False, indent=2))
    print(f"Wrote: {manifest_path}")
    print(f"Wrote: {summary_path}")
    if not summary["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

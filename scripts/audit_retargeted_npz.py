#!/usr/bin/env python3
"""Audit Kengo retargeted NPZ motions and write CSV/JSON/Markdown reports.

The input contract is the deployment motion format used by Kengo:
``framerate``, ``joint_names``, ``joint_pos``, ``base_pos_w`` and
``base_quat_w``.  The audit is deliberately reference-free: it checks
kinematic validity and temporal quality, but does not claim source-to-target
pose error when source keypoints are unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ROBOT_XML = PROJECT_DIR / "asset/robot/kengo_description/mjcf/kengo.xml"
EXPECTED_KEYS = (
    "framerate",
    "joint_names",
    "joint_pos",
    "base_pos_w",
    "base_quat_w",
)
FOOT_BODY_NAMES = (
    "left_foot_end_link",
    "left_toe_link",
    "right_foot_end_link",
    "right_toe_link",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit retargeted Kengo NPZ motions.")
    parser.add_argument("input_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_ROBOT_XML)
    parser.add_argument("--joint-velocity-limit", type=float, default=15.0)
    parser.add_argument("--joint-velocity-warning", type=float, default=10.0)
    parser.add_argument("--root-speed-warning", type=float, default=5.0)
    parser.add_argument("--root-angular-speed-warning", type=float, default=12.566370614359172)
    parser.add_argument("--foot-contact-height", type=float, default=0.05)
    parser.add_argument("--foot-penetration-depth", type=float, default=0.02)
    parser.add_argument("--foot-slip-warning", type=float, default=0.5)
    parser.add_argument("--limit-tolerance", type=float, default=1e-4)
    parser.add_argument("--quat-norm-tolerance", type=float, default=1e-3)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def finite_percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return 0.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(np.percentile(finite, percentile))


def finite_max(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(np.max(finite))


def model_joint_contract(
    model: mujoco.MjModel,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    names: list[str] = []
    qpos_addresses: list[int] = []
    lowers: list[float] = []
    uppers: list[float] = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name is None:
            raise ValueError(f"Unnamed non-free joint at model index {joint_id}")
        names.append(name)
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        lowers.append(float(model.jnt_range[joint_id, 0]))
        uppers.append(float(model.jnt_range[joint_id, 1]))
    return (
        names,
        np.asarray(qpos_addresses, dtype=np.int32),
        np.asarray(lowers, dtype=np.float64),
        np.asarray(uppers, dtype=np.float64),
    )


def load_motion(
    path: Path,
    model_joint_names: list[str],
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as payload:
        missing = [key for key in EXPECTED_KEYS if key not in payload.files]
        if missing:
            raise ValueError(f"missing keys: {missing}")
        fps = float(np.asarray(payload["framerate"]).reshape(()))
        source_joint_names = [str(item) for item in payload["joint_names"].tolist()]
        joint_pos = np.asarray(payload["joint_pos"], dtype=np.float64)
        base_pos = np.asarray(payload["base_pos_w"], dtype=np.float64)
        base_quat = np.asarray(payload["base_quat_w"], dtype=np.float64)

    if fps <= 0.0 or not math.isfinite(fps):
        raise ValueError(f"invalid framerate: {fps}")
    if joint_pos.ndim != 2:
        raise ValueError(f"joint_pos must be 2D, got {joint_pos.shape}")
    if base_pos.shape != (joint_pos.shape[0], 3):
        raise ValueError(
            f"base_pos_w shape mismatch: {base_pos.shape}, frames={joint_pos.shape[0]}"
        )
    if base_quat.shape != (joint_pos.shape[0], 4):
        raise ValueError(
            f"base_quat_w shape mismatch: {base_quat.shape}, frames={joint_pos.shape[0]}"
        )
    if joint_pos.shape[0] == 0:
        raise ValueError("motion has zero frames")
    if len(source_joint_names) != joint_pos.shape[1]:
        raise ValueError(
            f"joint_names count {len(source_joint_names)} != joint_pos columns {joint_pos.shape[1]}"
        )
    if len(set(source_joint_names)) != len(source_joint_names):
        raise ValueError("joint_names contains duplicates")
    missing_joints = [name for name in model_joint_names if name not in source_joint_names]
    extra_joints = [name for name in source_joint_names if name not in model_joint_names]
    if missing_joints or extra_joints:
        raise ValueError(
            f"joint schema mismatch; missing={missing_joints}, extra={extra_joints}"
        )
    reorder = [source_joint_names.index(name) for name in model_joint_names]
    return fps, joint_pos[:, reorder], base_pos, base_quat, source_joint_names


def foot_kinematics(
    model: mujoco.MjModel,
    qpos_addresses: np.ndarray,
    joint_pos: np.ndarray,
    base_pos: np.ndarray,
    base_quat_normalized: np.ndarray,
    foot_body_ids: np.ndarray,
) -> np.ndarray:
    data = mujoco.MjData(model)
    positions = np.empty((joint_pos.shape[0], len(foot_body_ids), 3), dtype=np.float32)
    for frame_idx in range(joint_pos.shape[0]):
        data.qpos[:3] = base_pos[frame_idx]
        data.qpos[3:7] = base_quat_normalized[frame_idx]
        data.qpos[qpos_addresses] = joint_pos[frame_idx]
        mujoco.mj_kinematics(model, data)
        positions[frame_idx] = data.xpos[foot_body_ids]
    return positions


def classify(row: dict[str, Any], args: argparse.Namespace) -> tuple[str, str]:
    failures: list[str] = []
    warnings: list[str] = []
    if row["finite_fraction"] < 1.0:
        failures.append("non_finite")
    if row["joint_limit_max_excess_rad"] > args.limit_tolerance:
        failures.append("joint_limit")
    if row["quat_norm_max_error"] > args.quat_norm_tolerance:
        failures.append("quat_norm")
    if row["joint_velocity_max_rad_s"] > args.joint_velocity_limit:
        failures.append("joint_velocity")

    if row["joint_velocity_max_rad_s"] > args.joint_velocity_warning:
        warnings.append("joint_velocity")
    if row["root_speed_max_m_s"] > args.root_speed_warning:
        warnings.append("root_speed")
    if row["root_angular_speed_max_rad_s"] > args.root_angular_speed_warning:
        warnings.append("root_angular_speed")
    if row["foot_penetration_frame_fraction"] > 0.01:
        warnings.append("foot_penetration")
    slip = row["foot_slip_contact_p95_m_s"]
    if math.isfinite(slip) and slip > args.foot_slip_warning:
        warnings.append("foot_slip")

    reasons = sorted(set(failures + warnings))
    if failures:
        return "FAIL", ";".join(reasons)
    if warnings:
        return "WARN", ";".join(reasons)
    return "PASS", ""


def audit_one(
    path: Path,
    input_root: Path,
    model: mujoco.MjModel,
    model_joint_names: list[str],
    qpos_addresses: np.ndarray,
    joint_lowers: np.ndarray,
    joint_uppers: np.ndarray,
    foot_body_ids: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    relative = path.relative_to(input_root).as_posix()
    row: dict[str, Any] = {
        "relative_path": relative,
        "status": "FAIL",
        "reasons": "load_error",
        "error": "",
    }
    try:
        fps, joint_pos, base_pos, base_quat, _source_names = load_motion(
            path, model_joint_names
        )
        arrays = (joint_pos, base_pos, base_quat)
        finite_count = sum(int(np.count_nonzero(np.isfinite(array))) for array in arrays)
        value_count = sum(int(array.size) for array in arrays)
        finite_fraction = finite_count / value_count if value_count else 0.0

        quat_norm = np.linalg.norm(base_quat, axis=1)
        safe_norm = np.where(quat_norm > 1e-12, quat_norm, 1.0)
        normalized_quat = base_quat / safe_norm[:, None]
        quat_norm_error = np.abs(quat_norm - 1.0)

        joint_velocity = np.diff(joint_pos, axis=0) * fps
        joint_abs_velocity = np.abs(joint_velocity)
        if joint_abs_velocity.size:
            peak_flat_idx = int(np.nanargmax(joint_abs_velocity))
            peak_frame_idx, peak_joint_idx = np.unravel_index(
                peak_flat_idx, joint_abs_velocity.shape
            )
            peak_joint_name = model_joint_names[int(peak_joint_idx)]
            peak_joint_frame = int(peak_frame_idx + 1)
        else:
            peak_joint_name = ""
            peak_joint_frame = 0

        joint_acceleration = np.diff(joint_velocity, axis=0) * fps
        root_velocity = np.diff(base_pos, axis=0) * fps
        root_speed = np.linalg.norm(root_velocity, axis=1)
        root_acceleration = np.diff(root_velocity, axis=0) * fps
        root_acceleration_norm = np.linalg.norm(root_acceleration, axis=1)

        if normalized_quat.shape[0] > 1:
            quat_dot = np.sum(normalized_quat[1:] * normalized_quat[:-1], axis=1)
            quat_delta_angle = 2.0 * np.arccos(np.clip(np.abs(quat_dot), 0.0, 1.0))
            root_angular_speed = quat_delta_angle * fps
        else:
            root_angular_speed = np.empty(0, dtype=np.float64)

        lower_excess = np.maximum(joint_lowers[None, :] - joint_pos, 0.0)
        upper_excess = np.maximum(joint_pos - joint_uppers[None, :], 0.0)
        limit_excess = np.maximum(lower_excess, upper_excess)
        limit_violation_mask = limit_excess > args.limit_tolerance
        limit_margin = np.minimum(
            joint_pos - joint_lowers[None, :], joint_uppers[None, :] - joint_pos
        )
        saturation_mask = (limit_margin >= -args.limit_tolerance) & (limit_margin <= 0.005)

        foot_positions = foot_kinematics(
            model=model,
            qpos_addresses=qpos_addresses,
            joint_pos=joint_pos,
            base_pos=base_pos,
            base_quat_normalized=normalized_quat,
            foot_body_ids=foot_body_ids,
        )
        minimum_foot_z_per_frame = np.min(foot_positions[:, :, 2], axis=1)
        penetration_frame_mask = minimum_foot_z_per_frame < -args.foot_penetration_depth

        side_centers = np.stack(
            (
                np.mean(foot_positions[:, 0:2, :], axis=1),
                np.mean(foot_positions[:, 2:4, :], axis=1),
            ),
            axis=1,
        )
        side_min_z = np.stack(
            (
                np.min(foot_positions[:, 0:2, 2], axis=1),
                np.min(foot_positions[:, 2:4, 2], axis=1),
            ),
            axis=1,
        )
        side_contact = side_min_z <= args.foot_contact_height
        side_planar_velocity = np.linalg.norm(
            np.diff(side_centers[:, :, :2], axis=0) * fps, axis=2
        )
        contact_intervals = side_contact[1:] & side_contact[:-1]
        contact_slip = side_planar_velocity[contact_intervals]

        row.update(
            {
                "frames": int(joint_pos.shape[0]),
                "fps": float(fps),
                "duration_s": float(joint_pos.shape[0] / fps),
                "finite_fraction": float(finite_fraction),
                "quat_norm_max_error": finite_max(quat_norm_error),
                "quat_norm_p99_error": finite_percentile(quat_norm_error, 99.0),
                "joint_limit_violation_values": int(np.count_nonzero(limit_violation_mask)),
                "joint_limit_violation_frames": int(
                    np.count_nonzero(np.any(limit_violation_mask, axis=1))
                ),
                "joint_limit_max_excess_rad": finite_max(limit_excess),
                "joint_limit_saturation_fraction": float(np.mean(saturation_mask)),
                "joint_velocity_max_rad_s": finite_max(joint_abs_velocity),
                "joint_velocity_p99_9_rad_s": finite_percentile(joint_abs_velocity, 99.9),
                "joint_acceleration_max_rad_s2": finite_max(np.abs(joint_acceleration)),
                "joint_acceleration_p99_9_rad_s2": finite_percentile(
                    np.abs(joint_acceleration), 99.9
                ),
                "peak_velocity_joint": peak_joint_name,
                "peak_velocity_frame": peak_joint_frame,
                "root_speed_max_m_s": finite_max(root_speed),
                "root_speed_p99_m_s": finite_percentile(root_speed, 99.0),
                "root_acceleration_max_m_s2": finite_max(root_acceleration_norm),
                "root_angular_speed_max_rad_s": finite_max(root_angular_speed),
                "root_angular_speed_p99_rad_s": finite_percentile(
                    root_angular_speed, 99.0
                ),
                "base_height_min_m": finite_max(-base_pos[:, 2]) * -1.0,
                "base_height_max_m": finite_max(base_pos[:, 2]),
                "foot_point_min_z_m": finite_max(-minimum_foot_z_per_frame) * -1.0,
                "foot_penetration_frame_fraction": float(np.mean(penetration_frame_mask)),
                "foot_contact_interval_fraction": float(np.mean(contact_intervals))
                if contact_intervals.size
                else 0.0,
                "foot_slip_contact_p95_m_s": finite_percentile(contact_slip, 95.0)
                if contact_slip.size
                else math.nan,
                "foot_slip_contact_max_m_s": finite_max(contact_slip)
                if contact_slip.size
                else math.nan,
            }
        )
        row["status"], row["reasons"] = classify(row, args)
    except Exception as exc:  # Continue the corpus audit and record the exact failure.
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def json_number(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def summarize(rows: list[dict[str, Any]], elapsed_s: float, args: argparse.Namespace) -> dict[str, Any]:
    statuses = Counter(str(row.get("status", "FAIL")) for row in rows)
    fps_counts = Counter(round(float(row["fps"]), 6) for row in rows if "fps" in row)
    reason_counts: Counter[str] = Counter()
    for row in rows:
        for reason in str(row.get("reasons", "")).split(";"):
            if reason:
                reason_counts[reason] += 1

    def metric_summary(name: str, subset: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        selected = rows if subset is None else subset
        values = np.asarray(
            [
                float(row[name])
                for row in selected
                if name in row and row[name] is not None
            ],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        if not values.size:
            return {"median": None, "p95": None, "max": None}
        return {
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95.0)),
            "max": float(np.max(values)),
        }

    valid_rows = [row for row in rows if "frames" in row]
    groups: dict[str, Any] = {}
    group_names = sorted(
        {str(row["relative_path"]).split("/", 1)[0] for row in valid_rows}
    )
    for group_name in group_names:
        group_rows = [
            row
            for row in valid_rows
            if str(row["relative_path"]).split("/", 1)[0] == group_name
        ]
        group_reasons: Counter[str] = Counter()
        for row in group_rows:
            for reason in str(row.get("reasons", "")).split(";"):
                if reason:
                    group_reasons[reason] += 1
        groups[group_name] = {
            "files": len(group_rows),
            "duration_s": float(sum(float(row["duration_s"]) for row in group_rows)),
            "status_counts": dict(Counter(str(row["status"]) for row in group_rows)),
            "reason_counts": dict(group_reasons),
            "metrics": {
                name: metric_summary(name, group_rows)
                for name in (
                    "joint_velocity_max_rad_s",
                    "joint_velocity_p99_9_rad_s",
                    "root_speed_max_m_s",
                    "root_angular_speed_max_rad_s",
                    "foot_penetration_frame_fraction",
                    "foot_slip_contact_p95_m_s",
                )
            },
        }
    return {
        "input_root": str(args.input_root.resolve()),
        "robot_xml": str(args.robot_xml.resolve()),
        "generated_unix_time": time.time(),
        "elapsed_s": elapsed_s,
        "files_total": len(rows),
        "files_loaded": len(valid_rows),
        "frames_total": int(sum(int(row["frames"]) for row in valid_rows)),
        "duration_total_s": float(sum(float(row["duration_s"]) for row in valid_rows)),
        "status_counts": dict(statuses),
        "reason_counts": dict(reason_counts),
        "fps_counts": {str(key): value for key, value in sorted(fps_counts.items())},
        "groups": groups,
        "peak_velocity_joint_counts_fail": dict(
            Counter(
                str(row.get("peak_velocity_joint", ""))
                for row in rows
                if row.get("status") == "FAIL" and row.get("peak_velocity_joint")
            ).most_common()
        ),
        "thresholds": {
            "joint_velocity_limit_rad_s": args.joint_velocity_limit,
            "joint_velocity_warning_rad_s": args.joint_velocity_warning,
            "root_speed_warning_m_s": args.root_speed_warning,
            "root_angular_speed_warning_rad_s": args.root_angular_speed_warning,
            "foot_contact_height_m": args.foot_contact_height,
            "foot_penetration_depth_m": args.foot_penetration_depth,
            "foot_slip_warning_m_s": args.foot_slip_warning,
            "limit_tolerance_rad": args.limit_tolerance,
            "quat_norm_tolerance": args.quat_norm_tolerance,
        },
        "metrics_across_clips": {
            name: metric_summary(name)
            for name in (
                "joint_velocity_max_rad_s",
                "joint_velocity_p99_9_rad_s",
                "joint_acceleration_max_rad_s2",
                "root_speed_max_m_s",
                "root_angular_speed_max_rad_s",
                "joint_limit_max_excess_rad",
                "foot_point_min_z_m",
                "foot_penetration_frame_fraction",
                "foot_slip_contact_p95_m_s",
            )
        },
        "limitations": [
            "Source human/keypoint trajectories are not present under the audited directory, so source-to-target positional or rotational error cannot be measured.",
            "Foot contact is inferred kinematically from foot endpoint height; no force/contact labels are stored in the NPZ files.",
            "The audit is kinematic and does not prove dynamic feasibility, actuator torque feasibility, or closed-loop tracking stability.",
        ],
    }


def write_markdown(
    output_path: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    status_counts = summary["status_counts"]
    reason_counts = summary["reason_counts"]
    metrics = summary["metrics_across_clips"]
    severity = {"FAIL": 0, "WARN": 1, "PASS": 2}
    ranked = sorted(
        rows,
        key=lambda row: (
            severity.get(str(row.get("status")), 0),
            -float(row.get("joint_velocity_max_rad_s", 0.0) or 0.0),
            -float(row.get("foot_slip_contact_p95_m_s", 0.0) or 0.0),
        ),
    )
    lines = [
        "# Kengo 重定向动作质量审计",
        "",
        f"- 输入目录：`{summary['input_root']}`",
        f"- 文件：{summary['files_total']}（成功读取 {summary['files_loaded']}）",
        f"- 总帧数：{summary['frames_total']:,}",
        f"- 总时长：{summary['duration_total_s'] / 3600.0:.3f} 小时",
        f"- 结论计数：PASS {status_counts.get('PASS', 0)} / WARN {status_counts.get('WARN', 0)} / FAIL {status_counts.get('FAIL', 0)}",
        "",
        "## 判定口径",
        "",
        "FAIL：数据无效、关节越限、四元数未归一化，或关节峰值速度超过项目配置的 15 rad/s。",
        "WARN：关节峰值速度超过 10 rad/s、根节点速度超过 5 m/s、根节点角速度超过 4π rad/s、超过 1% 帧出现脚端低于 -2 cm，或推断接触期脚滑 P95 超过 0.5 m/s。",
        "",
        "## 汇总指标（逐片段统计后的分布）",
        "",
        "| 指标 | 中位数 | P95 | 最大/最差 |",
        "|---|---:|---:|---:|",
    ]
    for name, label in (
        ("joint_velocity_max_rad_s", "关节峰值速度 rad/s"),
        ("joint_velocity_p99_9_rad_s", "关节速度 P99.9 rad/s"),
        ("joint_acceleration_max_rad_s2", "关节峰值加速度 rad/s²"),
        ("root_speed_max_m_s", "根节点峰值线速度 m/s"),
        ("root_angular_speed_max_rad_s", "根节点峰值角速度 rad/s"),
        ("joint_limit_max_excess_rad", "最大关节越限 rad"),
        ("foot_point_min_z_m", "脚端最低高度 m"),
        ("foot_penetration_frame_fraction", "穿地帧占比"),
        ("foot_slip_contact_p95_m_s", "接触期脚滑 P95 m/s"),
    ):
        item = metrics[name]
        values = [item["median"], item["p95"], item["max"]]
        formatted = ["n/a" if value is None else f"{value:.6g}" for value in values]
        lines.append(f"| {label} | {formatted[0]} | {formatted[1]} | {formatted[2]} |")

    lines.extend(["", "## 触发原因", ""])
    if reason_counts:
        for reason, count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- `{reason}`：{count} 段")
    else:
        lines.append("- 无")

    lines.extend(
        [
            "",
            "## 子数据集对比",
            "",
            "| 子目录 | 片段 | 时长 h | PASS | WARN | FAIL | 峰值关节速度中位数 | 脚滑 P95 中位数 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group_name, group in summary.get("groups", {}).items():
        counts = group["status_counts"]
        velocity = group["metrics"]["joint_velocity_max_rad_s"]["median"]
        slip = group["metrics"]["foot_slip_contact_p95_m_s"]["median"]
        lines.append(
            f"| `{group_name}` | {group['files']} | {group['duration_s'] / 3600.0:.3f} | "
            f"{counts.get('PASS', 0)} | {counts.get('WARN', 0)} | {counts.get('FAIL', 0)} | "
            f"{velocity:.3f} | {slip:.3f} |"
        )

    fail_peak_joints = summary.get("peak_velocity_joint_counts_fail", {})
    if fail_peak_joints:
        lines.extend(["", "FAIL 片段的峰值关节分布（前 10）：", ""])
        for joint_name, count in list(fail_peak_joints.items())[:10]:
            lines.append(f"- `{joint_name}`：{count} 段")

    lines.extend(
        [
            "",
            "## 优先复查片段（最多 50 段）",
            "",
            "| 状态 | 片段 | 原因 | 峰值关节速度 | 接触期脚滑 P95 | 穿地帧占比 |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in ranked[:50]:
        slip = row.get("foot_slip_contact_p95_m_s")
        slip_text = "n/a" if slip is None or not math.isfinite(float(slip)) else f"{float(slip):.3f}"
        lines.append(
            "| {status} | `{path}` | {reasons} | {vel:.3f} | {slip} | {pen:.3%} |".format(
                status=row.get("status", "FAIL"),
                path=str(row.get("relative_path", "")).replace("|", "\\|"),
                reasons=row.get("reasons", "") or "—",
                vel=float(row.get("joint_velocity_max_rad_s", 0.0) or 0.0),
                slip=slip_text,
                pen=float(row.get("foot_penetration_frame_fraction", 0.0) or 0.0),
            )
        )

    lines.extend(
        [
            "",
            "## 限制",
            "",
            "- 当前目录没有源人体/关键点轨迹，因此无法计算源—目标逐点位置误差或旋转误差。",
            "- 脚接触由脚端高度推断；NPZ 不含接触力或接触标签。",
            "- 本报告是运动学审计，不能替代扭矩可行性、动力学稳定性或闭环上机测试。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    robot_xml = args.robot_xml.expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_root}")
    if not robot_xml.is_file():
        raise FileNotFoundError(f"Robot MJCF not found: {robot_xml}")
    files = sorted(input_root.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ files found under: {input_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    model_joint_names, qpos_addresses, joint_lowers, joint_uppers = model_joint_contract(model)
    foot_body_ids = []
    for name in FOOT_BODY_NAMES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Foot body not found in model: {name}")
        foot_body_ids.append(int(body_id))

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, path in enumerate(files, start=1):
        rows.append(
            audit_one(
                path=path,
                input_root=input_root,
                model=model,
                model_joint_names=model_joint_names,
                qpos_addresses=qpos_addresses,
                joint_lowers=joint_lowers,
                joint_uppers=joint_uppers,
                foot_body_ids=np.asarray(foot_body_ids, dtype=np.int32),
                args=args,
            )
        )
        if index == 1 or index % args.progress_every == 0 or index == len(files):
            elapsed = time.perf_counter() - started
            print(
                f"Audited {index}/{len(files)} files "
                f"({index / max(elapsed, 1e-9):.1f} files/s)",
                flush=True,
            )

    elapsed_s = time.perf_counter() - started
    all_fields: list[str] = ["relative_path", "status", "reasons", "error"]
    for row in rows:
        for key in row:
            if key not in all_fields:
                all_fields.append(key)
    csv_path = output_dir / "per_file_quality.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows, elapsed_s, args)
    json_path = output_dir / "quality_summary.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_number) + "\n",
        encoding="utf-8",
    )
    markdown_path = output_dir / "quality_report_zh.md"
    write_markdown(markdown_path, summary, rows)
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {markdown_path}")


if __name__ == "__main__":
    main()

"""Realtime SMPL-24 to Kengo full-body retargeting primitives.

The implementation deliberately follows the kinematic method used by
``scripts/robot_retarget.py``:

* normalize the observed human link directions to Kengo link lengths;
* install weighted position/orientation FrameTasks from ``config/robot/kengo.yaml``;
* solve the constrained MuJoCo configuration with Mink and DAQP;
* warm-start every frame from the previous successful configuration.

This module has no ROS imports and never creates a control publisher.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Mapping

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml


SMPL_JOINT_COUNT = 24
SMPL_BODY_INDEX = OrderedDict(
    (
        ("hips", 0),
        ("left_up_leg", 1),
        ("left_leg", 4),
        ("left_foot", 7),
        ("left_toe", 10),
        ("right_up_leg", 2),
        ("right_leg", 5),
        ("right_foot", 8),
        ("right_toe", 11),
        ("spine1", 3),
        ("spine2", 6),
        ("chest", 9),
        ("neck", 12),
        ("head", 15),
        ("left_shoulder", 13),
        ("left_arm", 16),
        ("left_fore_arm", 18),
        ("left_hand", 20),
        ("right_shoulder", 14),
        ("right_arm", 17),
        ("right_fore_arm", 19),
        ("right_hand", 21),
    )
)

DERIVED_BODY_CENTERS = {
    "hips_mean": ("left_up_leg", "right_up_leg"),
    "shoulder_mean": ("left_arm", "right_arm"),
}

SKELETON_LINKS = OrderedDict(
    (
        ("left_hip", ("hips_mean", "left_up_leg")),
        ("left_thigh", ("left_up_leg", "left_leg")),
        ("left_calf", ("left_leg", "left_foot")),
        ("right_hip", ("hips_mean", "right_up_leg")),
        ("right_thigh", ("right_up_leg", "right_leg")),
        ("right_calf", ("right_leg", "right_foot")),
        ("neck", ("hips_mean", "shoulder_mean")),
        ("head", ("shoulder_mean", "head")),
        ("left_shoulder", ("shoulder_mean", "left_arm")),
        ("left_arm", ("left_arm", "left_fore_arm")),
        ("left_fore_arm", ("left_fore_arm", "left_hand")),
        ("right_shoulder", ("shoulder_mean", "right_arm")),
        ("right_arm", ("right_arm", "right_fore_arm")),
        ("right_fore_arm", ("right_fore_arm", "right_hand")),
    )
)

BODY_CHILDREN = {
    "left_up_leg": "left_leg",
    "left_leg": "left_foot",
    "left_foot": "left_toe",
    "right_up_leg": "right_leg",
    "right_leg": "right_foot",
    "right_foot": "right_toe",
    "spine1": "spine2",
    "spine2": "chest",
    "chest": "neck",
    "neck": "head",
    "left_shoulder": "left_arm",
    "left_arm": "left_fore_arm",
    "left_fore_arm": "left_hand",
    "right_shoulder": "right_arm",
    "right_arm": "right_fore_arm",
    "right_fore_arm": "right_hand",
}

BODY_LOCAL_DIRECTIONS = {
    "left_up_leg": np.array((0.0, -1.0, 0.0)),
    "left_leg": np.array((0.0, -1.0, 0.0)),
    "left_foot": np.array((0.0, -0.054, 0.125)),
    "right_up_leg": np.array((0.0, -1.0, 0.0)),
    "right_leg": np.array((0.0, -1.0, 0.0)),
    "right_foot": np.array((0.0, -0.054, 0.125)),
    "spine1": np.array((0.0, 1.0, 0.0)),
    "spine2": np.array((0.0, 1.0, 0.0)),
    "chest": np.array((0.0, 1.0, 0.0)),
    "neck": np.array((0.0, 1.0, 0.0)),
    "left_shoulder": np.array((1.0, 0.0, 0.0)),
    "left_arm": np.array((1.0, 0.0, 0.0)),
    "left_fore_arm": np.array((1.0, 0.0, 0.0)),
    "right_shoulder": np.array((-1.0, 0.0, 0.0)),
    "right_arm": np.array((-1.0, 0.0, 0.0)),
    "right_fore_arm": np.array((-1.0, 0.0, 0.0)),
}

# Same SDK rest-axis correction used by the deployed SMPL projector.
SDK_TO_SMPL_REST_ROTATION = np.diag((-1.0, 1.0, -1.0))

KENGO_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "waist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)


class RetargetError(RuntimeError):
    """A frame or solve failed validation without changing the last solution."""


@dataclass(frozen=True)
class TargetFrame:
    positions: Mapping[str, np.ndarray]
    quaternions_wxyz: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class RetargetResult:
    joint_names: tuple[str, ...]
    positions: np.ndarray
    root_position: np.ndarray
    root_quaternion_xyzw: np.ndarray
    task_error: float
    iterations: int
    solve_duration_ms: float


def _unit(vector: np.ndarray, label: str, minimum: float = 1.0e-5) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(value))
    if not math.isfinite(length) or length < minimum:
        raise RetargetError(f"{label} is degenerate")
    return value / length


def validate_smpl24(
    positions: np.ndarray, quaternions_xyzw: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(positions, dtype=np.float64)
    quat = np.asarray(quaternions_xyzw, dtype=np.float64)
    if pos.shape != (SMPL_JOINT_COUNT, 3):
        raise RetargetError(f"expected positions shape (24, 3), got {pos.shape}")
    if quat.shape != (SMPL_JOINT_COUNT, 4):
        raise RetargetError(f"expected quaternion shape (24, 4), got {quat.shape}")
    if not np.isfinite(pos).all() or not np.isfinite(quat).all():
        raise RetargetError("pose contains non-finite values")
    if float(np.max(np.abs(pos))) > 50.0:
        raise RetargetError("pose exceeds the 50 metre tracking envelope")
    norms = np.linalg.norm(quat, axis=1)
    if np.any(norms < 1.0e-6):
        raise RetargetError("pose contains a zero-length quaternion")
    return np.ascontiguousarray(pos), np.ascontiguousarray(quat / norms[:, None])


def torso_basis(positions: np.ndarray) -> np.ndarray:
    """Return rows mapping the tracking frame to Kengo forward/left/up."""

    right = _unit(positions[17] - positions[16], "shoulder axis", minimum=0.05)
    up_hint = positions[12] - positions[0]
    up = _unit(up_hint - right * float(np.dot(up_hint, right)), "torso up", minimum=0.05)
    left = -right
    forward = _unit(np.cross(left, up), "torso forward")
    basis = np.stack((forward, left, up), axis=0)
    if float(np.linalg.det(basis)) < 0.999:
        raise RetargetError("torso basis is not right-handed")
    return basis


def _average_quaternions_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    right_aligned = right if float(np.dot(left, right)) >= 0.0 else -right
    result = left + right_aligned
    return result / np.linalg.norm(result)


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_u = _unit(source, "source direction")
    target_u = _unit(target, "target direction")
    cosine = float(np.clip(np.dot(source_u, target_u), -1.0, 1.0))
    axis = np.cross(source_u, target_u)
    sine = float(np.linalg.norm(axis))
    if sine < 1.0e-8:
        if cosine > 0.0:
            return np.eye(3)
        fallback = np.zeros(3)
        fallback[int(np.argmin(np.abs(source_u)))] = 1.0
        axis = _unit(np.cross(source_u, fallback), "antiparallel rotation axis")
        return Rotation.from_rotvec(math.pi * axis).as_matrix()
    axis /= sine
    return Rotation.from_rotvec(math.atan2(sine, cosine) * axis).as_matrix()


def _align_bone_rotations(
    positions: Mapping[str, np.ndarray], rotations: dict[str, np.ndarray]
) -> None:
    for body_name, child_name in BODY_CHILDREN.items():
        if body_name not in positions or child_name not in positions:
            continue
        bone = positions[child_name] - positions[body_name]
        if float(np.linalg.norm(bone)) < 1.0e-6:
            continue
        predicted = rotations[body_name] @ BODY_LOCAL_DIRECTIONS[body_name]
        rotations[body_name] = _rotation_between(predicted, bone) @ rotations[body_name]


def _two_bone_knee(
    hip: np.ndarray, knee: np.ndarray, foot: np.ndarray, target_foot: np.ndarray
) -> np.ndarray:
    upper = knee - hip
    lower = foot - knee
    upper_len = float(np.linalg.norm(upper))
    lower_len = float(np.linalg.norm(lower))
    original = foot - hip
    original_u = _unit(original, "hip-to-foot")
    knee_projection = hip + float(np.dot(knee - hip, original_u)) * original_u
    bend_pref = knee - knee_projection
    if float(np.linalg.norm(bend_pref)) < 1.0e-6:
        bend_pref = np.cross(original_u, np.array((0.0, 0.0, 1.0)))
    if float(np.linalg.norm(bend_pref)) < 1.0e-6:
        bend_pref = np.cross(original_u, np.array((0.0, 1.0, 0.0)))
    bend_pref = _unit(bend_pref, "knee bend preference")

    target_vec = target_foot - hip
    distance = float(np.linalg.norm(target_vec))
    target_u = _unit(target_vec, "target hip-to-foot")
    distance = float(np.clip(distance, 1.0e-6, upper_len + lower_len - 1.0e-6))
    along = (upper_len * upper_len - lower_len * lower_len + distance * distance) / (
        2.0 * distance
    )
    height = math.sqrt(max(upper_len * upper_len - along * along, 0.0))
    bend_dir = bend_pref - float(np.dot(bend_pref, target_u)) * target_u
    if float(np.linalg.norm(bend_dir)) < 1.0e-6:
        bend_dir = np.cross(target_u, np.array((0.0, 1.0, 0.0)))
    bend_dir = _unit(bend_dir, "target knee bend")
    return hip + along * target_u + height * bend_dir


def _bend_leg(positions: dict[str, np.ndarray], side: str, offset_degrees: float) -> None:
    hip_name = f"{side}_up_leg"
    knee_name = f"{side}_leg"
    foot_name = f"{side}_foot"
    hip, knee, foot = positions[hip_name], positions[knee_name], positions[foot_name]
    upper = knee - hip
    lower = foot - knee
    upper_len = float(np.linalg.norm(upper))
    lower_len = float(np.linalg.norm(lower))
    current_cos = float(
        np.clip(np.dot(upper, lower) / max(upper_len * lower_len, 1.0e-8), -1.0, 1.0)
    )
    blended_angle = math.acos(current_cos) + math.radians(float(offset_degrees))
    target_length = math.sqrt(
        max(
            upper_len * upper_len
            + lower_len * lower_len
            + 2.0 * upper_len * lower_len * math.cos(blended_angle),
            0.0,
        )
    )
    target_foot = hip + _unit(foot - hip, f"{side} hip-to-foot") * target_length
    positions[knee_name] = _two_bone_knee(hip, knee, foot, target_foot)
    positions[foot_name] = target_foot


def _axis_map_matrix(config: Mapping[str, object], body_name: str) -> np.ndarray:
    body_config = config.get(body_name, {})
    if not isinstance(body_config, Mapping):
        raise RetargetError(f"invalid key_frame_config entry for {body_name}")
    axes = body_config.get("axis_map_cols")
    if not isinstance(axes, Mapping):
        return np.eye(3)
    matrix = np.column_stack(
        tuple(np.asarray(axes[name], dtype=np.float64) for name in ("x", "y", "z"))
    )
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise RetargetError(f"invalid axis map for {body_name}")
    return matrix


def _offset_matrix(config: Mapping[str, object], body_name: str) -> np.ndarray:
    body_config = config.get(body_name, {})
    if not isinstance(body_config, Mapping):
        raise RetargetError(f"invalid key_frame_config entry for {body_name}")
    degrees = np.asarray(body_config.get("offset_deg_xyz", (0.0, 0.0, 0.0)), dtype=np.float64)
    if degrees.shape != (3,) or not np.isfinite(degrees).all():
        raise RetargetError(f"invalid local rotation offset for {body_name}")
    # The offline pipeline composes Rx @ Ry @ Rz, not scipy's intrinsic xyz helper.
    rx, ry, rz = np.radians(degrees)
    rot_x = Rotation.from_rotvec(np.array((rx, 0.0, 0.0))).as_matrix()
    rot_y = Rotation.from_rotvec(np.array((0.0, ry, 0.0))).as_matrix()
    rot_z = Rotation.from_rotvec(np.array((0.0, 0.0, rz))).as_matrix()
    return rot_x @ rot_y @ rot_z


def build_target_frame(
    positions: np.ndarray,
    quaternions_xyzw: np.ndarray,
    robot_link_lengths: Mapping[str, float],
    robot_links: Mapping[str, tuple[str, str]],
    key_frame_config: Mapping[str, object],
    knee_angle_offset_degrees: float,
) -> TargetFrame:
    pos, quat = validate_smpl24(positions, quaternions_xyzw)
    basis = torso_basis(pos)
    canonical_positions = (basis @ (pos - pos[0]).T).T
    sdk_rotations = Rotation.from_quat(quat).as_matrix()
    canonical_rotations = np.einsum(
        "ij,njk,kl->nil", basis, sdk_rotations, SDK_TO_SMPL_REST_ROTATION
    )

    body_positions: dict[str, np.ndarray] = {}
    body_rotations: dict[str, np.ndarray] = {}
    for body_name, source_index in SMPL_BODY_INDEX.items():
        body_positions[body_name] = canonical_positions[source_index].copy()
        body_rotations[body_name] = canonical_rotations[source_index].copy()
    for body_name, (left_name, right_name) in DERIVED_BODY_CENTERS.items():
        body_positions[body_name] = 0.5 * (
            body_positions[left_name] + body_positions[right_name]
        )
        left_quat = Rotation.from_matrix(body_rotations[left_name]).as_quat()[[3, 0, 1, 2]]
        right_quat = Rotation.from_matrix(body_rotations[right_name]).as_quat()[[3, 0, 1, 2]]
        average = _average_quaternions_wxyz(left_quat, right_quat)
        body_rotations[body_name] = Rotation.from_quat(average[[1, 2, 3, 0]]).as_matrix()

    _align_bone_rotations(body_positions, body_rotations)

    scaled_positions = {name: value.copy() for name, value in body_positions.items()}
    for link_name in robot_links:
        if link_name not in SKELETON_LINKS or link_name not in robot_link_lengths:
            raise RetargetError(f"missing link geometry for {link_name}")
        parent, child = SKELETON_LINKS[link_name]
        direction = _unit(
            body_positions[child] - body_positions[parent], f"{link_name} source link"
        )
        scaled_positions[child] = (
            scaled_positions[parent] + float(robot_link_lengths[link_name]) * direction
        )

    _bend_leg(scaled_positions, "left", knee_angle_offset_degrees)
    _bend_leg(scaled_positions, "right", knee_angle_offset_degrees)
    _align_bone_rotations(scaled_positions, body_rotations)

    # Put the ankle keypoints on a stable z=0 reference. Translation does not
    # affect the solved scalar joints, but keeps the free root deterministic.
    floor = min(scaled_positions["left_foot"][2], scaled_positions["right_foot"][2])
    for name in scaled_positions:
        scaled_positions[name] = scaled_positions[name] - np.array((0.0, 0.0, floor))

    target_positions: dict[str, np.ndarray] = {
        "hips_mean": scaled_positions["hips_mean"].astype(np.float64)
    }
    hips_rotation = body_rotations["hips_mean"] @ _axis_map_matrix(
        key_frame_config, "hips_mean"
    ) @ _offset_matrix(key_frame_config, "hips_mean")
    target_quaternions: dict[str, np.ndarray] = {
        "hips_mean": Rotation.from_matrix(hips_rotation).as_quat()[[3, 0, 1, 2]]
    }
    for link_name in robot_links:
        _parent, child = SKELETON_LINKS[link_name]
        adjusted = body_rotations[child] @ _axis_map_matrix(
            key_frame_config, child
        ) @ _offset_matrix(key_frame_config, child)
        target_positions[link_name] = scaled_positions[child].astype(np.float64)
        target_quaternions[link_name] = Rotation.from_matrix(adjusted).as_quat()[
            [3, 0, 1, 2]
        ]
    return TargetFrame(target_positions, target_quaternions)


class RealtimeKengoRetargeter:
    """Stateful, failure-atomic full-body Mink retargeter."""

    def __init__(self, model_path: Path | str, config_path: Path | str) -> None:
        self.model_path = Path(model_path).resolve()
        self.config_path = Path(config_path).resolve()
        with self.config_path.open("r", encoding="utf-8") as stream:
            raw_config = yaml.safe_load(stream)
        if not isinstance(raw_config, Mapping):
            raise RetargetError("Kengo config must be a mapping")
        self.raw_config = raw_config
        self.robot_links = OrderedDict(
            (str(name), tuple(pair)) for name, pair in raw_config["robot_links"].items()
        )
        self.ik_match_table = OrderedDict(
            (str(name), tuple(entry)) for name, entry in raw_config["ik_match_table"].items()
        )
        self.key_frame_config = raw_config.get("key_frame_config", {})
        self.knee_angle_offset_degrees = float(
            raw_config.get("knee_angle_offset_degrees", 15.0)
        )
        self.max_iterations = 50
        self.convergence_delta = 0.001
        self.damping = 1.0
        self.solver = "daqp"

        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self._apply_joint_limit_offsets(raw_config.get("joints_limit_offset_degrees", {}))
        self.configuration = mink.Configuration(self.model)
        self._neutral_q = self.configuration.q.copy()
        self.limits = [mink.ConfigurationLimit(self.model)]
        self.tasks: list[mink.FrameTask] = []
        self.task_by_keypoint: dict[str, mink.FrameTask] = {}
        for keypoint_name, entry in self.ik_match_table.items():
            frame_name, position_cost, orientation_cost = entry
            if float(position_cost) == 0.0 and float(orientation_cost) == 0.0:
                continue
            task = mink.FrameTask(
                frame_name=str(frame_name),
                frame_type="body",
                position_cost=float(position_cost),
                orientation_cost=float(orientation_cost),
                lm_damping=1.0,
            )
            self.tasks.append(task)
            self.task_by_keypoint[keypoint_name] = task
        self.joint_names, self.qpos_indices = self._actuated_joint_qpos()
        if self.joint_names != KENGO_JOINT_NAMES:
            raise RetargetError(
                "unexpected Kengo actuator contract: " + ",".join(self.joint_names)
            )
        self.robot_link_lengths = self._robot_link_lengths()

    def _apply_joint_limit_offsets(self, offset_config: object) -> None:
        if not isinstance(offset_config, Mapping):
            raise RetargetError("joints_limit_offset_degrees must be a mapping")
        for name_fragment, values in offset_config.items():
            if isinstance(values, (list, tuple)) and len(values) == 2:
                lower_offset, upper_offset = map(float, values)
            else:
                lower_offset, upper_offset = float(values), 0.0
            matched = False
            for joint_id in range(self.model.njnt):
                joint_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
                )
                if joint_name is None or str(name_fragment) not in joint_name:
                    continue
                matched = True
                self.model.jnt_range[joint_id, 0] += math.radians(lower_offset)
                self.model.jnt_range[joint_id, 1] += math.radians(upper_offset)
            if not matched:
                raise RetargetError(f"joint limit offset matched nothing: {name_fragment}")

    def _actuated_joint_qpos(self) -> tuple[tuple[str, ...], np.ndarray]:
        names: list[str] = []
        indices: list[int] = []
        seen: set[int] = set()
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id < 0 or joint_id in seen:
                continue
            seen.add(joint_id)
            joint_type = int(self.model.jnt_type[joint_id])
            if joint_type not in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name is None:
                raise RetargetError(f"actuated joint {joint_id} has no name")
            names.append(name)
            indices.append(int(self.model.jnt_qposadr[joint_id]))
        return tuple(names), np.asarray(indices, dtype=np.int32)

    def _robot_link_lengths(self) -> dict[str, float]:
        mujoco.mj_forward(self.model, self.configuration.data)
        result: dict[str, float] = {}
        for link_name, (start_body, end_body) in self.robot_links.items():
            start_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, start_body
            )
            end_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, end_body)
            if start_id < 0 or end_id < 0:
                raise RetargetError(f"missing MJCF body for {link_name}")
            result[link_name] = float(
                np.linalg.norm(
                    self.configuration.data.xpos[end_id]
                    - self.configuration.data.xpos[start_id]
                )
            )
        return result

    def reset(self) -> None:
        self.configuration.update(self._neutral_q.copy())

    def make_targets(
        self, positions: np.ndarray, quaternions_xyzw: np.ndarray
    ) -> TargetFrame:
        return build_target_frame(
            positions,
            quaternions_xyzw,
            self.robot_link_lengths,
            self.robot_links,
            self.key_frame_config,
            self.knee_angle_offset_degrees,
        )

    def _set_targets(self, frame: TargetFrame) -> None:
        for keypoint_name, task in self.task_by_keypoint.items():
            if keypoint_name not in frame.positions:
                raise RetargetError(f"target frame is missing {keypoint_name}")
            task.set_target(
                mink.SE3.from_rotation_and_translation(
                    mink.SO3(frame.quaternions_wxyz[keypoint_name]),
                    frame.positions[keypoint_name],
                )
            )

    def _error(self) -> float:
        value = float(
            np.linalg.norm(
                np.concatenate(
                    tuple(task.compute_error(self.configuration) for task in self.tasks)
                )
            )
        )
        if not math.isfinite(value):
            raise RetargetError("Mink task error is non-finite")
        return value

    def solve_target_frame(self, frame: TargetFrame) -> RetargetResult:
        before = self.configuration.q.copy()
        started = time.perf_counter()
        try:
            self._set_targets(frame)
            current_error = self._error()
            timestep = float(self.model.opt.timestep)
            iteration_count = 0
            while True:
                velocity = mink.solve_ik(
                    self.configuration,
                    self.tasks,
                    timestep,
                    self.solver,
                    self.damping,
                    limits=self.limits,
                )
                if not np.isfinite(velocity).all():
                    raise RetargetError("Mink returned a non-finite velocity")
                self.configuration.integrate_inplace(velocity, timestep)
                iteration_count += 1
                next_error = self._error()
                if (
                    current_error - next_error <= self.convergence_delta
                    or iteration_count >= self.max_iterations + 1
                ):
                    current_error = next_error
                    break
                current_error = next_error
            qpos = self.configuration.q.copy()
            if not np.isfinite(qpos).all():
                raise RetargetError("retarget result contains non-finite qpos")
            root_quat = qpos[3:7]
            root_quat /= np.linalg.norm(root_quat)
            return RetargetResult(
                self.joint_names,
                qpos[self.qpos_indices].astype(np.float64),
                qpos[:3].astype(np.float64),
                root_quat[[1, 2, 3, 0]].astype(np.float64),
                current_error,
                iteration_count,
                (time.perf_counter() - started) * 1000.0,
            )
        except Exception:
            self.configuration.update(before)
            raise

    def solve(self, positions: np.ndarray, quaternions_xyzw: np.ndarray) -> RetargetResult:
        return self.solve_target_frame(self.make_targets(positions, quaternions_xyzw))


def synthetic_smpl24() -> tuple[np.ndarray, np.ndarray]:
    """A deterministic PICO-style upright pose used by tests and self-test."""

    positions = np.zeros((24, 3), dtype=np.float64)
    positions[0] = (0.0, 1.00, 0.0)
    positions[1], positions[2] = (-0.09, 0.95, 0.0), (0.09, 0.95, 0.0)
    positions[3] = (0.0, 1.12, 0.0)
    positions[4], positions[5] = (-0.09, 0.55, 0.02), (0.09, 0.55, 0.02)
    positions[6] = (0.0, 1.25, 0.0)
    positions[7], positions[8] = (-0.09, 0.12, 0.0), (0.09, 0.12, 0.0)
    positions[9] = (0.0, 1.38, 0.0)
    positions[10], positions[11] = (-0.09, 0.05, -0.12), (0.09, 0.05, -0.12)
    positions[12] = (0.0, 1.52, 0.0)
    positions[13], positions[14] = (-0.11, 1.47, 0.0), (0.11, 1.47, 0.0)
    positions[15] = (0.0, 1.72, 0.0)
    positions[16], positions[17] = (-0.24, 1.45, 0.0), (0.24, 1.45, 0.0)
    positions[18], positions[19] = (-0.50, 1.43, -0.02), (0.50, 1.43, -0.02)
    positions[20], positions[21] = (-0.74, 1.41, -0.04), (0.74, 1.41, -0.04)
    positions[22], positions[23] = (-0.80, 1.41, -0.04), (0.80, 1.41, -0.04)
    quaternions = np.zeros((24, 4), dtype=np.float64)
    quaternions[:, 3] = 1.0
    return positions, quaternions

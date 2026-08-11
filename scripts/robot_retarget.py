"""Retarget keypoint motions to a robot and export MuJoCo qpos CSV.

This script loads robot and keypoint settings from a robot YAML config,
runs IK-based retargeting frame by frame, and saves the resulting motion
as a CSV under output_data/robot_motion.

Usage:
    # Run with a specific robot config from terminal
    python scripts/robot_retarget.py --config config/robot/h2.yaml

    # Override render_debug from terminal (highest priority)
    python scripts/robot_retarget.py --config config/robot/agibot_x2.yaml --render-debug

    # Override keypoints_path from terminal by motion stem only
    python scripts/robot_retarget.py --config config/robot/h2.yaml --keypoints-name body_check_001__A548_M_from_g1
"""

import argparse
import mink
import mujoco
import mujoco.viewer
import numpy as np
import yaml
import pickle
import time
import csv
import copy
import os
import glfw
from tqdm import tqdm
from scipy.spatial.transform import Rotation as Rot

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Global pause state and keyboard callback
_PAUSED = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retarget keypoint motions to a robot from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("config", "robot", "h2.yaml"),
        help="Robot YAML config path (default: config/robot/h2.yaml)",
    )
    parser.add_argument(
        "--keypoints-name",
        type=str,
        default=None,
        help=(
            "Override keypoints_path by motion stem only. "
            "Example: --keypoints-name body_check_001__A548_M_from_g1 -> "
            "output_data/keypoints/<config_stem>/<name>_keypoints.pkl"
        ),
    )
    render_debug_group = parser.add_mutually_exclusive_group()
    render_debug_group.add_argument(
        "--render-debug",
        dest="render_debug",
        action="store_true",
        help="Force enable MuJoCo debug viewer (overrides YAML render_debug).",
    )
    render_debug_group.add_argument(
        "--no-render-debug",
        dest="render_debug",
        action="store_false",
        help="Force disable MuJoCo debug viewer (overrides YAML render_debug).",
    )
    parser.set_defaults(render_debug=None)
    return parser.parse_args()


def key_callback(keycode: int) -> None:
    """Toggle pause/play with the space key."""
    global _PAUSED
    if keycode == glfw.KEY_SPACE:
        _PAUSED = not _PAUSED


class RobotRetarget:

    LEGACY_CONTACT_CONFIG_TO_NAMES = {
        "foot_end_link": ("left_foot_end", "right_foot_end"),
        "toe_link": ("left_toe", "right_toe"),
        "hand_link": ("left_hand", "right_hand"),
    }

    def __init__(
        self,
        model_path: str,
        keypoint_path: str,
        ik_match_table: dict,
        solver: str = "daqp",
        verbose: bool = False,
        damping: float = 1.0,
        render_debug: bool = False,
        joints_limit_offset_degrees: dict | None = None,
        contact_body_names: list | tuple | dict | None = None,
        contact_position_cost: float = 10.0,
        initialization_sweep_frames: int = 0,
        motion_validation: dict | None = None,
    ):
        self.xml_file = model_path
        self.keypoint_path = keypoint_path
        self.ik_match_table = ik_match_table
        self.model = self._load_model_with_ground(self.xml_file)
        self.data = mujoco.MjData(self.model)
        self.max_iter = 50
        self.damping = damping
        self.render_debug = render_debug
        self.joints_limit_offset_degrees = joints_limit_offset_degrees or {}
        self.contact_body_names = self._normalize_contact_body_names(contact_body_names)
        self.contact_position_cost = float(contact_position_cost)
        if (
            isinstance(initialization_sweep_frames, bool)
            or int(initialization_sweep_frames) != initialization_sweep_frames
        ):
            raise ValueError("initialization_sweep_frames must be a non-negative integer")
        self.initialization_sweep_frames = int(initialization_sweep_frames)
        if self.initialization_sweep_frames < 0:
            raise ValueError("initialization_sweep_frames must be a non-negative integer")
        self.motion_validation = dict(motion_validation or {})
        self.verbose = verbose
        self.solver = solver
        self.human_body_to_task = {}
        self.task_errors = {}
        self.result_pos = []
        self.contact_state_name_to_idx = {}
        self.keypoint_name_to_idx = {}
        self.body_name_to_source_keypoint = {}
        self.body_name_to_contact_task = {}
        self.contact_targets = []
        self.current_contact_points = []

        self.robot_motor_names = {}

        self.setup_retarget_configuration()
        self.load_keypoints()
        self.setup_contact_targets()

    def _normalize_contact_body_names(self, contact_body_names):
        if contact_body_names is None:
            return []
        if isinstance(contact_body_names, dict):
            flattened_body_names = []
            for field_name, contact_names in self.LEGACY_CONTACT_CONFIG_TO_NAMES.items():
                body_names = contact_body_names.get(field_name, [])
                if len(body_names) != len(contact_names):
                    continue
                flattened_body_names.extend(body_names)
            return flattened_body_names
        if isinstance(contact_body_names, (list, tuple)):
            return list(contact_body_names)
        raise TypeError(
            "contact_body_names must be a list/tuple of body names or a legacy contact-body mapping"
        )

    def _load_model_with_ground(self, xml_file: str) -> mujoco.MjModel:
        """Load the robot XML and add a MuJoCo default-style ground (checker plane + skybox).

        The ground geom has collision disabled (contype/conaffinity=0) and serves only as a
        visual reference; it does not affect mink IK's pure kinematic solving.
        """
        spec = mujoco.MjSpec.from_file(xml_file)

        # Skip if ground/skybox assets already exist to avoid name conflicts
        existing_tex = {t.name for t in spec.textures}
        existing_mat = {m.name for m in spec.materials}

        if "skybox" not in existing_tex:
            spec.add_texture(
                name="skybox",
                type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                rgb1=[0.3, 0.5, 0.7],
                rgb2=[0.0, 0.0, 0.0],
                width=512,
                height=512,
            )
        if "groundplane" not in existing_tex:
            spec.add_texture(
                name="groundplane",
                type=mujoco.mjtTexture.mjTEXTURE_2D,
                builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                rgb1=[0.2, 0.3, 0.4],
                rgb2=[0.1, 0.2, 0.3],
                width=300,
                height=300,
            )
        if "groundplane" not in existing_mat:
            spec.add_material(
                name="groundplane",
                textures=["", "groundplane"],
                texrepeat=[5, 5],
                texuniform=True,
                reflectance=0.2,
            )

        # Lighting: main directional light + angled fill light to avoid an overly dark scene
        spec.worldbody.add_light(
            pos=[0, 0, 20.0],
            dir=[0, 0, -1],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            diffuse=[0.7, 0.7, 0.7],
            specular=[0.3, 0.3, 0.3],
        )
        spec.worldbody.add_light(
            pos=[4, 4, 6.0],
            dir=[-0.5, -0.5, -1],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            diffuse=[0.4, 0.4, 0.4],
            specular=[0.1, 0.1, 0.1],
        )

        # Add the (infinite) ground plane, collision disabled, for visualization only
        ground = spec.worldbody.add_geom(
            name="ground",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0, 0, 0.05],
            material="groundplane",
            pos=[0, 0, 0],
        )
        ground.contype = 0
        ground.conaffinity = 0

        return spec.compile()

    def load_keypoints(self):
        with open(self.keypoint_path, "rb") as f:
                keypoints_data = pickle.load(f)
        self.keypoint_names = keypoints_data["keypoint_names"]
        self.keypoint_name_to_idx = {
            keypoint_name: idx for idx, keypoint_name in enumerate(self.keypoint_names)
        }
        self.keypoints_pos = keypoints_data["positions"] 
        self.keypoints_quat = keypoints_data["quaternions"]  
        self.contact_names = keypoints_data.get("contact_names", [])
        self.contact_state_name_to_idx = {
            contact_name: idx for idx, contact_name in enumerate(self.contact_names)
        }
        self.contact_seq = keypoints_data["contact_states"]
        self.fps = keypoints_data.get("fps", 30)
        self.time_step = 1.0 / self.fps
        self.num_frames = self.keypoints_pos.shape[0]
        self.num_keypoints = self.keypoints_pos.shape[1]

    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)

        self.tasks = []
        self.robot_frame_names = []  # Robot body name corresponding to each task

        # Apply offsets to the configured joint limits: raise the lower bound
        self._apply_joints_limit_offset()
        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        # VELOCITY_LIMITS = {k: 3*np.pi for k in self.robot_motor_names.keys()}
        # self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS)) 

        for keypoint_name, entry in self.ik_match_table.items():
            robot_frame, pos_weight, rot_weight = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=robot_frame,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                # Use the keypoint name as the key for indexing keypoint data in update_targets
                self.human_body_to_task[keypoint_name] = task
                self.body_name_to_source_keypoint[robot_frame] = keypoint_name

                self.tasks.append(task)
                self.robot_frame_names.append(robot_frame)
                self.task_errors[task] = []
        pass

    def setup_contact_targets(self):
        if not self.contact_body_names or not self.contact_state_name_to_idx:
            return

        if len(self.contact_body_names) != len(self.contact_names):
            if self.verbose:
                print(
                    "[contact target] skip: robot contact_links count "
                    f"{len(self.contact_body_names)} != keypoint contact_names count {len(self.contact_names)}"
                )
            return

        for body_name, contact_name in zip(self.contact_body_names, self.contact_names):
            contact_state_idx = self.contact_state_name_to_idx.get(contact_name)
            if contact_state_idx is None:
                continue

            source_keypoint_name = self._resolve_body_source_keypoint(body_name)
            if source_keypoint_name is None:
                if self.verbose:
                    print(f"[contact target] could not find a source keypoint for {body_name}")
                continue

            task = self._ensure_contact_task(body_name)
            self.contact_targets.append(
                {
                    "body_name": body_name,
                    "contact_name": contact_name,
                    "contact_state_idx": contact_state_idx,
                    "source_keypoint_name": source_keypoint_name,
                    "mean_positions": self._build_contact_mean_positions(
                        contact_state_idx=contact_state_idx,
                        source_keypoint_name=source_keypoint_name,
                    ),
                    "task": task,
                }
            )

        if self.verbose and self.contact_targets:
            summary = [
                f"{item['contact_name']}->{item['body_name']} (src={item['source_keypoint_name']})"
                for item in self.contact_targets
            ]
            print(f"[contact target] enabled: {summary}")

    def _resolve_body_source_keypoint(self, body_name):
        if body_name in self.keypoint_name_to_idx:
            return body_name
        return self.body_name_to_source_keypoint.get(body_name)

    def _ensure_contact_task(self, body_name):
        existing_task = self.body_name_to_contact_task.get(body_name)
        if existing_task is not None:
            return existing_task

        task = mink.FrameTask(
            frame_name=body_name,
            frame_type="body",
            position_cost=self.contact_position_cost,
            orientation_cost=0.0,
            lm_damping=1,
        )
        self.body_name_to_contact_task[body_name] = task
        self.tasks.append(task)
        self.robot_frame_names.append(body_name)
        self.task_errors[task] = []
        return task

    def _get_target_pose(self, frame_idx, keypoint_name):
        keypoint_idx = self.keypoint_name_to_idx[keypoint_name]
        target_pos = self.keypoints_pos[frame_idx, keypoint_idx]
        target_quat = self.keypoints_quat[frame_idx, keypoint_idx]
        return target_pos, target_quat

    def _get_body_pose(self, body_name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body not found: {body_name}")
        return self.configuration.data.xpos[body_id], self.configuration.data.xquat[body_id]

    def _build_contact_mean_positions(self, contact_state_idx, source_keypoint_name):
        mean_positions = np.zeros((self.num_frames, 3), dtype=np.float64)
        contact_states = self.contact_seq[:, contact_state_idx].astype(bool)
        if not np.any(contact_states):
            return mean_positions

        source_keypoint_idx = self.keypoint_name_to_idx[source_keypoint_name]
        source_positions = np.asarray(
            self.keypoints_pos[:, source_keypoint_idx, :], dtype=np.float64
        )

        frame_idx = 0
        while frame_idx < self.num_frames:
            if not contact_states[frame_idx]:
                frame_idx += 1
                continue

            interval_end = frame_idx + 1
            while interval_end < self.num_frames and contact_states[interval_end]:
                interval_end += 1

            interval_mean = np.mean(source_positions[frame_idx:interval_end], axis=0)
            mean_positions[frame_idx:interval_end] = interval_mean
            frame_idx = interval_end

        return mean_positions

    def _get_contact_locked_position(self, frame_idx, contact_target):
        contact_state_idx = contact_target["contact_state_idx"]
        source_keypoint_name = contact_target["source_keypoint_name"]

        target_pos, _ = self._get_target_pose(frame_idx, source_keypoint_name)
        in_contact = bool(self.contact_seq[frame_idx, contact_state_idx])
        if in_contact:
            return contact_target["mean_positions"][frame_idx], True
        return target_pos, False
    
    def _apply_joints_limit_offset(self):
        """Add offsets to the lower/upper limits of matching joints per joints_limit_offset_degrees.

        The key is a joint-name substring (e.g. knee_joint / elbow_joint) and matches all joints
        in the model whose name contains that substring (e.g. left_knee_joint, right_knee_joint).
        The value is [lower_offset_deg, upper_offset_deg]: the lower offset is added to the lower
        bound and the upper offset to the upper bound (positive raises, negative lowers). A single
        scalar value is also supported (applied to the lower bound only).
        """
        if not self.joints_limit_offset_degrees:
            return

        # Collect all joint names in the model
        all_joint_names = []
        for joint_id in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name is not None:
                all_joint_names.append((joint_id, name))

        for key, offset_deg in self.joints_limit_offset_degrees.items():
            # Support [lower_offset, upper_offset] or a single scalar (lower bound only)
            if isinstance(offset_deg, (list, tuple)):
                if len(offset_deg) != 2:
                    raise ValueError(
                        f"joints_limit_offset_degrees['{key}'] must be two values [lower, upper], "
                        f"got {offset_deg}"
                    )
                lower_offset_rad = np.radians(float(offset_deg[0]))
                upper_offset_rad = np.radians(float(offset_deg[1]))
            else:
                lower_offset_rad = np.radians(float(offset_deg))
                upper_offset_rad = 0.0

            if lower_offset_rad == 0.0 and upper_offset_rad == 0.0:
                continue
            matched = [(jid, name) for jid, name in all_joint_names if key in name]
            if not matched:
                if self.verbose:
                    print(f"[joint limit] no matching joint found: {key}")
                continue
            for joint_id, joint_name in matched:
                lower, upper = self.model.jnt_range[joint_id]
                new_lower = lower + lower_offset_rad
                new_upper = upper + upper_offset_rad
                # Ensure the lower bound does not exceed the upper bound
                if new_lower > new_upper:
                    new_lower, new_upper = new_upper, new_lower
                self.model.jnt_range[joint_id, 0] = new_lower
                self.model.jnt_range[joint_id, 1] = new_upper
                if self.verbose:
                    print(
                        f"[joint limit] {joint_name} range: "
                        f"[{lower:.4f}, {upper:.4f}] -> [{new_lower:.4f}, {new_upper:.4f}] rad"
                    )
 
    def update_targets(self, frame_idx):
        # Record the keypoint targets (pos, quat_wxyz) mapped to tasks for the current frame, for visualization
        self.current_targets = []
        self.current_contact_points = []
        for keypoint_name, task in self.human_body_to_task.items():
            target_pos, target_quat = self._get_target_pose(frame_idx, keypoint_name)
            task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(target_quat), target_pos))
            robot_frame, pos_weight, rot_weight = self.ik_match_table[keypoint_name]
            self.current_targets.append(
                {
                    "pos": np.asarray(target_pos, dtype=np.float64),
                    "quat": np.asarray(target_quat, dtype=np.float64),
                    "robot_frame": robot_frame,
                    "pos_weight": float(pos_weight),
                    "rot_weight": float(rot_weight),
                }
            )

        for contact_target in self.contact_targets:
            source_keypoint_name = contact_target["source_keypoint_name"]
            target_pos, is_active = self._get_contact_locked_position(frame_idx, contact_target)
            if is_active:
                _, target_quat = self._get_target_pose(frame_idx, source_keypoint_name)
            else:
                target_pos, target_quat = self._get_body_pose(contact_target["body_name"])
            contact_target["task"].set_target(
                mink.SE3.from_rotation_and_translation(mink.SO3(target_quat), target_pos)
            )
            if is_active:
                self.current_contact_points.append(np.asarray(target_pos, dtype=np.float64))
    
    def error(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks]
            )
        )

    def _solve_current_targets(self):
        """Solve the targets currently installed on the IK tasks.

        Keep this convergence rule identical for warm-up and exported frames so
        initialization cannot silently use a more permissive solver than the
        actual retargeting pass.
        """
        curr_error = self.error()
        dt = self.configuration.model.opt.timestep
        vel = mink.solve_ik(
            self.configuration,
            self.tasks,
            dt,
            self.solver,
            self.damping,
            limits=self.ik_limits,
        )
        self.configuration.integrate_inplace(vel, dt)
        next_error = self.error()
        num_iter = 0
        while curr_error - next_error > 0.001 and num_iter < self.max_iter:
            curr_error = next_error
            vel = mink.solve_ik(
                self.configuration,
                self.tasks,
                dt,
                self.solver,
                self.damping,
                limits=self.ik_limits,
            )
            self.configuration.integrate_inplace(vel, dt)
            next_error = self.error()
            num_iter += 1
        return next_error

    def _warm_start_initial_pose(self):
        """Initialize frame zero from a short bidirectional target sweep.

        Solving the first target directly from the MJCF default pose can select
        a poor local IK branch.  A short forward/backward sweep supplies nearby
        temporal context, then returns the configuration to frame zero before
        any output is recorded.  This avoids blending or modifying the source
        motion itself.
        """
        last_frame = min(self.initialization_sweep_frames, self.num_frames - 1)
        if last_frame <= 0:
            return

        for frame_idx in range(last_frame + 1):
            self.update_targets(frame_idx)
            self._solve_current_targets()
        for frame_idx in range(last_frame - 1, -1, -1):
            self.update_targets(frame_idx)
            self._solve_current_targets()

        if self.verbose:
            print(
                "[IK initialization] bidirectional sweep complete: "
                f"frames 0..{last_frame}..0"
            )
    
    def _draw_pose(self, scene, pos, quat_wxyz, point_rgba, axis_alpha=1.0,
                   point_radius=0.035, axis_radius=0.005, axis_length=0.08):
        """Draw a pose in the scene: one small sphere + red/green/blue (XYZ) axes. Returns whether it succeeded."""
        max_geoms = len(scene.geoms)
        if scene.ngeom + 4 > max_geoms:
            return False

        pos = np.asarray(pos, dtype=np.float64)
        identity_mat = np.eye(3, dtype=np.float64).reshape(-1)
        axis_colors = (
            np.array([1.0, 0.0, 0.0, axis_alpha], dtype=np.float32),  # X red
            np.array([0.0, 1.0, 0.0, axis_alpha], dtype=np.float32),  # Y green
            np.array([0.0, 0.0, 1.0, axis_alpha], dtype=np.float32),  # Z blue
        )

        # Position sphere
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([point_radius, point_radius, point_radius], dtype=np.float64),
            pos,
            identity_mat,
            np.asarray(point_rgba, dtype=np.float32),
        )
        scene.ngeom += 1

        # Derive the rotation matrix from the quaternion (wxyz); its columns are the axis directions
        rot_mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rot_mat, np.asarray(quat_wxyz, dtype=np.float64))
        rot_mat = rot_mat.reshape(3, 3)
        # A cylinder defaults along its local z axis, so rotate each axis to its target direction
        base_to_axis = (
            Rot.from_euler("y", 90, degrees=True).as_matrix(),   # z->x
            Rot.from_euler("x", -90, degrees=True).as_matrix(),  # z->y
            np.eye(3),                                           # z->z
        )
        for axis_idx in range(3):
            axis_world_rot = rot_mat @ base_to_axis[axis_idx]
            axis_dir = rot_mat[:, axis_idx]
            center = pos + axis_dir * (0.5 * axis_length)
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_CYLINDER,
                np.array([axis_radius, 0.5 * axis_length, 0.0], dtype=np.float64),
                center,
                axis_world_rot.reshape(-1),
                axis_colors[axis_idx],
            )
            scene.ngeom += 1
        return True

    def _draw_point(self, scene, pos, point_rgba, point_radius=0.04):
        """Draw only a single spherical point in the scene."""
        max_geoms = len(scene.geoms)
        if scene.ngeom + 1 > max_geoms:
            return False

        pos = np.asarray(pos, dtype=np.float64)
        identity_mat = np.eye(3, dtype=np.float64).reshape(-1)
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([point_radius, point_radius, point_radius], dtype=np.float64),
            pos,
            identity_mat,
            np.asarray(point_rgba, dtype=np.float32),
        )
        scene.ngeom += 1
        return True

    def draw_target_keypoints(self, viewer, point_radius=0.035, axis_radius=0.005, axis_length=0.08, targets=None):
        """Draw the positions and axes of the target keypoints and the corresponding robot bodies.

        - Target keypoint: yellow sphere + full-brightness axes.
        - Robot body: cyan sphere + semi-transparent axes.
        """
        if targets is None:
            targets = getattr(self, "current_targets", None)
        scene = viewer.user_scn
        scene.ngeom = 0

        target_rgba = np.array([1.0, 1.0, 0.0, 0.9], dtype=np.float32)   # yellow
        robot_rgba = np.array([0.0, 1.0, 1.0, 0.9], dtype=np.float32)    # cyan
        contact_rgba = np.array([1.0, 0.0, 0.0, 0.95], dtype=np.float32) # red

        # 1) Target keypoints
        if targets:
            for target in targets:
                if isinstance(target, dict):
                    target_pos = target["pos"]
                    target_quat_wxyz = target["quat"]
                    robot_frame = target.get("robot_frame")
                    pos_weight = target.get("pos_weight", 1.0)
                    rot_weight = target.get("rot_weight", 1.0)
                else:
                    # Backward compatibility with the old (pos, quat) tuple format
                    target_pos, target_quat_wxyz = target
                    robot_frame = None
                    pos_weight = 1.0
                    rot_weight = 1.0

                # Zero position weight: draw the sphere at the corresponding body's position (overlapping)
                if pos_weight == 0 and robot_frame is not None:
                    body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, robot_frame)
                    if body_id >= 0:
                        target_pos = self.configuration.data.xpos[body_id]

                if rot_weight == 0:
                    # Zero orientation weight: show only the sphere, no axes
                    if not self._draw_point(
                        scene, target_pos, target_rgba,
                        point_radius=point_radius,
                    ):
                        break
                else:
                    if not self._draw_pose(
                        scene, target_pos, target_quat_wxyz, target_rgba,
                        axis_alpha=1.0,
                        point_radius=point_radius, axis_radius=axis_radius, axis_length=axis_length,
                    ):
                        break

        # 2) Current position and orientation of the corresponding robot bodies
        data = self.configuration.data
        for robot_frame in self.robot_frame_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, robot_frame)
            if body_id < 0:
                continue
            body_pos = data.xpos[body_id]
            body_quat = data.xquat[body_id]  # wxyz
            if not self._draw_pose(
                scene, body_pos, body_quat, robot_rgba,
                axis_alpha=0.5,
                point_radius=point_radius, axis_radius=axis_radius, axis_length=axis_length,
            ):
                break

        # 3) Currently active contact points, drawn as red spheres
        for contact_pos in self.current_contact_points:
            if not self._draw_point(
                scene,
                contact_pos,
                contact_rgba,
                point_radius=max(point_radius * 1.5, 0.02),
            ):
                break
    
    def retarget(self):
        global _PAUSED
        viewer = None
        _PAUSED = False

        self._warm_start_initial_pose()

        if self.render_debug:
            viewer = mujoco.viewer.launch_passive(
                self.model, self.configuration.data, key_callback=key_callback
            )
            print("Controls: Space play/pause")

        try:
            for frame_idx in tqdm(range(self.num_frames), desc="Retargeting", unit="frame"):
                start_time = time.time()
                self.update_targets(frame_idx)
                self._solve_current_targets()
                curr_pos = self.configuration.data.qpos.copy()
                self.result_pos.append(curr_pos)

                if viewer is not None:
                    if not viewer.is_running():
                        break
                    mujoco.mj_forward(self.model, self.configuration.data)
                    self.draw_target_keypoints(viewer)
                    viewer.sync()
                    # Stay on the current frame while paused, until unpaused or the window is closed
                    while _PAUSED and viewer.is_running():
                        viewer.sync()
                        time.sleep(0.02)
                    if not viewer.is_running():
                        break
                    end_time = time.time()
                    elapsed = end_time - start_time
                    time.sleep(max(0, self.time_step - elapsed))
                    # time.sleep(0.02)  # reduce CPU usage
                    
        finally:
            if viewer is not None:
                viewer.close()
    
    def _actuated_joint_qpos(self):
        """Return actuator-order scalar joint names and qpos column indices."""
        joint_names = []
        qpos_indices = []
        seen_joint_ids = set()
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id < 0 or joint_id in seen_joint_ids:
                continue
            seen_joint_ids.add(joint_id)
            joint_type = int(self.model.jnt_type[joint_id])
            if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                continue
            joint_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )
            joint_names.append(joint_name or f"joint_{joint_id}")
            qpos_indices.append(int(self.model.jnt_qposadr[joint_id]))
        return joint_names, np.asarray(qpos_indices, dtype=np.int64)

    @staticmethod
    def _slerp_wxyz(quat_a, quat_b, blend):
        """Shortest-path interpolation between two WXYZ unit quaternions."""
        quat_a = np.asarray(quat_a, dtype=np.float64)
        quat_b = np.asarray(quat_b, dtype=np.float64)
        quat_a = quat_a / np.linalg.norm(quat_a)
        quat_b = quat_b / np.linalg.norm(quat_b)
        dot = float(np.dot(quat_a, quat_b))
        if dot < 0.0:
            quat_b = -quat_b
            dot = -dot
        dot = np.clip(dot, -1.0, 1.0)
        if dot > 0.9995:
            result = (1.0 - blend) * quat_a + blend * quat_b
            return result / np.linalg.norm(result)
        angle = np.arccos(dot)
        sin_angle = np.sin(angle)
        return (
            np.sin((1.0 - blend) * angle) / sin_angle * quat_a
            + np.sin(blend * angle) / sin_angle * quat_b
        )

    def _interpolate_qpos(self, qpos_a, qpos_b, blend):
        """Interpolate a MuJoCo qpos without linearly blending quaternions."""
        result = (1.0 - blend) * qpos_a + blend * qpos_b
        for joint_id in range(self.model.njnt):
            joint_type = int(self.model.jnt_type[joint_id])
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                quat_slice = slice(qpos_adr + 3, qpos_adr + 7)
            elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                quat_slice = slice(qpos_adr, qpos_adr + 4)
            else:
                continue
            result[quat_slice] = self._slerp_wxyz(
                qpos_a[quat_slice], qpos_b[quat_slice], blend
            )
        return result

    def _subdivide_qpos_intervals(self, result, subdivisions):
        """Subdivide selected qpos intervals while preserving every keyframe."""
        subdivisions = np.asarray(subdivisions, dtype=np.int64)
        expected_shape = (max(result.shape[0] - 1, 0),)
        if subdivisions.shape != expected_shape:
            raise ValueError(
                f"subdivisions must have shape {expected_shape}, got {subdivisions.shape}"
            )
        if np.any(subdivisions < 1):
            raise ValueError("Every interval subdivision count must be at least one")

        inserted_frames = int(np.sum(subdivisions - 1))
        if inserted_frames == 0:
            return result

        resampled = [result[0].copy()]
        for frame_idx, subdivision_count in enumerate(subdivisions):
            for subdivision_idx in range(1, int(subdivision_count) + 1):
                if subdivision_idx == subdivision_count:
                    # Preserve every original keyframe bit-for-bit.  Besides
                    # avoiding needless roundoff, this keeps the source's
                    # chosen quaternion sign at interval boundaries.
                    resampled.append(result[frame_idx + 1].copy())
                    continue
                blend = subdivision_idx / float(subdivision_count)
                resampled.append(
                    self._interpolate_qpos(
                        result[frame_idx], result[frame_idx + 1], blend
                    )
                )
        return np.asarray(resampled, dtype=np.float64)

    def _resample_to_joint_velocity_limit(self, result, max_joint_velocity):
        """Locally time-scale qpos intervals that exceed a joint speed limit.

        Each original interval is subdivided just enough to satisfy the limit
        at the unchanged motion fps.  This preserves every original keyframe
        and the geometric path, unlike position clipping, while extending only
        the portions of the motion that are physically too fast.
        """
        _, qpos_indices = self._actuated_joint_qpos()
        if result.shape[0] < 2 or qpos_indices.size == 0:
            return result

        interval_peak_velocity = (
            np.max(np.abs(np.diff(result[:, qpos_indices], axis=0)), axis=1)
            * float(self.fps)
        )
        subdivisions = np.maximum(
            1,
            np.ceil(interval_peak_velocity / max_joint_velocity - 1.0e-12).astype(np.int64),
        )
        inserted_frames = int(np.sum(subdivisions - 1))
        if inserted_frames == 0:
            return result
        resampled = self._subdivide_qpos_intervals(result, subdivisions)
        if self.verbose:
            original_duration = (result.shape[0] - 1) / float(self.fps)
            resampled_duration = (resampled.shape[0] - 1) / float(self.fps)
            print(
                "[motion time scaling] "
                f"inserted {inserted_frames} frames to enforce "
                f"{max_joint_velocity:.4f} rad/s; duration "
                f"{original_duration:.3f}s -> {resampled_duration:.3f}s"
            )
        return resampled

    @staticmethod
    def _local_velocity_ratio_profile(velocity_norm, window, min_speed):
        """Return per-interval local spike ratios and neighbor medians."""
        velocity_norm = np.asarray(velocity_norm, dtype=np.float64)
        ratios = np.zeros_like(velocity_norm)
        medians = np.zeros_like(velocity_norm)
        for interval_idx, speed in enumerate(velocity_norm):
            if speed < min_speed:
                continue
            begin = max(0, interval_idx - window)
            end = min(velocity_norm.size, interval_idx + window + 1)
            neighbors = np.concatenate(
                (
                    velocity_norm[begin:interval_idx],
                    velocity_norm[interval_idx + 1 : end],
                )
            )
            if neighbors.size == 0:
                continue
            local_median = float(np.median(neighbors))
            medians[interval_idx] = local_median
            ratios[interval_idx] = float(speed / max(local_median, 1.0e-6))
        return ratios, medians

    def _resample_to_local_velocity_ratio(
        self,
        result,
        max_local_ratio,
        window,
        min_speed,
        max_passes=8,
    ):
        """Locally time-scale isolated joint-velocity spikes until they validate.

        The discontinuity metric uses the L2 norm across actuated joint
        velocities.  An offending interval is subdivided just enough to fall
        below either the configured local-ratio ceiling or the validator's
        minimum-speed gate.  Recomputing the profile after each pass makes the
        operation robust to the changed local neighborhood without smoothing
        or dropping any source keyframe.
        """
        _, qpos_indices = self._actuated_joint_qpos()
        if result.shape[0] < 3 or qpos_indices.size == 0:
            return result

        original = result
        resampled = result
        total_inserted = 0
        worst_ratio_before = 0.0
        passes = 0
        for pass_idx in range(max_passes):
            joint_vel = (
                np.diff(resampled[:, qpos_indices], axis=0) * float(self.fps)
            )
            velocity_norm = np.linalg.norm(joint_vel, axis=1)
            ratios, medians = self._local_velocity_ratio_profile(
                velocity_norm, window, min_speed
            )
            worst_ratio = float(np.max(ratios)) if ratios.size else 0.0
            if pass_idx == 0:
                worst_ratio_before = worst_ratio
            violating = np.flatnonzero(ratios > max_local_ratio + 1.0e-12)
            if violating.size == 0:
                break

            subdivisions = np.ones(velocity_norm.shape, dtype=np.int64)
            for interval_idx in violating:
                speed = float(velocity_norm[interval_idx])
                local_ratio_limit = max_local_ratio * max(
                    float(medians[interval_idx]), 1.0e-6
                )
                # If the neighborhood is nearly stationary, crossing just
                # below min_speed makes the exact same gate used by validation
                # stop classifying the slowed interval as a discontinuity.
                if min_speed > 0.0:
                    local_ratio_limit = max(
                        local_ratio_limit,
                        min_speed * (1.0 - 1.0e-6),
                    )
                if local_ratio_limit <= 0.0 or not np.isfinite(local_ratio_limit):
                    raise ValueError(
                        "Cannot time-scale a local velocity spike with a non-positive target speed"
                    )
                subdivision_count = int(
                    np.ceil(speed / local_ratio_limit - 1.0e-12)
                )
                # A ratio violation always needs a real time extension, even
                # when floating-point rounding produces ceil(...) == 1.
                subdivisions[interval_idx] = max(2, subdivision_count)

            inserted = int(np.sum(subdivisions - 1))
            resampled = self._subdivide_qpos_intervals(resampled, subdivisions)
            total_inserted += inserted
            passes = pass_idx + 1
        else:
            joint_vel = (
                np.diff(resampled[:, qpos_indices], axis=0) * float(self.fps)
            )
            ratios, _ = self._local_velocity_ratio_profile(
                np.linalg.norm(joint_vel, axis=1), window, min_speed
            )
            worst_ratio = float(np.max(ratios)) if ratios.size else 0.0
            raise ValueError(
                "Local velocity time scaling did not converge after "
                f"{max_passes} passes (ratio={worst_ratio:.3f}, "
                f"limit={max_local_ratio:.3f})"
            )

        if total_inserted and self.verbose:
            final_joint_vel = (
                np.diff(resampled[:, qpos_indices], axis=0) * float(self.fps)
            )
            final_ratios, _ = self._local_velocity_ratio_profile(
                np.linalg.norm(final_joint_vel, axis=1), window, min_speed
            )
            worst_ratio_after = (
                float(np.max(final_ratios)) if final_ratios.size else 0.0
            )
            original_duration = (original.shape[0] - 1) / float(self.fps)
            resampled_duration = (resampled.shape[0] - 1) / float(self.fps)
            print(
                "[local continuity time scaling] "
                f"inserted {total_inserted} frames in {passes} pass(es); "
                f"local ratio {worst_ratio_before:.3f}->{worst_ratio_after:.3f}; "
                f"duration {original_duration:.3f}s->{resampled_duration:.3f}s"
            )
        return resampled

    def _postprocess_result_motion(self, result):
        self._validate_qpos_values(result)
        max_joint_velocity = self.motion_validation.get("max_joint_velocity_rad_s")
        velocity_limit_mode = self.motion_validation.get("velocity_limit_mode", "reject")
        if velocity_limit_mode not in ("reject", "resample"):
            raise ValueError("velocity_limit_mode must be 'reject' or 'resample'")
        if velocity_limit_mode == "reject":
            return result

        if max_joint_velocity is not None:
            max_joint_velocity = float(max_joint_velocity)
            if not np.isfinite(max_joint_velocity) or max_joint_velocity <= 0:
                raise ValueError("max_joint_velocity_rad_s must be finite and positive")
            result = self._resample_to_joint_velocity_limit(
                result, max_joint_velocity
            )

        max_local_ratio = self.motion_validation.get("max_local_velocity_ratio")
        if max_local_ratio is not None:
            max_local_ratio = float(max_local_ratio)
            window = int(
                self.motion_validation.get("local_velocity_window_frames", 6)
            )
            min_speed = float(
                self.motion_validation.get(
                    "min_discontinuity_velocity_rad_s", 5.0
                )
            )
            if not np.isfinite(max_local_ratio) or max_local_ratio <= 1.0:
                raise ValueError(
                    "max_local_velocity_ratio must be finite and greater than 1"
                )
            if window < 1:
                raise ValueError("local_velocity_window_frames must be at least 1")
            if not np.isfinite(min_speed) or min_speed < 0:
                raise ValueError(
                    "min_discontinuity_velocity_rad_s must be finite and non-negative"
                )
            result = self._resample_to_local_velocity_ratio(
                result,
                max_local_ratio=max_local_ratio,
                window=window,
                min_speed=min_speed,
            )
        return result

    def _validate_qpos_values(self, result):
        """Validate qpos values needed before any quaternion interpolation."""
        if not np.isfinite(result).all():
            raise ValueError("Retarget result contains non-finite qpos values")
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError(f"Motion fps must be finite and positive, got {self.fps}")

        for joint_id in range(self.model.njnt):
            joint_type = int(self.model.jnt_type[joint_id])
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                quat_slice = slice(qpos_adr + 3, qpos_adr + 7)
            elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                quat_slice = slice(qpos_adr, qpos_adr + 4)
            else:
                continue
            quat_norm = np.linalg.norm(result[:, quat_slice], axis=1)
            max_quat_error = float(np.max(np.abs(quat_norm - 1.0)))
            if max_quat_error > 1.0e-4:
                raise ValueError(
                    f"Joint quaternion norm error {max_quat_error:.6g} exceeds 1e-4"
                )

    def _validate_result_motion(self, result):
        """Validate numeric and temporal continuity before publishing a CSV."""
        self._validate_qpos_values(result)

        joint_names, qpos_indices = self._actuated_joint_qpos()
        if result.shape[0] < 2 or qpos_indices.size == 0:
            return
        joint_vel = np.diff(result[:, qpos_indices], axis=0) * float(self.fps)
        abs_joint_vel = np.abs(joint_vel)

        max_joint_velocity = self.motion_validation.get("max_joint_velocity_rad_s")
        if max_joint_velocity is not None:
            max_joint_velocity = float(max_joint_velocity)
            if not np.isfinite(max_joint_velocity) or max_joint_velocity <= 0:
                raise ValueError("max_joint_velocity_rad_s must be finite and positive")
            peak_flat_idx = int(np.argmax(abs_joint_vel))
            interval_idx, joint_idx = np.unravel_index(peak_flat_idx, abs_joint_vel.shape)
            peak = float(abs_joint_vel[interval_idx, joint_idx])
            if peak > max_joint_velocity + 1.0e-8:
                raise ValueError(
                    "Retarget result exceeds max joint velocity: "
                    f"{joint_names[joint_idx]}={peak:.4f} rad/s at frame interval "
                    f"{interval_idx}->{interval_idx + 1}, limit={max_joint_velocity:.4f} rad/s"
                )

        max_local_ratio = self.motion_validation.get("max_local_velocity_ratio")
        if max_local_ratio is not None and joint_vel.shape[0] >= 2:
            max_local_ratio = float(max_local_ratio)
            window = int(self.motion_validation.get("local_velocity_window_frames", 6))
            min_speed = float(
                self.motion_validation.get("min_discontinuity_velocity_rad_s", 5.0)
            )
            if not np.isfinite(max_local_ratio) or max_local_ratio <= 1.0:
                raise ValueError("max_local_velocity_ratio must be finite and greater than 1")
            if window < 1:
                raise ValueError("local_velocity_window_frames must be at least 1")
            if not np.isfinite(min_speed) or min_speed < 0:
                raise ValueError(
                    "min_discontinuity_velocity_rad_s must be finite and non-negative"
                )

            velocity_norm = np.linalg.norm(joint_vel, axis=1)
            ratios, _ = self._local_velocity_ratio_profile(
                velocity_norm, window, min_speed
            )
            worst_interval = int(np.argmax(ratios)) if ratios.size else -1
            worst_ratio = (
                float(ratios[worst_interval]) if worst_interval >= 0 else 0.0
            )
            if worst_ratio > max_local_ratio:
                raise ValueError(
                    "Retarget result contains a temporal joint-velocity discontinuity: "
                    f"frame interval {worst_interval}->{worst_interval + 1}, "
                    f"local ratio={worst_ratio:.3f}, limit={max_local_ratio:.3f}"
                )

        if self.verbose:
            peak = float(np.max(abs_joint_vel))
            first_l2 = float(np.linalg.norm(joint_vel[0]))
            first_peak = float(np.max(abs_joint_vel[0]))
            print(
                "[motion validation] "
                f"peak={peak:.4f} rad/s, first-interval L2={first_l2:.4f} rad/s, "
                f"first-interval peak={first_peak:.4f} rad/s"
            )

    def save_results_as_csv(self, output_path):
        if len(self.result_pos) == 0:
            raise ValueError("No retarget results to save. Run retarget() first.")

        # [num_frame, num_joint], the first 7 dims are posXYZ + quat(wxyz)
        result = np.asarray(self.result_pos, dtype=np.float64)
        if result.ndim != 2 or result.shape[1] < 7:
            raise ValueError(
                f"result_pos shape must be [num_frame, num_joint>=7], got {result.shape}"
            )
        result = self._postprocess_result_motion(result)
        self._validate_result_motion(result)

        # Reorder the first-7-column quaternion from wxyz (cols 3..6) to xyzw
        result_xyzw = result.copy()
        result_xyzw[:, 3:7] = result[:, [4, 5, 6, 3]]

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(result_xyzw.tolist())

        if self.verbose:
            print(f"Saved retarget results to: {output_path} (shape={result_xyzw.shape})")



if  __name__ == "__main__":
    args = parse_args()
    workspace_root = os.path.abspath(os.path.join(SCRIPT_DIR,"..", "robot", ".."))
    config_path = os.path.expanduser(args.config)
    if not os.path.isabs(config_path):
        config_path = os.path.join(workspace_root, config_path)

    print("mujoco version: ", mujoco.__version__)
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    ik_match_table = config.get("ik_match_table", {})
    robot_xml_path = config.get("robot_xml_path", "")
    keypoints_path = config.get("keypoints_path", "")
    keypoints_idx = config.get("keypoints_idx","")

    config_name = os.path.splitext(os.path.basename(config_path))[0]
    if args.keypoints_name:
        keypoints_path = os.path.join(
            "output_data",
            "keypoints",
            config_name,
            f"{args.keypoints_name}_keypoints.pkl",
        )

    if robot_xml_path and not os.path.isabs(robot_xml_path):
        robot_xml_path = os.path.join(workspace_root, robot_xml_path)
    if keypoints_path and not os.path.isabs(keypoints_path):
        keypoints_path = os.path.join(workspace_root, keypoints_path)

    verbose_mode = config.get("verbose", False)
    config_render_debug = bool(config.get("render_debug", False))
    render_debug = config_render_debug if args.render_debug is None else bool(args.render_debug)
    joints_limit_offset_degrees = config.get("joints_limit_offset_degrees", {})
    contact_body_names = config.get("contact_links")
    if contact_body_names is None:
        contact_body_names = {
            field_name: config.get(field_name, [])
            for field_name in RobotRetarget.LEGACY_CONTACT_CONFIG_TO_NAMES
        }
    contact_position_cost = config.get("contact_pos_fixed_factor", 10.0)
    initialization_sweep_frames = config.get("initialization_sweep_frames", 0)
    motion_validation = config.get("motion_validation", {})
    
    robot_retarget = RobotRetarget(
        model_path=robot_xml_path,
        keypoint_path=keypoints_path,
        ik_match_table=ik_match_table,
        solver="daqp",
        verbose=verbose_mode,
        render_debug=render_debug,
        joints_limit_offset_degrees=joints_limit_offset_degrees,
        contact_body_names=contact_body_names,
        contact_position_cost=contact_position_cost,
        initialization_sweep_frames=initialization_sweep_frames,
        motion_validation=motion_validation,
    )

    robot_retarget.retarget()

    keypoint_stem = os.path.splitext(os.path.basename(keypoints_path))[0]
    if keypoint_stem.endswith("_keypoints"):
        keypoint_stem = keypoint_stem[: -len("_keypoints")]
    output_csv = os.path.join(
        workspace_root,
        "output_data/robot_motion",
        f"{keypoint_stem}_{config_name}.csv",
    )
    robot_retarget.save_results_as_csv(output_csv)

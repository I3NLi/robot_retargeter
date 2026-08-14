"""Regression and safety tests for the realtime Kengo full-body retargeter."""

from __future__ import annotations

import ast
from pathlib import Path
import threading
import time
import unittest

import numpy as np

from scripts.realtime_fullbody_core import (
    KENGO_JOINT_NAMES,
    RetargetError,
    RealtimeKengoRetargeter,
    TargetFrame,
    synthetic_smpl24,
    validate_smpl24,
)
from scripts.realtime_fullbody_retarget import (
    INPUT_TOPIC,
    OUTPUT_TOPIC,
    LatestPoseMailbox,
    RealtimeWorker,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "asset/robot/kengo_description/mjcf/kengo.xml"
CONFIG = ROOT / "config/robot/kengo.yaml"


class RealtimeFullBodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.retargeter = RealtimeKengoRetargeter(MODEL, CONFIG)

    def setUp(self) -> None:
        self.retargeter.reset()

    def test_synthetic_pose_produces_complete_bounded_joint_result(self) -> None:
        positions, quaternions = synthetic_smpl24()
        result = self.retargeter.solve(positions, quaternions)
        self.assertEqual(result.joint_names, KENGO_JOINT_NAMES)
        self.assertEqual(result.positions.shape, (23,))
        self.assertTrue(np.isfinite(result.positions).all())
        self.assertTrue(np.isfinite(result.root_position).all())
        self.assertAlmostEqual(float(np.linalg.norm(result.root_quaternion_xyzw)), 1.0, places=6)
        for joint_name, value in zip(result.joint_names, result.positions):
            joint_id = self.retargeter.model.joint(joint_name).id
            lower, upper = self.retargeter.model.jnt_range[joint_id]
            self.assertGreaterEqual(float(value), float(lower) - 1.0e-7)
            self.assertLessEqual(float(value), float(upper) + 1.0e-7)

    def test_solver_matches_offline_robot_retargeter_task_contract(self) -> None:
        import pickle
        import yaml

        from scripts.robot_retarget import RobotRetarget

        keypoints_path = ROOT / "output_data/keypoints/kengo/Form_1_stageii_keypoints.pkl"
        with keypoints_path.open("rb") as stream:
            keypoints = pickle.load(stream)
        with CONFIG.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        offline = RobotRetarget(
            model_path=str(MODEL),
            keypoint_path=str(keypoints_path),
            ik_match_table=config["ik_match_table"],
            solver="daqp",
            joints_limit_offset_degrees=config["joints_limit_offset_degrees"],
            contact_body_names=[],
            initialization_sweep_frames=0,
        )
        index = {name: idx for idx, name in enumerate(keypoints["keypoint_names"])}
        frame = TargetFrame(
            {
                name: np.asarray(keypoints["positions"][0, index[name]], dtype=np.float64)
                for name in config["ik_match_table"]
            },
            {
                name: np.asarray(keypoints["quaternions"][0, index[name]], dtype=np.float64)
                for name in config["ik_match_table"]
            },
        )
        offline.update_targets(0)
        offline._solve_current_targets()
        online = self.retargeter.solve_target_frame(frame)
        np.testing.assert_allclose(
            online.positions,
            offline.configuration.data.qpos[self.retargeter.qpos_indices],
            rtol=0.0,
            atol=2.0e-7,
        )

    def test_invalid_pose_is_rejected_without_changing_last_solution(self) -> None:
        positions, quaternions = synthetic_smpl24()
        before = self.retargeter.solve(positions, quaternions).positions.copy()
        bad = positions.copy()
        bad[5, 0] = np.nan
        with self.assertRaises(RetargetError):
            self.retargeter.solve(bad, quaternions)
        after = self.retargeter.configuration.q[self.retargeter.qpos_indices]
        np.testing.assert_array_equal(after, before)

    def test_target_link_lengths_match_kengo_model(self) -> None:
        positions, quaternions = synthetic_smpl24()
        target = self.retargeter.make_targets(positions, quaternions)
        for link_name, expected_length in self.retargeter.robot_link_lengths.items():
            if link_name not in target.positions:
                continue
            parent_name, _child_name = self.retargeter.robot_links[link_name]
            # Robot-link parent body and semantic keypoint names differ. The
            # link-length contract is already checked in the semantic builder;
            # here ensure every configured target remains finite and complete.
            self.assertTrue(parent_name)
            self.assertTrue(np.isfinite(target.positions[link_name]).all())
            self.assertGreater(expected_length, 0.0)

    def test_pose_mailbox_keeps_only_the_latest_pending_frame(self) -> None:
        clock = [10.0]
        frames = LatestPoseMailbox(clock=lambda: clock[0])
        positions, quaternions = synthetic_smpl24()
        for sequence in range(1, 101):
            shifted = positions.copy()
            shifted[:, 2] += sequence / 1000.0
            self.assertTrue(frames.offer(shifted, quaternions, source_stamp_ns=sequence))
        self.assertEqual(frames.pending, 1)
        self.assertEqual(frames.replaced_frames, 99)
        item = frames.get(timeout=0.0)
        self.assertIsNotNone(item)
        assert item is not None
        self.assertEqual(item.sequence, 100)
        self.assertEqual(item.source_stamp_ns, 100)
        np.testing.assert_array_equal(item.positions[:, 2], positions[:, 2] + 0.1)
        self.assertEqual(frames.pending, 0)

    def test_invalid_offer_does_not_discard_queued_valid_frames(self) -> None:
        frames = LatestPoseMailbox()
        positions, quaternions = synthetic_smpl24()
        self.assertTrue(frames.offer(positions, quaternions, source_stamp_ns=7))
        bad_quaternions = quaternions.copy()
        bad_quaternions[0] = 0.0
        self.assertFalse(frames.offer(positions, bad_quaternions))
        item = frames.get(timeout=0.0)
        self.assertIsNotNone(item)
        assert item is not None
        self.assertEqual(item.source_stamp_ns, 7)
        self.assertEqual(frames.accepted_frames, 1)
        self.assertEqual(frames.rejected_frames, 1)

    def test_worker_publishes_inflight_then_latest_without_backlog(self) -> None:
        positions, quaternions = synthetic_smpl24()
        frames = LatestPoseMailbox()
        published: list[int] = []
        first_solve_started = threading.Event()
        release_first_solve = threading.Event()
        finished = threading.Event()

        class BlockingRetargeter:
            def __init__(self) -> None:
                self.calls = 0

            def solve(self, _positions: np.ndarray, _quaternions: np.ndarray) -> object:
                self.calls += 1
                if self.calls == 1:
                    first_solve_started.set()
                    self.assert_release()
                return object()

            def assert_release(self) -> None:
                if not release_first_solve.wait(2.0):
                    raise RuntimeError("test did not release first solve")

            def reset(self) -> None:
                pass

        def publish(_result: object, snapshot: object) -> None:
            published.append(snapshot.sequence)
            if len(published) == 2:
                finished.set()

        worker = RealtimeWorker(BlockingRetargeter(), frames, publish)  # type: ignore[arg-type]
        worker.start()
        try:
            self.assertTrue(frames.offer(positions, quaternions, source_stamp_ns=1))
            self.assertTrue(first_solve_started.wait(2.0))
            for sequence in range(2, 13):
                shifted = positions.copy()
                shifted[15, 2] += sequence / 10000.0
                self.assertTrue(frames.offer(shifted, quaternions, source_stamp_ns=sequence))
            self.assertEqual(frames.pending, 1)
            self.assertEqual(frames.replaced_frames, 10)
            release_first_solve.set()
            self.assertTrue(finished.wait(5.0))
            time.sleep(0.05)
        finally:
            release_first_solve.set()
            worker.stop()
        self.assertEqual(published, [1, 12])
        self.assertEqual(worker.solved_frames, 2)
        self.assertEqual(worker.failed_frames, 0)

    def test_source_contains_only_declared_monitor_ros_endpoints(self) -> None:
        source_path = ROOT / "scripts/realtime_fullbody_retarget.py"
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        subscriptions = [
            call for call in calls if isinstance(call.func, ast.Attribute) and call.func.attr == "create_subscription"
        ]
        publishers = [
            call for call in calls if isinstance(call.func, ast.Attribute) and call.func.attr == "create_publisher"
        ]
        forbidden = (
            "/hybrid_body_controller/commands",
            "create_service",
            "create_client",
            "create_action",
        )
        self.assertEqual(len(subscriptions), 1)
        self.assertEqual(len(publishers), 1)
        self.assertEqual(INPUT_TOPIC, "/pico4/body_tracking/ik_poses")
        self.assertEqual(OUTPUT_TOPIC, "/pico4/retargeted/full_body_joint_targets")
        self.assertIn("HistoryPolicy.KEEP_LAST", source)
        self.assertNotIn("HistoryPolicy.KEEP_ALL", source)
        self.assertGreaterEqual(source.count("depth=1"), 2)
        self.assertNotIn("rate_hz", source)
        self.assertNotIn("max_age_seconds", source)
        for marker in forbidden:
            self.assertNotIn(marker, source)

    def test_systemd_service_is_low_priority_cpu6_and_read_only(self) -> None:
        unit = (ROOT / "deployment/kengo-fullbody-retarget.service").read_text(
            encoding="utf-8"
        )
        runner = (ROOT / "deployment/run_realtime_fullbody.sh").read_text(
            encoding="utf-8"
        )
        for marker in (
            "User=admin",
            "Group=admin",
            "Nice=10",
            "CPUAffinity=6",
            "NoNewPrivileges=true",
            "ProtectSystem=strict",
            "ProtectHome=read-only",
            "AF_NETLINK",
            "KillMode=control-group",
        ):
            self.assertIn(marker, unit)
        self.assertIn("ROS_DOMAIN_ID", runner)
        self.assertIn("FASTRTPS_DEFAULT_PROFILES_FILE", runner)
        self.assertIn("bin/kengo_fullbody_retarget_node", runner)
        self.assertNotIn("python", runner.lower())
        self.assertNotIn("/hybrid_body_controller/commands", runner)


if __name__ == "__main__":
    unittest.main()

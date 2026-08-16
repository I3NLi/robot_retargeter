"""Static and optional binary contracts for the native full-body service."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from scripts.realtime_fullbody_core import RealtimeKengoRetargeter, synthetic_smpl24


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "asset/robot/kengo_description/mjcf/kengo.xml"
CONFIG = ROOT / "config/robot/kengo.yaml"


class NativeFullBodyContractTests(unittest.TestCase):
    def test_cpp_node_is_monitor_only_keep_all_fifo(self) -> None:
        node = (ROOT / "cpp/src/fullbody_retarget_node.cpp").read_text(encoding="utf-8")
        core = (ROOT / "cpp/src/fullbody_retarget_core.cpp").read_text(encoding="utf-8")
        self.assertEqual(node.count("create_subscription<"), 2)
        self.assertEqual(node.count("create_publisher<"), 1)
        self.assertIn('kInputTopic[] = "/pico4/body_tracking/ik_poses"', node)
        self.assertIn(
            'kUpperBodyTopic[] = "/pico4/retargeted/joint_targets"', node
        )
        self.assertIn(
            'kOutputTopic[] = "/pico4/retargeted/full_body_joint_targets"', node
        )
        self.assertGreaterEqual(node.count("rclcpp::KeepAll()"), 2)
        self.assertIn("rclcpp::KeepLast(1)", node)
        self.assertIn("std::deque<Sample>", node)
        self.assertNotIn("pop_back", node)
        self.assertNotIn("KEEP_LAST", node)
        self.assertNotIn("/hybrid_body_controller/commands", node + core)
        self.assertIn("GraftPicoUpperBody", node)
        self.assertIn("kengo_torso_target_upper_grafted", node)
        self.assertIn("kengo_torso_target_camera_fullbody", node)
        self.assertIn("kUpperBodyMaximumAge", node)
        self.assertNotIn("if (!upper.has_value()) {\n          upper_unavailable_", node)
        for marker in ("create_service", "create_client", "create_wall_timer"):
            self.assertNotIn(marker, node)

    def test_upper_follow_velocity_is_bounded_and_hot_adjustable(self) -> None:
        node = (ROOT / "cpp/src/fullbody_retarget_node.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            '"max_upper_follow_velocity_rad_s"', node
        )
        self.assertIn("kDefaultUpperFollowVelocityRadS = 0.60", node)
        self.assertIn("kMinimumUpperFollowVelocityRadS = 0.10", node)
        self.assertIn("kMaximumUpperFollowVelocityRadS = 2.00", node)
        self.assertIn("add_on_set_parameters_callback", node)
        self.assertIn("std::atomic<double> upper_follow_velocity_rad_s_", node)
        self.assertIn("ApplyUpperFollowLimit(result, follow_time)", node)
        self.assertIn(".start_parameter_services(true)", node)

    def test_runner_executes_native_binary_without_python(self) -> None:
        runner = (ROOT / "deployment/run_realtime_fullbody.sh").read_text(
            encoding="utf-8"
        )
        builder = (ROOT / "deployment/build_realtime_fullbody_cpp.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('exec "${RELEASE_DIR}/bin/kengo_fullbody_retarget_node"', runner)
        self.assertNotIn("python", runner.lower())
        self.assertIn("/var/lib/kengo-robot-gui/upper_follow_speed_rad_s", runner)
        self.assertIn('max_upper_follow_velocity_rad_s:=${UPPER_FOLLOW_SPEED}', runner)
        self.assertIn("value >= 0.10 && value <= 2.00", runner)
        self.assertIn("libmujoco.so.3.3.4", builder)
        self.assertIn("--self-test", builder)

    def test_native_binary_matches_python_when_provided(self) -> None:
        binary_text = os.environ.get("KENGO_FULLBODY_CPP_BINARY", "")
        if not binary_text:
            self.skipTest("KENGO_FULLBODY_CPP_BINARY is not set")
        binary = Path(binary_text).resolve()
        self.assertTrue(binary.is_file())
        positions, quaternions = synthetic_smpl24()
        expected = RealtimeKengoRetargeter(MODEL, CONFIG).solve(
            positions, quaternions
        )
        frame = np.hstack((positions, quaternions))
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "pose.txt"
            np.savetxt(fixture, frame, fmt="%.17g")
            completed = subprocess.run(
                [
                    str(binary),
                    "--model",
                    str(MODEL),
                    "--config",
                    str(CONFIG),
                    "--self-test",
                    "--fixture",
                    str(fixture),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
        actual = json.loads(completed.stdout.splitlines()[0])
        np.testing.assert_allclose(
            actual["positions"], expected.positions, rtol=0.0, atol=2.0e-9
        )
        self.assertAlmostEqual(actual["task_error"], expected.task_error, places=9)
        self.assertEqual(actual["iterations"], expected.iterations)


if __name__ == "__main__":
    unittest.main()

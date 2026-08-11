"""Regression tests for Kengo retarget motion time scaling."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from scripts.robot_retarget import RobotRetarget


class _FakeRetarget:
    fps = 30.0
    verbose = False
    model = SimpleNamespace(njnt=0)

    _slerp_wxyz = staticmethod(RobotRetarget._slerp_wxyz)
    _interpolate_qpos = RobotRetarget._interpolate_qpos
    _subdivide_qpos_intervals = RobotRetarget._subdivide_qpos_intervals
    _resample_to_joint_velocity_limit = RobotRetarget._resample_to_joint_velocity_limit
    _local_velocity_ratio_profile = staticmethod(
        RobotRetarget._local_velocity_ratio_profile
    )
    _resample_to_local_velocity_ratio = (
        RobotRetarget._resample_to_local_velocity_ratio
    )

    def _actuated_joint_qpos(self):
        return ["joint"], np.asarray([0], dtype=np.int64)


class KengoMotionPostprocessTests(unittest.TestCase):
    def test_subdivides_only_an_over_speed_interval_and_preserves_keyframes(self) -> None:
        retarget = _FakeRetarget()
        source = np.asarray([[0.0], [1.0], [1.1]], dtype=np.float64)

        result = retarget._resample_to_joint_velocity_limit(source, 15.0)

        np.testing.assert_allclose(result, [[0.0], [0.5], [1.0], [1.1]])
        self.assertLessEqual(float(np.abs(np.diff(result[:, 0]) * retarget.fps).max()), 15.0)
        self.assertEqual(result[-1, 0], source[-1, 0])

    def test_motion_inside_limit_is_returned_without_resampling(self) -> None:
        retarget = _FakeRetarget()
        source = np.asarray([[0.0], [0.1], [0.2]], dtype=np.float64)
        result = retarget._resample_to_joint_velocity_limit(source, 15.0)
        self.assertIs(result, source)

    def test_quaternion_interpolation_uses_the_shortest_normalized_path(self) -> None:
        retarget = _FakeRetarget()
        result = retarget._slerp_wxyz(
            np.asarray([1.0, 0.0, 0.0, 0.0]),
            np.asarray([-1.0, 0.0, 0.0, 0.0]),
            0.5,
        )
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0)

    def test_local_velocity_spike_is_time_scaled_below_validation_gate(self) -> None:
        retarget = _FakeRetarget()
        source = np.asarray(
            [[0.0], [0.01], [0.02], [0.32], [0.33], [0.34]],
            dtype=np.float64,
        )

        result = retarget._resample_to_local_velocity_ratio(
            source,
            max_local_ratio=5.0,
            window=6,
            min_speed=5.0,
        )

        self.assertGreater(result.shape[0], source.shape[0])
        source_cursor = 0
        for frame in result:
            if source_cursor < len(source) and np.array_equal(
                frame, source[source_cursor]
            ):
                source_cursor += 1
        self.assertEqual(source_cursor, len(source))
        velocity_norm = np.linalg.norm(
            np.diff(result, axis=0) * retarget.fps, axis=1
        )
        ratios, _ = retarget._local_velocity_ratio_profile(
            velocity_norm, window=6, min_speed=5.0
        )
        self.assertLessEqual(float(ratios.max()), 5.0)

    def test_local_velocity_resampling_leaves_smooth_motion_unchanged(self) -> None:
        retarget = _FakeRetarget()
        source = np.asarray(
            [[0.0], [0.2], [0.4], [0.6], [0.8]], dtype=np.float64
        )

        result = retarget._resample_to_local_velocity_ratio(
            source,
            max_local_ratio=5.0,
            window=2,
            min_speed=5.0,
        )

        self.assertIs(result, source)


if __name__ == "__main__":
    unittest.main()

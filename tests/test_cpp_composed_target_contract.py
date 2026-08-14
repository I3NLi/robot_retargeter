"""Contracts for full-body upper + measured lower telemetry."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ComposedTargetContractTests(unittest.TestCase):
    def test_node_is_read_only_and_walk_independent(self) -> None:
        node = (ROOT / "cpp/src/fullbody_composed_target_node.cpp").read_text(
            encoding="utf-8"
        )
        self.assertEqual(node.count("create_subscription<"), 2)
        self.assertEqual(node.count("create_publisher<"), 1)
        self.assertIn(
            '"/pico4/retargeted/full_body_joint_targets"', node
        )
        self.assertIn('"/hybrid_body_controller/joint_states"', node)
        self.assertIn(
            '"/pico4/retargeted/composed_joint_targets"', node
        )
        self.assertIn("KeepAll()", node)
        self.assertIn("KeepLast(1)", node)
        self.assertIn("kMeasuredMaximumAge = 50ms", node)
        for forbidden in (
            "/hybrid_body_controller/commands",
            "HybridJointCommand",
            "create_timer",
            "create_service",
            "create_client",
        ):
            self.assertNotIn(forbidden, node)

    def test_core_uses_target_upper_and_measured_waist_legs(self) -> None:
        header = (
            ROOT / "cpp/include/kengo_fullbody/composed_target.hpp"
        ).read_text(encoding="utf-8")
        core = (ROOT / "cpp/src/fullbody_composed_target.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn("kComposedTargetJointCount = 23", header)
        self.assertIn("kComposedTargetUpperJointCount = 10", header)
        self.assertIn("? target[index]", core)
        self.assertIn(": measured[index]", core)
        self.assertIn("must contain exactly 23 joints", core)

    def test_unit_is_independent_and_low_priority(self) -> None:
        unit = (
            ROOT / "deployment/kengo-fullbody-composed-target.service"
        ).read_text(encoding="utf-8")
        runner = (
            ROOT / "deployment/run_fullbody_composed_target.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("CPUAffinity=7", unit)
        self.assertIn("Nice=10", unit)
        self.assertIn("ProtectSystem=strict", unit)
        self.assertIn("kengo_fullbody_composed_target_node", runner)
        self.assertNotIn("Walk", runner)


if __name__ == "__main__":
    unittest.main()

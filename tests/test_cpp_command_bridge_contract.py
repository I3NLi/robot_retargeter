"""Static contracts for the opt-in native commands publisher."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class NativeCommandBridgeContractTests(unittest.TestCase):
    def test_node_has_only_the_requested_ros_interfaces(self) -> None:
        node = (ROOT / "cpp/src/fullbody_command_bridge_node.cpp").read_text(
            encoding="utf-8"
        )
        self.assertEqual(node.count("create_subscription<"), 2)
        self.assertEqual(node.count("create_publisher<"), 1)
        self.assertIn('kTargetTopic[] =\n    "/pico4/retargeted/full_body_joint_targets"', node)
        self.assertIn(
            'kFeedbackTopic[] = "/hybrid_body_controller/joint_states"', node
        )
        self.assertIn('kCommandTopic[] = "/hybrid_body_controller/commands"', node)
        self.assertIn("kCommandPeriod = 20ms", node)
        self.assertIn("kTargetMaximumAge = 200ms", node)
        self.assertIn("kFeedbackMaximumAge = 50ms", node)
        self.assertIn("publisher_count > 1", node)
        self.assertNotIn("create_service", node)
        self.assertNotIn("create_client", node)

    def test_composer_uses_bounded_upper_and_measured_lower(self) -> None:
        core = (ROOT / "cpp/src/fullbody_command_composer.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn("kCommandUpperJointCount = 10", (
            ROOT / "cpp/include/kengo_fullbody/command_composer.hpp"
        ).read_text(encoding="utf-8"))
        self.assertIn("kMaximumTargetErrorRad = 0.25", core)
        self.assertIn("kMaximumUpperVelocityRadPerSecond = 0.60", core)
        self.assertIn("canonical_position = measured.canonical", core)
        self.assertIn("kMeasuredLowerKp = 25.0", core)
        self.assertIn("kMeasuredLowerKd = 1.0", core)

    def test_service_is_fail_closed_until_explicitly_armed(self) -> None:
        runner = (ROOT / "deployment/run_fullbody_command_bridge.sh").read_text(
            encoding="utf-8"
        )
        unit = (ROOT / "deployment/kengo-fullbody-command-bridge.service").read_text(
            encoding="utf-8"
        )
        self.assertIn('KENGO_FULLBODY_COMMAND_ARMED:-0', runner)
        self.assertIn("Environment=KENGO_FULLBODY_COMMAND_ARMED=0", unit)
        self.assertIn("CPUAffinity=5", unit)


if __name__ == "__main__":
    unittest.main()

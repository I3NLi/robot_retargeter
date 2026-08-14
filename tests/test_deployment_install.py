import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DeploymentInstallTests(unittest.TestCase):
    def test_native_build_archives_direct_command_publisher(self):
        cmake = (ROOT / "cpp" / "CMakeLists.txt").read_text(encoding="utf-8")
        build = (ROOT / "deployment" / "build_realtime_fullbody_cpp.sh").read_text(encoding="utf-8")
        self.assertIn("BUILD_LEGACY_COMMAND_BRIDGE", cmake)
        self.assertIn("-DBUILD_LEGACY_COMMAND_BRIDGE=OFF", build)
        self.assertIn('[[ ! -e "${candidate}/bin/kengo_fullbody_command_bridge_node" ]]', (ROOT / "deployment" / "install.sh").read_text(encoding="utf-8"))

    def test_robot_install_is_walk_gated_atomic_and_rollback_capable(self):
        source = (ROOT / "deployment" / "install.sh").read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count("safe_robot_state"), 3)
        self.assertIn("current_mode", source)
        self.assertIn("atomic_link", source)
        self.assertIn("rollback", source)
        self.assertIn('systemctl disable --now "${COMMAND_UNIT}"', source)
        self.assertIn("healthy_samples >= 4", source)

    def test_windows_wrapper_is_exact_private_and_password_safe(self):
        source = (ROOT / "deployment" / "install_remote.ps1").read_text(encoding="utf-8")
        self.assertIn("expected exactly 27 Kengo STL files", source)
        self.assertIn("-hostkey", source)
        self.assertIn("-pwfile", source)
        self.assertNotIn('"-pw",', source)
        self.assertIn("sudo -S -p '' --", source)
        self.assertIn("$global:LASTEXITCODE = 0", source)


if __name__ == "__main__":
    unittest.main()

#include "kengo_fullbody/composed_target.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
  const auto& canonical =
      kengo_fullbody::CanonicalComposedTargetJointNames();
  std::vector<std::string> target_names(canonical.rbegin(), canonical.rend());
  std::vector<std::string> measured_names(canonical.begin(), canonical.end());
  std::vector<double> target_positions(target_names.size());
  std::vector<double> measured_positions(measured_names.size());
  for (std::size_t i = 0; i < canonical.size(); ++i) {
    target_positions[i] = 100.0 + static_cast<double>(canonical.size() - 1 - i);
    measured_positions[i] = -100.0 - static_cast<double>(i);
  }
  // Keep the strict parser inside its bounded numeric envelope.
  for (std::size_t i = 0; i < target_positions.size(); ++i) {
    target_positions[i] *= 0.01;
    measured_positions[i] *= 0.01;
  }

  const auto result =
      kengo_fullbody::ComposeFullBodyUpperWithMeasuredLower(
          target_names, target_positions, measured_names, measured_positions);
  for (std::size_t i = 0; i < canonical.size(); ++i) {
    if (result.joint_names[i] != canonical[i]) {
      std::cerr << "joint-name order mismatch at " << i << '\n';
      return 1;
    }
    const double expected = i < kengo_fullbody::kComposedTargetUpperJointCount
                                ? 1.0 + static_cast<double>(i) * 0.01
                                : -1.0 - static_cast<double>(i) * 0.01;
    if (std::abs(result.positions[i] - expected) >= 1.0e-12) {
      std::cerr << "composition mismatch at " << i << '\n';
      return 1;
    }
  }

  bool duplicate_rejected = false;
  target_names[1] = target_names[0];
  try {
    static_cast<void>(
        kengo_fullbody::ComposeFullBodyUpperWithMeasuredLower(
            target_names, target_positions, measured_names,
            measured_positions));
  } catch (const kengo_fullbody::ComposedTargetError&) {
    duplicate_rejected = true;
  }
  if (!duplicate_rejected) {
    std::cerr << "duplicate target joint was accepted\n";
    return 1;
  }
  std::cout << "full-body upper + measured lower composition contract ok\n";
  return 0;
}

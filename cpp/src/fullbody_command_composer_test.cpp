#include "kengo_fullbody/command_composer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

bool Near(double first, double second, double tolerance = 1.0e-9) {
  return std::abs(first - second) <= tolerance;
}

int Fail(const char* message) {
  std::cerr << message << std::endl;
  return 1;
}

}  // namespace

int main() {
  const auto& canonical = kengo_fullbody::CanonicalCommandJointNames();
  std::vector<std::string> measured_names(canonical.rbegin(), canonical.rend());
  std::vector<double> measured_positions(measured_names.size());
  std::vector<std::string> target_names(canonical.begin(), canonical.end());
  std::vector<double> target_positions(target_names.size());
  for (std::size_t index = 0; index < measured_names.size(); ++index) {
    const auto match = std::find(canonical.begin(), canonical.end(),
                                 measured_names[index]);
    measured_positions[index] =
        0.01 * static_cast<double>(std::distance(canonical.begin(), match));
  }
  for (std::size_t index = 0; index < target_positions.size(); ++index) {
    target_positions[index] = 1.0 + 0.01 * static_cast<double>(index);
  }

  kengo_fullbody::FullBodyCommandComposer composer;
  const auto first = composer.Compose(measured_names, measured_positions,
                                      target_names, target_positions, 0.02);
  for (std::size_t output = 0; output < first.joint_names.size(); ++output) {
    const auto match = std::find(canonical.begin(), canonical.end(),
                                 first.joint_names[output]);
    const auto canonical_index = static_cast<std::size_t>(
        std::distance(canonical.begin(), match));
    if (canonical_index < kengo_fullbody::kCommandUpperJointCount) {
      if (!Near(first.position[output], measured_positions[output] + 0.012)) {
        return Fail("upper target did not use the 0.012 rad slew limit");
      }
      const bool shoulder =
          first.joint_names[output].find("_shoulder_") != std::string::npos;
      if (!Near(first.kp[output], shoulder ? 22.21 : 30.0) ||
          !Near(first.kd[output], shoulder ? 1.41 : 1.0)) {
        return Fail("upper gains are incorrect");
      }
    } else {
      if (!Near(first.position[output], measured_positions[output]) ||
          !Near(first.kp[output], 25.0) || !Near(first.kd[output], 1.0)) {
        return Fail("measured lower body was not preserved");
      }
    }
  }
  const auto second = composer.Compose(measured_names, measured_positions,
                                       target_names, target_positions, 0.02);
  for (std::size_t output = 0; output < second.joint_names.size(); ++output) {
    if (kengo_fullbody::IsCommandUpperJoint(second.joint_names[output]) &&
        !Near(second.position[output], first.position[output] + 0.012)) {
      return Fail("second command did not continue the bounded slew");
    }
  }
  const auto released = composer.Release(measured_names, measured_positions);
  for (std::size_t index = 0; index < released.position.size(); ++index) {
    if (!Near(released.position[index], measured_positions[index]) ||
        !Near(released.kp[index], 0.0) || !Near(released.kd[index], 0.0)) {
      return Fail("release frame is not zero-gain measured feedback");
    }
  }

  bool rejected = false;
  try {
    auto duplicate = target_names;
    duplicate.back() = duplicate.front();
    composer.Compose(measured_names, measured_positions, duplicate,
                     target_positions, 0.02);
  } catch (const kengo_fullbody::CommandCompositionError&) {
    rejected = true;
  }
  if (!rejected) {
    return Fail("duplicate target joint was accepted");
  }
  rejected = false;
  try {
    auto nonfinite = measured_positions;
    nonfinite.front() = std::numeric_limits<double>::quiet_NaN();
    composer.Compose(measured_names, nonfinite, target_names, target_positions,
                     0.02);
  } catch (const kengo_fullbody::CommandCompositionError&) {
    rejected = true;
  }
  if (!rejected) {
    return Fail("non-finite feedback was accepted");
  }

  std::cout << "fullbody command composition contract ok" << std::endl;
  return 0;
}

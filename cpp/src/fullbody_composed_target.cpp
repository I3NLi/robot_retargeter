#include "kengo_fullbody/composed_target.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <unordered_map>

namespace kengo_fullbody {
namespace {

const ComposedTargetNames kJointNames = {
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",   "left_elbow_joint",
    "left_wrist_roll_joint",     "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",         "right_wrist_roll_joint",
    "waist_yaw_joint",           "left_hip_pitch_joint",
    "left_hip_roll_joint",       "left_hip_yaw_joint",
    "left_knee_joint",           "left_ankle_pitch_joint",
    "left_ankle_roll_joint",     "right_hip_pitch_joint",
    "right_hip_roll_joint",      "right_hip_yaw_joint",
    "right_knee_joint",          "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
};

const std::unordered_map<std::string, std::size_t>& JointIndex() {
  static const std::unordered_map<std::string, std::size_t> index = []() {
    std::unordered_map<std::string, std::size_t> result;
    for (std::size_t i = 0; i < kJointNames.size(); ++i) {
      result.emplace(kJointNames[i], i);
    }
    return result;
  }();
  return index;
}

ComposedTargetValues ParseExact(const std::vector<std::string>& names,
                                const std::vector<double>& positions,
                                const char* label) {
  if (names.size() != kComposedTargetJointCount ||
      positions.size() != kComposedTargetJointCount) {
    throw ComposedTargetError(std::string(label) +
                              " must contain exactly 23 joints");
  }
  ComposedTargetValues canonical{};
  std::array<bool, kComposedTargetJointCount> seen{};
  for (std::size_t source = 0; source < names.size(); ++source) {
    const auto match = JointIndex().find(names[source]);
    if (match == JointIndex().end()) {
      throw ComposedTargetError(std::string(label) +
                                " contains an unknown joint");
    }
    const std::size_t canonical_index = match->second;
    const double value = positions[source];
    if (seen[canonical_index]) {
      throw ComposedTargetError(std::string(label) +
                                " contains a duplicate joint");
    }
    if (!std::isfinite(value) || std::abs(value) > 32.0) {
      throw ComposedTargetError(std::string(label) +
                                " contains an invalid position");
    }
    seen[canonical_index] = true;
    canonical[canonical_index] = value;
  }
  if (std::find(seen.begin(), seen.end(), false) != seen.end()) {
    throw ComposedTargetError(std::string(label) +
                              " is missing a required joint");
  }
  return canonical;
}

}  // namespace

const ComposedTargetNames& CanonicalComposedTargetJointNames() {
  return kJointNames;
}

bool IsComposedTargetUpperJoint(const std::string& joint_name) {
  const auto match = JointIndex().find(joint_name);
  return match != JointIndex().end() &&
         match->second < kComposedTargetUpperJointCount;
}

ComposedTarget ComposeFullBodyUpperWithMeasuredLower(
    const std::vector<std::string>& target_names,
    const std::vector<double>& target_positions,
    const std::vector<std::string>& measured_names,
    const std::vector<double>& measured_positions) {
  const ComposedTargetValues target =
      ParseExact(target_names, target_positions, "full-body target");
  const ComposedTargetValues measured =
      ParseExact(measured_names, measured_positions, "measured feedback");

  ComposedTarget result;
  result.joint_names = kJointNames;
  for (std::size_t index = 0; index < kComposedTargetJointCount; ++index) {
    result.positions[index] =
        index < kComposedTargetUpperJointCount ? target[index]
                                               : measured[index];
  }
  return result;
}

}  // namespace kengo_fullbody

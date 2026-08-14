#include "kengo_fullbody/command_composer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <unordered_map>
#include <utility>

namespace kengo_fullbody {
namespace {

constexpr double kMaximumTargetErrorRad = 0.25;
constexpr double kMaximumUpperVelocityRadPerSecond = 0.60;
constexpr double kMeasuredLowerKp = 25.0;
constexpr double kMeasuredLowerKd = 1.0;
constexpr double kShoulderKp = 22.21;
constexpr double kShoulderKd = 1.41;
constexpr double kArmKp = 30.0;
constexpr double kArmKd = 1.0;

const CommandNames kJointNames = {
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

const std::array<double, kCommandJointCount> kJointLower = {
    -3.49, -0.17, -2.61, -0.52, -2.09, -3.49, -2.44, -2.61,
    -0.52, -2.09, -2.70, -2.79, -0.43, -2.61, -0.31, -0.85,
    -0.26, -2.79, -2.09, -2.61, -0.31, -0.85, -0.26,
};

const std::array<double, kCommandJointCount> kJointUpper = {
    1.74, 2.44, 2.61, 1.57, 2.09, 1.74, 0.17, 2.61,
    1.57, 2.09, 2.70, 2.79, 2.09, 2.61, 2.39, 0.48,
    0.26, 2.79, 0.43, 2.61, 2.39, 0.48, 0.26,
};

struct ParsedSample {
  CommandValues canonical{};
  std::array<std::size_t, kCommandJointCount> source_indices{};
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

ParsedSample ParseExact(const std::vector<std::string>& names,
                        const std::vector<double>& positions,
                        const char* label) {
  if (names.size() != kCommandJointCount ||
      positions.size() != kCommandJointCount) {
    throw CommandCompositionError(std::string(label) +
                                  " must contain exactly 23 joints");
  }
  ParsedSample parsed;
  std::array<bool, kCommandJointCount> seen{};
  for (std::size_t source = 0; source < names.size(); ++source) {
    const auto match = JointIndex().find(names[source]);
    if (match == JointIndex().end()) {
      throw CommandCompositionError(std::string(label) +
                                    " contains an unknown joint");
    }
    const std::size_t canonical = match->second;
    const double value = positions[source];
    if (seen[canonical]) {
      throw CommandCompositionError(std::string(label) +
                                    " contains a duplicate joint");
    }
    if (!std::isfinite(value) || std::abs(value) > 32.0) {
      throw CommandCompositionError(std::string(label) +
                                    " contains an invalid position");
    }
    seen[canonical] = true;
    parsed.canonical[canonical] = value;
    parsed.source_indices[canonical] = source;
  }
  if (std::find(seen.begin(), seen.end(), false) != seen.end()) {
    throw CommandCompositionError(std::string(label) +
                                  " is missing a required joint");
  }
  return parsed;
}

bool IsShoulder(const std::string& name) {
  return name.find("_shoulder_") != std::string::npos;
}

}  // namespace

const CommandNames& CanonicalCommandJointNames() { return kJointNames; }

bool IsCommandUpperJoint(const std::string& joint_name) {
  const auto match = JointIndex().find(joint_name);
  return match != JointIndex().end() && match->second < kCommandUpperJointCount;
}

ComposedCommand FullBodyCommandComposer::Compose(
    const std::vector<std::string>& measured_names,
    const std::vector<double>& measured_positions,
    const std::vector<std::string>& target_names,
    const std::vector<double>& target_positions, double period_seconds) {
  if (!std::isfinite(period_seconds) || period_seconds <= 0.0 ||
      period_seconds > 0.1) {
    throw CommandCompositionError("command period is invalid");
  }
  const ParsedSample measured =
      ParseExact(measured_names, measured_positions, "measured feedback");
  const ParsedSample target =
      ParseExact(target_names, target_positions, "full-body target");
  const double maximum_step =
      kMaximumUpperVelocityRadPerSecond * period_seconds;

  CommandValues canonical_position = measured.canonical;
  CommandValues canonical_kp{};
  CommandValues canonical_kd{};
  for (std::size_t index = 0; index < kCommandJointCount; ++index) {
    if (index < kCommandUpperJointCount) {
      const double bounded_target = std::clamp(
          target.canonical[index],
          std::max(kJointLower[index],
                   measured.canonical[index] - kMaximumTargetErrorRad),
          std::min(kJointUpper[index],
                   measured.canonical[index] + kMaximumTargetErrorRad));
      const double base = has_previous_upper_ ? previous_upper_[index]
                                               : measured.canonical[index];
      canonical_position[index] =
          base + std::clamp(bounded_target - base, -maximum_step, maximum_step);
      previous_upper_[index] = canonical_position[index];
      canonical_kp[index] = IsShoulder(kJointNames[index]) ? kShoulderKp : kArmKp;
      canonical_kd[index] = IsShoulder(kJointNames[index]) ? kShoulderKd : kArmKd;
    } else {
      canonical_kp[index] = kMeasuredLowerKp;
      canonical_kd[index] = kMeasuredLowerKd;
    }
  }
  has_previous_upper_ = true;

  ComposedCommand command;
  for (std::size_t canonical = 0; canonical < kCommandJointCount;
       ++canonical) {
    const std::size_t output = measured.source_indices[canonical];
    command.joint_names[output] = kJointNames[canonical];
    command.position[output] = canonical_position[canonical];
    command.velocity[output] = 0.0;
    command.effort[output] = 0.0;
    command.kp[output] = canonical_kp[canonical];
    command.kd[output] = canonical_kd[canonical];
  }
  return command;
}

ComposedCommand FullBodyCommandComposer::Release(
    const std::vector<std::string>& measured_names,
    const std::vector<double>& measured_positions) const {
  const ParsedSample measured =
      ParseExact(measured_names, measured_positions, "measured feedback");
  ComposedCommand command;
  for (std::size_t canonical = 0; canonical < kCommandJointCount;
       ++canonical) {
    const std::size_t output = measured.source_indices[canonical];
    command.joint_names[output] = kJointNames[canonical];
    command.position[output] = measured.canonical[canonical];
  }
  return command;
}

void FullBodyCommandComposer::Reset() {
  has_previous_upper_ = false;
  previous_upper_.fill(0.0);
}

}  // namespace kengo_fullbody

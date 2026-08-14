#include "kengo_fullbody/retarget_core.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <string>

int main() {
  const std::array<std::string, kengo_fullbody::kKengoJointCount> names = {
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
  std::array<double, kengo_fullbody::kKengoJointCount> fullbody{};
  for (std::size_t index = 0; index < fullbody.size(); ++index) {
    fullbody[index] = 100.0 + static_cast<double>(index);
  }
  const std::array<double, kengo_fullbody::kPicoUpperBodyJointCount> upper = {
      0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4};
  kengo_fullbody::GraftPicoUpperBody(names, upper, fullbody);
  const std::array<std::size_t, kengo_fullbody::kPicoUpperBodyJointCount>
      grafted_indices = {0, 1, 2, 3, 5, 6, 7, 8};
  for (std::size_t index = 0; index < grafted_indices.size(); ++index) {
    if (std::abs(fullbody[grafted_indices[index]] - upper[index]) > 1.0e-12) {
      std::cerr << "upper-body graft mismatch" << std::endl;
      return 1;
    }
  }
  if (fullbody[4] != 104.0 || fullbody[9] != 109.0 ||
      fullbody[10] != 110.0 || fullbody[22] != 122.0) {
    std::cerr << "non-grafted joint was modified" << std::endl;
    return 1;
  }
  std::cout << "upper-body graft contract ok" << std::endl;
  return 0;
}

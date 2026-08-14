#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <mujoco/mujoco.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kengo_fullbody {

inline constexpr std::size_t kSmplJointCount = 24;
inline constexpr std::size_t kKengoJointCount = 23;
inline constexpr std::size_t kPicoUpperBodyJointCount = 8;

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using Quat = Eigen::Quaterniond;

struct PoseFrame {
  std::array<Vec3, kSmplJointCount> positions{};
  std::array<Quat, kSmplJointCount> orientations{};
};

struct RetargetResult {
  std::array<double, kKengoJointCount> positions{};
  Vec3 root_position{Vec3::Zero()};
  Quat root_orientation{Quat::Identity()};
  double task_error{0.0};
  int iterations{0};
  double solve_duration_ms{0.0};
};

class RetargetError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class RealtimeKengoRetargeter {
 public:
  RealtimeKengoRetargeter(const std::string& model_path,
                          const std::string& config_path);
  ~RealtimeKengoRetargeter();

  RealtimeKengoRetargeter(const RealtimeKengoRetargeter&) = delete;
  RealtimeKengoRetargeter& operator=(const RealtimeKengoRetargeter&) = delete;

  RetargetResult Solve(const PoseFrame& pose);
  void Reset();

  const std::array<std::string, kKengoJointCount>& joint_names() const {
    return joint_names_;
  }
  int nq() const;
  int nv() const;

 private:
  struct Impl;
  std::array<std::string, kKengoJointCount> joint_names_{};
  std::unique_ptr<Impl> impl_;
};

PoseFrame SyntheticSmpl24();
void ValidatePose(const PoseFrame& pose);
void GraftPicoUpperBody(
    const std::array<std::string, kKengoJointCount>& joint_names,
    const std::array<double, kPicoUpperBodyJointCount>& upper_positions,
    std::array<double, kKengoJointCount>& fullbody_positions);

}  // namespace kengo_fullbody

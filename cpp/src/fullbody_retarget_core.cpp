#include "kengo_fullbody/retarget_core.hpp"

#include <Eigen/Cholesky>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <sstream>

namespace kengo_fullbody {
namespace {

using Vec6 = Eigen::Matrix<double, 6, 1>;
using Mat6 = Eigen::Matrix<double, 6, 6>;

constexpr double kEpsilon = 1.0e-10;
constexpr double kLimitGain = 0.95;
constexpr double kConvergenceDelta = 0.001;
constexpr double kGlobalDamping = 1.0;
constexpr int kMaximumIterations = 51;

const std::array<std::string, kKengoJointCount> kExpectedJointNames = {
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
    "right_ankle_roll_joint"};

struct Transform {
  Mat3 rotation{Mat3::Identity()};
  Vec3 translation{Vec3::Zero()};
};

struct LinkDefinition {
  std::string name;
  std::string start_body;
  std::string end_body;
  double length{0.0};
};

struct KeyFrameAdjustment {
  Mat3 axis_map{Mat3::Identity()};
  Mat3 offset{Mat3::Identity()};
};

struct TaskDefinition {
  std::string keypoint;
  std::string body_name;
  int body_id{-1};
  Eigen::Matrix<double, 6, 1> cost{Eigen::Matrix<double, 6, 1>::Zero()};
  Transform target{};
};

struct TargetFrame {
  std::map<std::string, Vec3> positions;
  std::map<std::string, Quat> orientations;
};

Mat3 Skew(const Vec3& value) {
  Mat3 result;
  result << 0.0, -value.z(), value.y(), value.z(), 0.0, -value.x(),
      -value.y(), value.x(), 0.0;
  return result;
}

Vec3 Unit(const Vec3& value, const std::string& label,
          double minimum = 1.0e-5) {
  const double length = value.norm();
  if (!std::isfinite(length) || length < minimum) {
    throw RetargetError(label + " is degenerate");
  }
  return value / length;
}

bool Finite(const Mat3& value) { return value.array().isFinite().all(); }
bool Finite(const Vec3& value) { return value.array().isFinite().all(); }

Transform Compose(const Transform& lhs, const Transform& rhs) {
  return {lhs.rotation * rhs.rotation,
          lhs.rotation * rhs.translation + lhs.translation};
}

Transform Inverse(const Transform& value) {
  const Mat3 inverse_rotation = value.rotation.transpose();
  return {inverse_rotation, -inverse_rotation * value.translation};
}

Vec3 So3Log(const Mat3& rotation) {
  Quat quaternion(rotation);
  quaternion.normalize();
  if (quaternion.w() < 0.0) {
    quaternion.coeffs() *= -1.0;
  }
  Vec3 vector(quaternion.x(), quaternion.y(), quaternion.z());
  const double norm = vector.norm();
  if (norm < kEpsilon) {
    return Vec3::Zero();
  }
  return (2.0 * std::atan2(norm, quaternion.w()) / norm) * vector;
}

Mat3 So3LeftJacobianInverse(const Vec3& tangent) {
  const double theta_squared = tangent.squaredNorm();
  double beta = 0.0;
  if (theta_squared < kEpsilon) {
    beta = (1.0 / 12.0) *
           (1.0 + theta_squared / 60.0 *
                      (1.0 + theta_squared / 42.0 *
                                 (1.0 + theta_squared / 40.0)));
  } else {
    const double theta = std::sqrt(theta_squared);
    beta = (1.0 / theta_squared) *
           (1.0 - theta * std::sin(theta) /
                      (2.0 * (1.0 - std::cos(theta))));
  }
  const Mat3 wedge = Skew(tangent);
  return Mat3::Identity() - 0.5 * wedge + beta * wedge * wedge;
}

Mat3 Se3Q(const Vec6& tangent) {
  const Vec3 translation = tangent.head<3>();
  const Vec3 rotation = tangent.tail<3>();
  const double theta = rotation.norm();
  const double theta2 = theta * theta;
  const double a = 0.5;
  double b = 0.0;
  double c = 0.0;
  double d = 0.0;
  if (theta2 < kEpsilon) {
    b = (1.0 / 6.0) + (1.0 / 120.0) * theta2;
    c = -(1.0 / 24.0) + (1.0 / 720.0) * theta2;
    d = -(1.0 / 60.0);
  } else {
    const double theta4 = theta2 * theta2;
    const double sine = std::sin(theta);
    const double cosine = std::cos(theta);
    b = (theta - sine) / (theta2 * theta);
    c = (1.0 - 0.5 * theta2 - cosine) / theta4;
    d = (2.0 * theta - 3.0 * sine + theta * cosine) /
        (2.0 * theta4 * theta);
  }
  const Mat3 v = Skew(translation);
  const Mat3 w = Skew(rotation);
  const Mat3 vw = v * w;
  const Mat3 wv = vw.transpose();
  const Mat3 wvw = wv * w;
  const Mat3 vww = vw * w;
  return a * v + b * (wv + vw + wvw) -
         c * (vww - vww.transpose() - 3.0 * wvw) +
         d * (wvw * w + w * wvw);
}

Vec6 Se3Log(const Transform& transform) {
  Vec6 result;
  const Vec3 omega = So3Log(transform.rotation);
  const double theta2 = omega.squaredNorm();
  const Mat3 wedge = Skew(omega);
  Mat3 inverse_v = Mat3::Identity() - 0.5 * wedge;
  if (theta2 < kEpsilon) {
    inverse_v += (1.0 / 12.0) * wedge * wedge;
  } else {
    const double theta = std::sqrt(theta2);
    const double half_theta = 0.5 * theta;
    inverse_v +=
        ((1.0 - 0.5 * theta * std::cos(half_theta) /
                    std::sin(half_theta)) /
         theta2) *
        wedge * wedge;
  }
  result.head<3>() = inverse_v * transform.translation;
  result.tail<3>() = omega;
  return result;
}

Mat6 Se3LeftJacobianInverse(const Vec6& tangent) {
  if (tangent.tail<3>().squaredNorm() < kEpsilon) {
    return Mat6::Identity();
  }
  const Mat3 rotation_inverse = So3LeftJacobianInverse(tangent.tail<3>());
  const Mat3 q = Se3Q(tangent);
  Mat6 result = Mat6::Zero();
  result.topLeftCorner<3, 3>() = rotation_inverse;
  result.topRightCorner<3, 3>() =
      -rotation_inverse * q * rotation_inverse;
  result.bottomRightCorner<3, 3>() = rotation_inverse;
  return result;
}

Mat6 Se3Jlog(const Transform& transform) {
  return Se3LeftJacobianInverse(-Se3Log(transform));
}

Quat AverageQuaternions(const Quat& left, const Quat& right) {
  Quat aligned = right;
  if (left.dot(right) < 0.0) {
    aligned.coeffs() *= -1.0;
  }
  Quat result(left.w() + aligned.w(), left.x() + aligned.x(),
              left.y() + aligned.y(), left.z() + aligned.z());
  if (result.norm() < kEpsilon) {
    throw RetargetError("quaternion average is degenerate");
  }
  return result.normalized();
}

Mat3 RotationBetween(const Vec3& source, const Vec3& target) {
  const Vec3 source_u = Unit(source, "source direction");
  const Vec3 target_u = Unit(target, "target direction");
  const double cosine =
      std::clamp(source_u.dot(target_u), -1.0, 1.0);
  Vec3 axis = source_u.cross(target_u);
  const double sine = axis.norm();
  if (sine < 1.0e-8) {
    if (cosine > 0.0) {
      return Mat3::Identity();
    }
    Vec3 fallback = Vec3::Zero();
    Eigen::Index index = 0;
    source_u.cwiseAbs().minCoeff(&index);
    fallback[index] = 1.0;
    axis = Unit(source_u.cross(fallback), "antiparallel rotation axis");
    return Eigen::AngleAxisd(M_PI, axis).toRotationMatrix();
  }
  axis /= sine;
  return Eigen::AngleAxisd(std::atan2(sine, cosine), axis).toRotationMatrix();
}

Vec3 YamlVec3(const YAML::Node& node, const std::string& label) {
  if (!node.IsSequence() || node.size() != 3) {
    throw RetargetError(label + " must contain three values");
  }
  Vec3 result(node[0].as<double>(), node[1].as<double>(),
              node[2].as<double>());
  if (!Finite(result)) {
    throw RetargetError(label + " is non-finite");
  }
  return result;
}

Mat3 OffsetMatrix(const YAML::Node& key_frame_config,
                  const std::string& body_name) {
  const YAML::Node body = key_frame_config[body_name];
  if (!body || !body["offset_deg_xyz"]) {
    return Mat3::Identity();
  }
  const Vec3 degrees =
      YamlVec3(body["offset_deg_xyz"], body_name + " offset_deg_xyz");
  const Vec3 radians = degrees * (M_PI / 180.0);
  const Mat3 rx =
      Eigen::AngleAxisd(radians.x(), Vec3::UnitX()).toRotationMatrix();
  const Mat3 ry =
      Eigen::AngleAxisd(radians.y(), Vec3::UnitY()).toRotationMatrix();
  const Mat3 rz =
      Eigen::AngleAxisd(radians.z(), Vec3::UnitZ()).toRotationMatrix();
  return rx * ry * rz;
}

Mat3 AxisMapMatrix(const YAML::Node& key_frame_config,
                   const std::string& body_name) {
  const YAML::Node body = key_frame_config[body_name];
  if (!body || !body["axis_map_cols"]) {
    return Mat3::Identity();
  }
  const YAML::Node axes = body["axis_map_cols"];
  Mat3 result;
  result.col(0) = YamlVec3(axes["x"], body_name + " axis x");
  result.col(1) = YamlVec3(axes["y"], body_name + " axis y");
  result.col(2) = YamlVec3(axes["z"], body_name + " axis z");
  if (!Finite(result)) {
    throw RetargetError(body_name + " axis map is non-finite");
  }
  return result;
}

Vec3 TwoBoneKnee(const Vec3& hip, const Vec3& knee, const Vec3& foot,
                 const Vec3& target_foot) {
  const Vec3 upper = knee - hip;
  const Vec3 lower = foot - knee;
  const double upper_length = upper.norm();
  const double lower_length = lower.norm();
  const Vec3 original_u = Unit(foot - hip, "hip-to-foot");
  const Vec3 knee_projection = hip + (knee - hip).dot(original_u) * original_u;
  Vec3 bend_preference = knee - knee_projection;
  if (bend_preference.norm() < 1.0e-6) {
    bend_preference = original_u.cross(Vec3::UnitZ());
  }
  if (bend_preference.norm() < 1.0e-6) {
    bend_preference = original_u.cross(Vec3::UnitY());
  }
  bend_preference = Unit(bend_preference, "knee bend preference");

  const Vec3 target_vector = target_foot - hip;
  double distance = target_vector.norm();
  const Vec3 target_u = Unit(target_vector, "target hip-to-foot");
  distance = std::clamp(distance, 1.0e-6,
                        upper_length + lower_length - 1.0e-6);
  const double along =
      (upper_length * upper_length - lower_length * lower_length +
       distance * distance) /
      (2.0 * distance);
  const double height =
      std::sqrt(std::max(upper_length * upper_length - along * along, 0.0));
  Vec3 bend_direction =
      bend_preference - bend_preference.dot(target_u) * target_u;
  if (bend_direction.norm() < 1.0e-6) {
    bend_direction = target_u.cross(Vec3::UnitY());
  }
  bend_direction = Unit(bend_direction, "target knee bend");
  return hip + along * target_u + height * bend_direction;
}

void BendLeg(std::map<std::string, Vec3>& positions, const std::string& side,
             double offset_degrees) {
  const std::string hip_name = side + "_up_leg";
  const std::string knee_name = side + "_leg";
  const std::string foot_name = side + "_foot";
  const Vec3 hip = positions.at(hip_name);
  const Vec3 knee = positions.at(knee_name);
  const Vec3 foot = positions.at(foot_name);
  const Vec3 upper = knee - hip;
  const Vec3 lower = foot - knee;
  const double upper_length = upper.norm();
  const double lower_length = lower.norm();
  const double cosine = std::clamp(
      upper.dot(lower) / std::max(upper_length * lower_length, 1.0e-8),
      -1.0, 1.0);
  const double angle = std::acos(cosine) + offset_degrees * M_PI / 180.0;
  const double target_length = std::sqrt(std::max(
      upper_length * upper_length + lower_length * lower_length +
          2.0 * upper_length * lower_length * std::cos(angle),
      0.0));
  const Vec3 target_foot =
      hip + Unit(foot - hip, side + " hip-to-foot") * target_length;
  positions[knee_name] = TwoBoneKnee(hip, knee, foot, target_foot);
  positions[foot_name] = target_foot;
}

Eigen::VectorXd SolveBoxQp(const Eigen::MatrixXd& hessian,
                           const Eigen::VectorXd& linear,
                           const Eigen::VectorXd& lower,
                           const Eigen::VectorXd& upper) {
  const Eigen::Index size = linear.size();
  Eigen::VectorXd solution =
      hessian.ldlt().solve(-linear);
  if (!solution.array().isFinite().all()) {
    throw RetargetError("native QP unconstrained solve failed");
  }
  std::vector<int> active(static_cast<std::size_t>(size), 0);

  for (int iteration = 0; iteration < 4 * size + 32; ++iteration) {
    double largest_violation = 0.0;
    Eigen::Index violation_index = -1;
    int violation_side = 0;
    for (Eigen::Index index = 0; index < size; ++index) {
      if (active[static_cast<std::size_t>(index)] != 0) {
        continue;
      }
      if (solution[index] < lower[index] - 1.0e-10) {
        const double amount = lower[index] - solution[index];
        if (amount > largest_violation) {
          largest_violation = amount;
          violation_index = index;
          violation_side = -1;
        }
      } else if (solution[index] > upper[index] + 1.0e-10) {
        const double amount = solution[index] - upper[index];
        if (amount > largest_violation) {
          largest_violation = amount;
          violation_index = index;
          violation_side = 1;
        }
      }
    }
    if (violation_index >= 0) {
      active[static_cast<std::size_t>(violation_index)] = violation_side;
      solution[violation_index] =
          violation_side < 0 ? lower[violation_index] : upper[violation_index];
    }

    std::vector<Eigen::Index> free_indices;
    std::vector<Eigen::Index> active_indices;
    free_indices.reserve(static_cast<std::size_t>(size));
    active_indices.reserve(static_cast<std::size_t>(size));
    for (Eigen::Index index = 0; index < size; ++index) {
      (active[static_cast<std::size_t>(index)] == 0 ? free_indices
                                                    : active_indices)
          .push_back(index);
    }
    if (!free_indices.empty()) {
      Eigen::MatrixXd hff(free_indices.size(), free_indices.size());
      Eigen::VectorXd rhs(free_indices.size());
      for (std::size_t row = 0; row < free_indices.size(); ++row) {
        rhs[static_cast<Eigen::Index>(row)] = -linear[free_indices[row]];
        for (const Eigen::Index active_index : active_indices) {
          rhs[static_cast<Eigen::Index>(row)] -=
              hessian(free_indices[row], active_index) *
              solution[active_index];
        }
        for (std::size_t column = 0; column < free_indices.size(); ++column) {
          hff(static_cast<Eigen::Index>(row),
              static_cast<Eigen::Index>(column)) =
              hessian(free_indices[row], free_indices[column]);
        }
      }
      const Eigen::VectorXd free_solution = hff.ldlt().solve(rhs);
      if (!free_solution.array().isFinite().all()) {
        throw RetargetError("native QP active-set solve failed");
      }
      for (std::size_t index = 0; index < free_indices.size(); ++index) {
        solution[free_indices[index]] =
            free_solution[static_cast<Eigen::Index>(index)];
      }
    }

    bool bounds_satisfied = true;
    for (const Eigen::Index index : free_indices) {
      if (solution[index] < lower[index] - 1.0e-10 ||
          solution[index] > upper[index] + 1.0e-10) {
        bounds_satisfied = false;
        break;
      }
    }
    if (!bounds_satisfied) {
      continue;
    }

    const Eigen::VectorXd gradient = hessian * solution + linear;
    double largest_kkt_violation = 0.0;
    Eigen::Index release_index = -1;
    for (const Eigen::Index index : active_indices) {
      const int side = active[static_cast<std::size_t>(index)];
      const double violation =
          side < 0 ? std::max(-gradient[index], 0.0)
                   : std::max(gradient[index], 0.0);
      if (violation > largest_kkt_violation + 1.0e-10) {
        largest_kkt_violation = violation;
        release_index = index;
      }
    }
    if (release_index >= 0) {
      active[static_cast<std::size_t>(release_index)] = 0;
      continue;
    }
    return solution;
  }
  throw RetargetError("native box QP did not converge");
}

const std::map<std::string, int> kSmplBodyIndex = {
    {"hips", 0},          {"left_up_leg", 1}, {"left_leg", 4},
    {"left_foot", 7},    {"left_toe", 10},   {"right_up_leg", 2},
    {"right_leg", 5},    {"right_foot", 8},  {"right_toe", 11},
    {"spine1", 3},       {"spine2", 6},      {"chest", 9},
    {"neck", 12},        {"head", 15},       {"left_shoulder", 13},
    {"left_arm", 16},    {"left_fore_arm", 18},
    {"left_hand", 20},   {"right_shoulder", 14},
    {"right_arm", 17},   {"right_fore_arm", 19},
    {"right_hand", 21}};

const std::vector<std::pair<std::string, std::string>> kBodyChildren = {
    {"left_up_leg", "left_leg"},       {"left_leg", "left_foot"},
    {"left_foot", "left_toe"},         {"right_up_leg", "right_leg"},
    {"right_leg", "right_foot"},       {"right_foot", "right_toe"},
    {"spine1", "spine2"},              {"spine2", "chest"},
    {"chest", "neck"},                 {"neck", "head"},
    {"left_shoulder", "left_arm"},     {"left_arm", "left_fore_arm"},
    {"left_fore_arm", "left_hand"},    {"right_shoulder", "right_arm"},
    {"right_arm", "right_fore_arm"},   {"right_fore_arm", "right_hand"}};

const std::map<std::string, Vec3> kBodyLocalDirections = {
    {"left_up_leg", Vec3(0.0, -1.0, 0.0)},
    {"left_leg", Vec3(0.0, -1.0, 0.0)},
    {"left_foot", Vec3(0.0, -0.054, 0.125)},
    {"right_up_leg", Vec3(0.0, -1.0, 0.0)},
    {"right_leg", Vec3(0.0, -1.0, 0.0)},
    {"right_foot", Vec3(0.0, -0.054, 0.125)},
    {"spine1", Vec3(0.0, 1.0, 0.0)},
    {"spine2", Vec3(0.0, 1.0, 0.0)},
    {"chest", Vec3(0.0, 1.0, 0.0)},
    {"neck", Vec3(0.0, 1.0, 0.0)},
    {"left_shoulder", Vec3(1.0, 0.0, 0.0)},
    {"left_arm", Vec3(1.0, 0.0, 0.0)},
    {"left_fore_arm", Vec3(1.0, 0.0, 0.0)},
    {"right_shoulder", Vec3(-1.0, 0.0, 0.0)},
    {"right_arm", Vec3(-1.0, 0.0, 0.0)},
    {"right_fore_arm", Vec3(-1.0, 0.0, 0.0)}};

const std::map<std::string, std::pair<std::string, std::string>>
    kSkeletonLinks = {
        {"left_hip", {"hips_mean", "left_up_leg"}},
        {"left_thigh", {"left_up_leg", "left_leg"}},
        {"left_calf", {"left_leg", "left_foot"}},
        {"right_hip", {"hips_mean", "right_up_leg"}},
        {"right_thigh", {"right_up_leg", "right_leg"}},
        {"right_calf", {"right_leg", "right_foot"}},
        {"neck", {"hips_mean", "shoulder_mean"}},
        {"head", {"shoulder_mean", "head"}},
        {"left_shoulder", {"shoulder_mean", "left_arm"}},
        {"left_arm", {"left_arm", "left_fore_arm"}},
        {"left_fore_arm", {"left_fore_arm", "left_hand"}},
        {"right_shoulder", {"shoulder_mean", "right_arm"}},
        {"right_arm", {"right_arm", "right_fore_arm"}},
        {"right_fore_arm", {"right_fore_arm", "right_hand"}}};

void AlignBoneRotations(const std::map<std::string, Vec3>& positions,
                        std::map<std::string, Mat3>& rotations) {
  for (const auto& [body, child] : kBodyChildren) {
    const Vec3 bone = positions.at(child) - positions.at(body);
    if (bone.norm() < 1.0e-6) {
      continue;
    }
    const Vec3 predicted = rotations.at(body) * kBodyLocalDirections.at(body);
    rotations.at(body) = RotationBetween(predicted, bone) * rotations.at(body);
  }
}

}  // namespace

void ValidatePose(const PoseFrame& pose) {
  for (std::size_t index = 0; index < kSmplJointCount; ++index) {
    if (!Finite(pose.positions[index]) ||
        !pose.orientations[index].coeffs().array().isFinite().all()) {
      throw RetargetError("pose contains non-finite values");
    }
    if (pose.positions[index].cwiseAbs().maxCoeff() > 50.0) {
      throw RetargetError("pose exceeds the 50 metre tracking envelope");
    }
    if (pose.orientations[index].norm() < 1.0e-6) {
      throw RetargetError("pose contains a zero-length quaternion");
    }
  }
}

void GraftPicoUpperBody(
    const std::array<std::string, kKengoJointCount>& joint_names,
    const std::array<double, kPicoUpperBodyJointCount>& upper_positions,
    std::array<double, kKengoJointCount>& fullbody_positions) {
  static const std::array<std::string, kPicoUpperBodyJointCount>
      kUpperJointNames = {
          "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
          "left_shoulder_yaw_joint",   "left_elbow_joint",
          "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
          "right_shoulder_yaw_joint",   "right_elbow_joint",
      };

  for (std::size_t upper = 0; upper < kUpperJointNames.size(); ++upper) {
    const double value = upper_positions[upper];
    if (!std::isfinite(value) || std::abs(value) > 32.0) {
      throw RetargetError("Pico upper-body target is non-finite or out of range");
    }
    const auto match =
        std::find(joint_names.begin(), joint_names.end(), kUpperJointNames[upper]);
    if (match == joint_names.end()) {
      throw RetargetError("Pico upper-body joint is absent from the Kengo model: " +
                          kUpperJointNames[upper]);
    }
    const auto index = static_cast<std::size_t>(
        std::distance(joint_names.begin(), match));
    fullbody_positions[index] = value;
  }
}

struct RealtimeKengoRetargeter::Impl {
  std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model{nullptr,
                                                            mj_deleteModel};
  std::unique_ptr<mjData, decltype(&mj_deleteData)> data{nullptr, mj_deleteData};
  YAML::Node config;
  YAML::Node key_frame_config;
  std::vector<LinkDefinition> links;
  std::vector<TaskDefinition> tasks;
  std::array<int, kKengoJointCount> qpos_indices{};
  std::array<int, kKengoJointCount> dof_indices{};
  Eigen::VectorXd neutral_q;
  double knee_offset_degrees{15.0};

  Impl(const std::string& model_path, const std::string& config_path,
       std::array<std::string, kKengoJointCount>& joint_names) {
    config = YAML::LoadFile(config_path);
    if (!config.IsMap()) {
      throw RetargetError("Kengo config must be a mapping");
    }
    key_frame_config = config["key_frame_config"];
    knee_offset_degrees = config["knee_angle_offset_degrees"].as<double>(15.0);

    char error[1024] = {};
    model.reset(mj_loadXML(model_path.c_str(), nullptr, error, sizeof(error)));
    if (!model) {
      throw RetargetError(std::string("MuJoCo model load failed: ") + error);
    }
    ApplyJointLimitOffsets();
    data.reset(mj_makeData(model.get()));
    if (!data) {
      throw RetargetError("MuJoCo data allocation failed");
    }
    mj_resetData(model.get(), data.get());
    UpdateKinematics();
    neutral_q = Eigen::Map<const Eigen::VectorXd>(data->qpos, model->nq);
    ParseLinks();
    ParseTasks();
    ResolveActuatedJoints(joint_names);
  }

  void UpdateKinematics() {
    mj_kinematics(model.get(), data.get());
    mj_comPos(model.get(), data.get());
    if (model->neq > 0) {
      mj_makeConstraint(model.get(), data.get());
    }
  }

  void ApplyJointLimitOffsets() {
    const YAML::Node offsets = config["joints_limit_offset_degrees"];
    if (!offsets || !offsets.IsMap()) {
      throw RetargetError("joints_limit_offset_degrees must be a mapping");
    }
    for (const auto& entry : offsets) {
      const std::string fragment = entry.first.as<std::string>();
      const YAML::Node value = entry.second;
      double lower_offset = 0.0;
      double upper_offset = 0.0;
      if (value.IsSequence() && value.size() == 2) {
        lower_offset = value[0].as<double>();
        upper_offset = value[1].as<double>();
      } else {
        lower_offset = value.as<double>();
      }
      bool matched = false;
      for (int joint = 0; joint < model->njnt; ++joint) {
        const char* name = mj_id2name(model.get(), mjOBJ_JOINT, joint);
        if (name == nullptr || std::string(name).find(fragment) == std::string::npos) {
          continue;
        }
        matched = true;
        model->jnt_range[2 * joint] += lower_offset * M_PI / 180.0;
        model->jnt_range[2 * joint + 1] += upper_offset * M_PI / 180.0;
      }
      if (!matched) {
        throw RetargetError("joint limit offset matched nothing: " + fragment);
      }
    }
  }

  void ParseLinks() {
    const YAML::Node robot_links = config["robot_links"];
    if (!robot_links || !robot_links.IsMap()) {
      throw RetargetError("robot_links must be a mapping");
    }
    for (const auto& entry : robot_links) {
      const std::string name = entry.first.as<std::string>();
      const YAML::Node bodies = entry.second;
      if (!bodies.IsSequence() || bodies.size() != 2) {
        throw RetargetError("robot link " + name + " must name two bodies");
      }
      LinkDefinition link{name, bodies[0].as<std::string>(),
                          bodies[1].as<std::string>(), 0.0};
      const int start = mj_name2id(model.get(), mjOBJ_BODY, link.start_body.c_str());
      const int end = mj_name2id(model.get(), mjOBJ_BODY, link.end_body.c_str());
      if (start < 0 || end < 0) {
        throw RetargetError("missing MJCF body for " + name);
      }
      const Vec3 start_position = Eigen::Map<const Vec3>(data->xpos + 3 * start);
      const Vec3 end_position = Eigen::Map<const Vec3>(data->xpos + 3 * end);
      link.length = (end_position - start_position).norm();
      if (!std::isfinite(link.length) || link.length <= 0.0) {
        throw RetargetError("invalid robot link length for " + name);
      }
      links.push_back(link);
    }
  }

  void ParseTasks() {
    const YAML::Node table = config["ik_match_table"];
    if (!table || !table.IsMap()) {
      throw RetargetError("ik_match_table must be a mapping");
    }
    for (const auto& entry : table) {
      const std::string keypoint = entry.first.as<std::string>();
      const YAML::Node values = entry.second;
      if (!values.IsSequence() || values.size() != 3) {
        throw RetargetError("invalid task entry for " + keypoint);
      }
      const double position_cost = values[1].as<double>();
      const double orientation_cost = values[2].as<double>();
      if (position_cost == 0.0 && orientation_cost == 0.0) {
        continue;
      }
      TaskDefinition task;
      task.keypoint = keypoint;
      task.body_name = values[0].as<std::string>();
      task.body_id =
          mj_name2id(model.get(), mjOBJ_BODY, task.body_name.c_str());
      if (task.body_id < 0) {
        throw RetargetError("missing task body " + task.body_name);
      }
      task.cost << position_cost, position_cost, position_cost,
          orientation_cost, orientation_cost, orientation_cost;
      tasks.push_back(task);
    }
  }

  void ResolveActuatedJoints(
      std::array<std::string, kKengoJointCount>& joint_names) {
    std::set<int> seen;
    std::size_t count = 0;
    for (int actuator = 0; actuator < model->nu; ++actuator) {
      const int joint = model->actuator_trnid[2 * actuator];
      if (joint < 0 || !seen.insert(joint).second) {
        continue;
      }
      const int type = model->jnt_type[joint];
      if (type != mjJNT_HINGE && type != mjJNT_SLIDE) {
        continue;
      }
      if (count >= kKengoJointCount) {
        throw RetargetError("too many actuated joints");
      }
      const char* name = mj_id2name(model.get(), mjOBJ_JOINT, joint);
      if (name == nullptr) {
        throw RetargetError("actuated joint has no name");
      }
      joint_names[count] = name;
      qpos_indices[count] = model->jnt_qposadr[joint];
      dof_indices[count] = model->jnt_dofadr[joint];
      ++count;
    }
    if (count != kKengoJointCount || joint_names != kExpectedJointNames) {
      throw RetargetError("unexpected Kengo actuator contract");
    }
  }

  TargetFrame MakeTargets(const PoseFrame& pose) const {
    ValidatePose(pose);
    const Vec3 right =
        Unit(pose.positions[17] - pose.positions[16], "shoulder axis", 0.05);
    const Vec3 up_hint = pose.positions[12] - pose.positions[0];
    const Vec3 up = Unit(up_hint - right * up_hint.dot(right), "torso up", 0.05);
    const Vec3 left = -right;
    const Vec3 forward = Unit(left.cross(up), "torso forward");
    Mat3 basis;
    basis.row(0) = forward.transpose();
    basis.row(1) = left.transpose();
    basis.row(2) = up.transpose();
    if (basis.determinant() < 0.999) {
      throw RetargetError("torso basis is not right-handed");
    }
    const Mat3 sdk_to_smpl =
        (Vec3(-1.0, 1.0, -1.0)).asDiagonal();

    std::map<std::string, Vec3> body_positions;
    std::map<std::string, Mat3> body_rotations;
    for (const auto& [name, index] : kSmplBodyIndex) {
      body_positions[name] = basis * (pose.positions[index] - pose.positions[0]);
      body_rotations[name] =
          basis * pose.orientations[index].normalized().toRotationMatrix() *
          sdk_to_smpl;
    }
    body_positions["hips_mean"] =
        0.5 * (body_positions["left_up_leg"] + body_positions["right_up_leg"]);
    body_positions["shoulder_mean"] =
        0.5 * (body_positions["left_arm"] + body_positions["right_arm"]);
    body_rotations["hips_mean"] =
        AverageQuaternions(Quat(body_rotations["left_up_leg"]),
                           Quat(body_rotations["right_up_leg"]))
            .toRotationMatrix();
    body_rotations["shoulder_mean"] =
        AverageQuaternions(Quat(body_rotations["left_arm"]),
                           Quat(body_rotations["right_arm"]))
            .toRotationMatrix();
    AlignBoneRotations(body_positions, body_rotations);

    std::map<std::string, Vec3> scaled_positions = body_positions;
    for (const LinkDefinition& link : links) {
      const auto skeleton = kSkeletonLinks.find(link.name);
      if (skeleton == kSkeletonLinks.end()) {
        throw RetargetError("missing semantic link " + link.name);
      }
      const auto& [parent, child] = skeleton->second;
      const Vec3 direction =
          Unit(body_positions.at(child) - body_positions.at(parent),
               link.name + " source link");
      scaled_positions[child] =
          scaled_positions.at(parent) + link.length * direction;
    }
    BendLeg(scaled_positions, "left", knee_offset_degrees);
    BendLeg(scaled_positions, "right", knee_offset_degrees);
    AlignBoneRotations(scaled_positions, body_rotations);

    const double floor = std::min(scaled_positions.at("left_foot").z(),
                                  scaled_positions.at("right_foot").z());
    for (auto& [name, position] : scaled_positions) {
      static_cast<void>(name);
      position -= Vec3(0.0, 0.0, floor);
    }

    TargetFrame target;
    target.positions["hips_mean"] = scaled_positions.at("hips_mean");
    target.orientations["hips_mean"] = Quat(
        body_rotations.at("hips_mean") *
        AxisMapMatrix(key_frame_config, "hips_mean") *
        OffsetMatrix(key_frame_config, "hips_mean"));
    for (const LinkDefinition& link : links) {
      const auto& child = kSkeletonLinks.at(link.name).second;
      target.positions[link.name] = scaled_positions.at(child);
      target.orientations[link.name] = Quat(
          body_rotations.at(child) * AxisMapMatrix(key_frame_config, child) *
          OffsetMatrix(key_frame_config, child));
      target.orientations[link.name].normalize();
    }
    return target;
  }

  Transform BodyTransform(int body_id) const {
    const auto matrix = Eigen::Map<
        const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
        data->xmat + 9 * body_id);
    return {Mat3(matrix), Eigen::Map<const Vec3>(data->xpos + 3 * body_id)};
  }

  Eigen::MatrixXd BodyJacobian(int body_id) const {
    Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> position(3,
                                                                       model->nv);
    Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> rotation(3,
                                                                       model->nv);
    mj_jacBody(model.get(), data.get(), position.data(), rotation.data(), body_id);
    const Mat3 world_to_body = BodyTransform(body_id).rotation.transpose();
    Eigen::MatrixXd result(6, model->nv);
    result.topRows(3) = world_to_body * position;
    result.bottomRows(3) = world_to_body * rotation;
    return result;
  }

  Vec6 TaskError(const TaskDefinition& task) const {
    const Transform current = BodyTransform(task.body_id);
    return Se3Log(Compose(Inverse(current), task.target));
  }

  Eigen::MatrixXd TaskJacobian(const TaskDefinition& task) const {
    const Transform current = BodyTransform(task.body_id);
    const Transform target_to_body = Compose(Inverse(task.target), current);
    return -Se3Jlog(target_to_body) * BodyJacobian(task.body_id);
  }

  double ErrorNorm() const {
    double squared = 0.0;
    for (const TaskDefinition& task : tasks) {
      squared += TaskError(task).squaredNorm();
    }
    const double result = std::sqrt(squared);
    if (!std::isfinite(result)) {
      throw RetargetError("native task error is non-finite");
    }
    return result;
  }

  Eigen::VectorXd SolveDelta() const {
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(model->nv, model->nv);
    Eigen::VectorXd linear = Eigen::VectorXd::Zero(model->nv);
    double mu_total = 0.0;
    for (const TaskDefinition& task : tasks) {
      const Vec6 error = TaskError(task);
      const Eigen::MatrixXd jacobian = TaskJacobian(task);
      const Vec6 weighted_error = task.cost.array() * (-error.array());
      const Eigen::MatrixXd weighted_jacobian = task.cost.asDiagonal() * jacobian;
      hessian.noalias() += weighted_jacobian.transpose() * weighted_jacobian;
      linear.noalias() -= weighted_jacobian.transpose() * weighted_error;
      mu_total += weighted_error.squaredNorm();
    }
    hessian.diagonal().array() += kGlobalDamping + mu_total;

    Eigen::VectorXd lower = Eigen::VectorXd::Constant(
        model->nv, -std::numeric_limits<double>::infinity());
    Eigen::VectorXd upper = Eigen::VectorXd::Constant(
        model->nv, std::numeric_limits<double>::infinity());
    for (int joint = 0; joint < model->njnt; ++joint) {
      if (model->jnt_type[joint] == mjJNT_FREE ||
          model->jnt_limited[joint] == 0) {
        continue;
      }
      const int qpos = model->jnt_qposadr[joint];
      const int dof = model->jnt_dofadr[joint];
      lower[dof] = kLimitGain *
                   (data->qpos[qpos] - model->jnt_range[2 * joint]);
      lower[dof] *= -1.0;
      upper[dof] = kLimitGain *
                   (model->jnt_range[2 * joint + 1] - data->qpos[qpos]);
      if (lower[dof] > upper[dof]) {
        throw RetargetError("configuration limit interval is empty");
      }
    }
    return SolveBoxQp(hessian, linear, lower, upper);
  }

  RetargetResult Solve(const PoseFrame& pose) {
    const Eigen::VectorXd before =
        Eigen::Map<const Eigen::VectorXd>(data->qpos, model->nq);
    const auto started = std::chrono::steady_clock::now();
    try {
      const TargetFrame target = MakeTargets(pose);
      for (TaskDefinition& task : tasks) {
        task.target = {target.orientations.at(task.keypoint).toRotationMatrix(),
                       target.positions.at(task.keypoint)};
      }
      double current_error = ErrorNorm();
      int iterations = 0;
      while (true) {
        const Eigen::VectorXd delta = SolveDelta();
        if (!delta.array().isFinite().all()) {
          throw RetargetError("native QP returned a non-finite displacement");
        }
        mj_integratePos(model.get(), data->qpos, delta.data(), 1.0);
        UpdateKinematics();
        ++iterations;
        const double next_error = ErrorNorm();
        if (current_error - next_error <= kConvergenceDelta ||
            iterations >= kMaximumIterations) {
          current_error = next_error;
          break;
        }
        current_error = next_error;
      }
      RetargetResult result;
      for (std::size_t index = 0; index < kKengoJointCount; ++index) {
        result.positions[index] = data->qpos[qpos_indices[index]];
        if (!std::isfinite(result.positions[index])) {
          throw RetargetError("retarget result contains non-finite qpos");
        }
      }
      result.root_position = Eigen::Map<const Vec3>(data->qpos);
      result.root_orientation =
          Quat(data->qpos[3], data->qpos[4], data->qpos[5], data->qpos[6])
              .normalized();
      result.task_error = current_error;
      result.iterations = iterations;
      result.solve_duration_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - started)
              .count();
      return result;
    } catch (...) {
      Eigen::Map<Eigen::VectorXd>(data->qpos, model->nq) = before;
      UpdateKinematics();
      throw;
    }
  }
};

RealtimeKengoRetargeter::RealtimeKengoRetargeter(
    const std::string& model_path, const std::string& config_path)
    : impl_(std::make_unique<Impl>(model_path, config_path, joint_names_)) {}

RealtimeKengoRetargeter::~RealtimeKengoRetargeter() = default;

RetargetResult RealtimeKengoRetargeter::Solve(const PoseFrame& pose) {
  return impl_->Solve(pose);
}

void RealtimeKengoRetargeter::Reset() {
  Eigen::Map<Eigen::VectorXd>(impl_->data->qpos, impl_->model->nq) =
      impl_->neutral_q;
  impl_->UpdateKinematics();
}

int RealtimeKengoRetargeter::nq() const { return impl_->model->nq; }
int RealtimeKengoRetargeter::nv() const { return impl_->model->nv; }

PoseFrame SyntheticSmpl24() {
  PoseFrame pose;
  pose.positions[0] = Vec3(0.0, 1.00, 0.0);
  pose.positions[1] = Vec3(-0.09, 0.95, 0.0);
  pose.positions[2] = Vec3(0.09, 0.95, 0.0);
  pose.positions[3] = Vec3(0.0, 1.12, 0.0);
  pose.positions[4] = Vec3(-0.09, 0.55, 0.02);
  pose.positions[5] = Vec3(0.09, 0.55, 0.02);
  pose.positions[6] = Vec3(0.0, 1.25, 0.0);
  pose.positions[7] = Vec3(-0.09, 0.12, 0.0);
  pose.positions[8] = Vec3(0.09, 0.12, 0.0);
  pose.positions[9] = Vec3(0.0, 1.38, 0.0);
  pose.positions[10] = Vec3(-0.09, 0.05, -0.12);
  pose.positions[11] = Vec3(0.09, 0.05, -0.12);
  pose.positions[12] = Vec3(0.0, 1.52, 0.0);
  pose.positions[13] = Vec3(-0.11, 1.47, 0.0);
  pose.positions[14] = Vec3(0.11, 1.47, 0.0);
  pose.positions[15] = Vec3(0.0, 1.72, 0.0);
  pose.positions[16] = Vec3(-0.24, 1.45, 0.0);
  pose.positions[17] = Vec3(0.24, 1.45, 0.0);
  pose.positions[18] = Vec3(-0.50, 1.43, -0.02);
  pose.positions[19] = Vec3(0.50, 1.43, -0.02);
  pose.positions[20] = Vec3(-0.74, 1.41, -0.04);
  pose.positions[21] = Vec3(0.74, 1.41, -0.04);
  pose.positions[22] = Vec3(-0.80, 1.41, -0.04);
  pose.positions[23] = Vec3(0.80, 1.41, -0.04);
  for (Quat& orientation : pose.orientations) {
    orientation = Quat::Identity();
  }
  return pose;
}

}  // namespace kengo_fullbody

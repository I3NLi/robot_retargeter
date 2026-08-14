#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace kengo_fullbody {

inline constexpr std::size_t kComposedTargetJointCount = 23;
inline constexpr std::size_t kComposedTargetUpperJointCount = 10;

using ComposedTargetNames =
    std::array<std::string, kComposedTargetJointCount>;
using ComposedTargetValues =
    std::array<double, kComposedTargetJointCount>;

struct ComposedTarget {
  ComposedTargetNames joint_names{};
  ComposedTargetValues positions{};
};

class ComposedTargetError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

const ComposedTargetNames& CanonicalComposedTargetJointNames();
bool IsComposedTargetUpperJoint(const std::string& joint_name);

ComposedTarget ComposeFullBodyUpperWithMeasuredLower(
    const std::vector<std::string>& target_names,
    const std::vector<double>& target_positions,
    const std::vector<std::string>& measured_names,
    const std::vector<double>& measured_positions);

}  // namespace kengo_fullbody

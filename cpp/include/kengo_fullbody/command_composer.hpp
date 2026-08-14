#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace kengo_fullbody {

inline constexpr std::size_t kCommandJointCount = 23;
inline constexpr std::size_t kCommandUpperJointCount = 10;

using CommandNames = std::array<std::string, kCommandJointCount>;
using CommandValues = std::array<double, kCommandJointCount>;

struct ComposedCommand {
  CommandNames joint_names{};
  CommandValues position{};
  CommandValues velocity{};
  CommandValues effort{};
  CommandValues kp{};
  CommandValues kd{};
};

class CommandCompositionError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

const CommandNames& CanonicalCommandJointNames();
bool IsCommandUpperJoint(const std::string& joint_name);

class FullBodyCommandComposer {
 public:
  ComposedCommand Compose(const std::vector<std::string>& measured_names,
                          const std::vector<double>& measured_positions,
                          const std::vector<std::string>& target_names,
                          const std::vector<double>& target_positions,
                          double period_seconds);
  ComposedCommand Release(const std::vector<std::string>& measured_names,
                          const std::vector<double>& measured_positions) const;
  void Reset();

 private:
  bool has_previous_upper_{false};
  CommandValues previous_upper_{};
};

}  // namespace kengo_fullbody

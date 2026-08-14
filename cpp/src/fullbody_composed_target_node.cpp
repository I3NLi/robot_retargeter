#include "kengo_fullbody/composed_target.hpp"

#include <hdas2/msg/hybrid_joint_state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace std::chrono_literals;

constexpr char kFullBodyTargetTopic[] =
    "/pico4/retargeted/full_body_joint_targets";
constexpr char kMeasuredJointTopic[] =
    "/hybrid_body_controller/joint_states";
constexpr char kComposedTargetTopic[] =
    "/pico4/retargeted/composed_joint_targets";
constexpr auto kMeasuredMaximumAge = 50ms;

struct MeasuredFeed {
  std::vector<std::string> names;
  std::vector<double> positions;
  std::chrono::steady_clock::time_point received_at{};
};

class FullBodyComposedTargetNode final : public rclcpp::Node {
 public:
  explicit FullBodyComposedTargetNode(const rclcpp::NodeOptions& options)
      : Node("kengo_fullbody_composed_target", options) {
    const auto target_qos =
        rclcpp::QoS(rclcpp::KeepAll()).reliable().durability_volatile();
    const auto measured_qos =
        rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
    const auto output_qos =
        rclcpp::QoS(rclcpp::KeepAll()).reliable().durability_volatile();

    measured_subscription_ =
        create_subscription<hdas2::msg::HybridJointState>(
            kMeasuredJointTopic, measured_qos,
            [this](hdas2::msg::HybridJointState::ConstSharedPtr message) {
              MeasuredFeed feed;
              feed.names = message->name;
              feed.positions = message->position;
              feed.received_at = std::chrono::steady_clock::now();
              measured_ = std::move(feed);
            });
    target_subscription_ =
        create_subscription<sensor_msgs::msg::JointState>(
            kFullBodyTargetTopic, target_qos,
            [this](sensor_msgs::msg::JointState::ConstSharedPtr message) {
              PublishIfMeasuredFresh(*message);
            });
    composed_publisher_ = create_publisher<sensor_msgs::msg::JointState>(
        kComposedTargetTopic, output_qos);
  }

 private:
  void PublishIfMeasuredFresh(const sensor_msgs::msg::JointState& target) {
    if (!measured_.has_value()) {
      missing_measured_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    const auto current = std::chrono::steady_clock::now();
    if (current - measured_->received_at > kMeasuredMaximumAge) {
      stale_measured_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    try {
      const auto composed =
          kengo_fullbody::ComposeFullBodyUpperWithMeasuredLower(
              target.name, target.position, measured_->names,
              measured_->positions);
      sensor_msgs::msg::JointState output;
      output.header.stamp = now();
      output.header.frame_id =
          "kengo_fullbody_upper_measured_lower_target_v1";
      output.name.assign(composed.joint_names.begin(),
                         composed.joint_names.end());
      output.position.assign(composed.positions.begin(),
                             composed.positions.end());
      composed_publisher_->publish(output);
      published_.fetch_add(1, std::memory_order_relaxed);
    } catch (const std::exception& exception) {
      rejected_.fetch_add(1, std::memory_order_relaxed);
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
                            "composed target rejected: %s", exception.what());
    }
  }

  std::optional<MeasuredFeed> measured_;
  std::atomic<std::uint64_t> published_{0};
  std::atomic<std::uint64_t> rejected_{0};
  std::atomic<std::uint64_t> missing_measured_{0};
  std::atomic<std::uint64_t> stale_measured_{0};
  rclcpp::Subscription<hdas2::msg::HybridJointState>::SharedPtr
      measured_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr
      target_subscription_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr
      composed_publisher_;
};

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.enable_rosout(false)
      .start_parameter_services(false)
      .start_parameter_event_publisher(false);
  auto node = std::make_shared<FullBodyComposedTargetNode>(options);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

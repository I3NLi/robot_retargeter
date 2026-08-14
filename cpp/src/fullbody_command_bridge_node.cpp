#include "kengo_fullbody/command_composer.hpp"

#include <hdas2/msg/hybrid_joint_command.hpp>
#include <hdas2/msg/hybrid_joint_state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace std::chrono_literals;

constexpr char kTargetTopic[] =
    "/pico4/retargeted/full_body_joint_targets";
constexpr char kFeedbackTopic[] = "/hybrid_body_controller/joint_states";
constexpr char kCommandTopic[] = "/hybrid_body_controller/commands";
constexpr auto kTargetMaximumAge = 200ms;
constexpr auto kFeedbackMaximumAge = 50ms;
constexpr auto kCommandPeriod = 20ms;

struct JointFeed {
  std::vector<std::string> names;
  std::vector<double> positions;
  std::chrono::steady_clock::time_point received_at{};
  std::uint64_t sequence{0};
};

class FullBodyCommandBridge final : public rclcpp::Node {
 public:
  explicit FullBodyCommandBridge(const rclcpp::NodeOptions& options)
      : Node("kengo_fullbody_command_bridge", options) {
    const auto target_qos =
        rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
    const auto feedback_qos =
        rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
    const auto command_qos = rclcpp::QoS(rclcpp::KeepLast(1))
                                 .reliable()
                                 .transient_local();
    target_subscription_ = create_subscription<sensor_msgs::msg::JointState>(
        kTargetTopic, target_qos,
        [this](sensor_msgs::msg::JointState::ConstSharedPtr message) {
          ReceiveTarget(*message);
        });
    feedback_subscription_ =
        create_subscription<hdas2::msg::HybridJointState>(
            kFeedbackTopic, feedback_qos,
            [this](hdas2::msg::HybridJointState::ConstSharedPtr message) {
              ReceiveFeedback(*message);
            });
    command_publisher_ =
        create_publisher<hdas2::msg::HybridJointCommand>(kCommandTopic,
                                                         command_qos);
    timer_ = create_wall_timer(kCommandPeriod, [this]() { Tick(); });
  }

 private:
  void ReceiveTarget(const sensor_msgs::msg::JointState& message) {
    JointFeed feed;
    feed.names = message.name;
    feed.positions = message.position;
    feed.received_at = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(mutex_);
    feed.sequence = ++target_sequence_;
    target_ = std::move(feed);
  }

  void ReceiveFeedback(const hdas2::msg::HybridJointState& message) {
    JointFeed feed;
    feed.names = message.name;
    feed.positions = message.position;
    feed.received_at = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(mutex_);
    feed.sequence = ++feedback_sequence_;
    feedback_ = std::move(feed);
  }

  static hdas2::msg::HybridJointCommand ToMessage(
      const kengo_fullbody::ComposedCommand& command,
      const rclcpp::Time& stamp, const std::string& frame_id) {
    hdas2::msg::HybridJointCommand output;
    output.header.stamp = stamp;
    output.header.frame_id = frame_id;
    output.joint_name.assign(command.joint_names.begin(),
                             command.joint_names.end());
    output.position.assign(command.position.begin(), command.position.end());
    output.velocity.assign(command.velocity.begin(), command.velocity.end());
    output.effort.assign(command.effort.begin(), command.effort.end());
    output.kp.assign(command.kp.begin(), command.kp.end());
    output.kd.assign(command.kd.begin(), command.kd.end());
    return output;
  }

  void Tick() {
    JointFeed target;
    JointFeed feedback;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!target_.has_value() || !feedback_.has_value()) {
        return;
      }
      target = *target_;
      feedback = *feedback_;
    }

    const auto current = std::chrono::steady_clock::now();
    const bool feeds_fresh = current - target.received_at <= kTargetMaximumAge &&
                             current - feedback.received_at <=
                                 kFeedbackMaximumAge;
    const std::size_t publisher_count = count_publishers(kCommandTopic);
    if (publisher_count > 1) {
      if (!publisher_conflict_.exchange(true)) {
        RCLCPP_ERROR(get_logger(),
                     "another commands publisher is present; publishing is "
                     "disabled");
      }
      active_ = false;
      composer_.Reset();
      return;
    }
    publisher_conflict_ = false;

    if (!feeds_fresh) {
      if (active_) {
        try {
          const auto release =
              composer_.Release(feedback.names, feedback.positions);
          command_publisher_->publish(
              ToMessage(release, now(), "kengo_fullbody_command_released"));
        } catch (const std::exception& exception) {
          RCLCPP_ERROR(get_logger(), "release command rejected: %s",
                       exception.what());
        }
      }
      active_ = false;
      composer_.Reset();
      return;
    }

    try {
      const auto command = composer_.Compose(
          feedback.names, feedback.positions, target.names, target.positions,
          std::chrono::duration<double>(kCommandPeriod).count());
      command_publisher_->publish(ToMessage(
          command, now(), "kengo_fullbody_upper_measured_lower_v1"));
      active_ = true;
      published_.fetch_add(1, std::memory_order_relaxed);
    } catch (const std::exception& exception) {
      rejected_.fetch_add(1, std::memory_order_relaxed);
      if (active_) {
        try {
          const auto release =
              composer_.Release(feedback.names, feedback.positions);
          command_publisher_->publish(
              ToMessage(release, now(), "kengo_fullbody_command_released"));
        } catch (const std::exception&) {
        }
      }
      active_ = false;
      composer_.Reset();
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
                            "command composition rejected: %s",
                            exception.what());
    }
  }

  std::mutex mutex_;
  std::optional<JointFeed> target_;
  std::optional<JointFeed> feedback_;
  std::uint64_t target_sequence_{0};
  std::uint64_t feedback_sequence_{0};
  kengo_fullbody::FullBodyCommandComposer composer_;
  bool active_{false};
  std::atomic<bool> publisher_conflict_{false};
  std::atomic<std::uint64_t> published_{0};
  std::atomic<std::uint64_t> rejected_{0};
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr
      target_subscription_;
  rclcpp::Subscription<hdas2::msg::HybridJointState>::SharedPtr
      feedback_subscription_;
  rclcpp::Publisher<hdas2::msg::HybridJointCommand>::SharedPtr
      command_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.enable_rosout(false)
      .start_parameter_services(false)
      .start_parameter_event_publisher(false);
  auto node = std::make_shared<FullBodyCommandBridge>(options);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

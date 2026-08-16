#include "kengo_fullbody/retarget_core.hpp"

#include <geometry_msgs/msg/pose_array.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace {

using kengo_fullbody::PoseFrame;
using kengo_fullbody::RealtimeKengoRetargeter;
using kengo_fullbody::RetargetError;
using kengo_fullbody::RetargetResult;

constexpr char kInputTopic[] = "/pico4/body_tracking/ik_poses";
constexpr char kUpperBodyTopic[] = "/pico4/retargeted/joint_targets";
constexpr char kOutputTopic[] = "/pico4/retargeted/full_body_joint_targets";
constexpr auto kUpperBodyMaximumAge = std::chrono::milliseconds(250);
constexpr char kUpperFollowVelocityParameter[] =
    "max_upper_follow_velocity_rad_s";
constexpr double kDefaultUpperFollowVelocityRadS = 0.60;
constexpr double kMinimumUpperFollowVelocityRadS = 0.10;
constexpr double kMaximumUpperFollowVelocityRadS = 2.00;
constexpr double kNominalFollowPeriodSeconds = 0.02;
constexpr double kMaximumFollowPeriodSeconds = 0.10;

const std::array<std::string, kengo_fullbody::kPicoUpperBodyJointCount>
    kUpperBodyJointNames = {
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",   "left_elbow_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",   "right_elbow_joint",
    };

constexpr std::size_t kFollowLimitedJointCount = 10;
const std::array<std::string, kFollowLimitedJointCount>
    kFollowLimitedJointNames = {
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",   "left_elbow_joint",
    "left_wrist_roll_joint",     "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",         "right_wrist_roll_joint",
};

struct Options {
  std::string model;
  std::string config;
  std::string fixture;
  std::string replay;
  bool self_test{false};
  bool emit_all{false};
  int benchmark_frames{100};
};

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    auto require_value = [&]() -> std::string {
      if (++index >= argc) {
        throw std::runtime_error("missing value for " + argument);
      }
      return argv[index];
    };
    if (argument == "--model") {
      options.model = require_value();
    } else if (argument == "--config") {
      options.config = require_value();
    } else if (argument == "--fixture") {
      options.fixture = require_value();
    } else if (argument == "--replay") {
      options.replay = require_value();
    } else if (argument == "--emit-all") {
      options.emit_all = true;
    } else if (argument == "--self-test") {
      options.self_test = true;
    } else if (argument == "--benchmark-frames") {
      options.benchmark_frames = std::stoi(require_value());
    } else if (argument == "--ros-args") {
      break;
    } else {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  if (options.model.empty() || options.config.empty()) {
    throw std::runtime_error("--model and --config are required");
  }
  if (options.benchmark_frames < 1 || options.benchmark_frames > 100000) {
    throw std::runtime_error("--benchmark-frames is out of range");
  }
  if (!options.fixture.empty() && !options.replay.empty()) {
    throw std::runtime_error("--fixture and --replay are mutually exclusive");
  }
  return options;
}

PoseFrame ReadFixture(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open fixture: " + path);
  }
  PoseFrame pose;
  for (std::size_t index = 0; index < kengo_fullbody::kSmplJointCount;
       ++index) {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double qx = 0.0;
    double qy = 0.0;
    double qz = 0.0;
    double qw = 0.0;
    if (!(input >> x >> y >> z >> qx >> qy >> qz >> qw)) {
      throw std::runtime_error("fixture must contain exactly 24 pose rows");
    }
    pose.positions[index] = kengo_fullbody::Vec3(x, y, z);
    pose.orientations[index] = kengo_fullbody::Quat(qw, qx, qy, qz);
  }
  std::string extra;
  if (input >> extra) {
    throw std::runtime_error("fixture has trailing data");
  }
  return pose;
}

std::vector<PoseFrame> ReadReplay(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open replay: " + path);
  }
  std::vector<double> values;
  double value = 0.0;
  while (input >> value) {
    values.push_back(value);
  }
  constexpr std::size_t kValuesPerFrame =
      kengo_fullbody::kSmplJointCount * 7;
  if (values.empty() || values.size() % kValuesPerFrame != 0) {
    throw std::runtime_error("replay must contain complete 24x7 frames");
  }
  std::vector<PoseFrame> frames(values.size() / kValuesPerFrame);
  std::size_t cursor = 0;
  for (PoseFrame& frame : frames) {
    for (std::size_t joint = 0; joint < kengo_fullbody::kSmplJointCount;
         ++joint) {
      frame.positions[joint] = kengo_fullbody::Vec3(
          values[cursor], values[cursor + 1], values[cursor + 2]);
      frame.orientations[joint] = kengo_fullbody::Quat(
          values[cursor + 6], values[cursor + 3], values[cursor + 4],
          values[cursor + 5]);
      cursor += 7;
    }
  }
  return frames;
}

void PrintResult(const RetargetResult& result) {
  std::cout << std::setprecision(17) << "{\"positions\":[";
  for (std::size_t index = 0; index < result.positions.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << result.positions[index];
  }
  std::cout << "],\"task_error\":" << result.task_error
            << ",\"iterations\":" << result.iterations
            << ",\"solve_duration_ms\":" << result.solve_duration_ms
            << "}" << std::endl;
}

int RunSelfTest(const Options& options) {
  RealtimeKengoRetargeter retargeter(options.model, options.config);
  std::vector<PoseFrame> replay;
  if (!options.replay.empty()) {
    replay = ReadReplay(options.replay);
  } else if (!options.fixture.empty()) {
    replay.push_back(ReadFixture(options.fixture));
  }
  PoseFrame pose = replay.empty() ? kengo_fullbody::SyntheticSmpl24()
                                  : replay.front();
  const int frame_count = replay.empty()
                              ? options.benchmark_frames
                              : static_cast<int>(replay.size());
  std::vector<double> durations;
  durations.reserve(static_cast<std::size_t>(frame_count));
  RetargetResult result;
  for (int frame = 0; frame < frame_count; ++frame) {
    if (replay.empty()) {
      const double phase = frame_count == 1
                               ? 0.0
                               : static_cast<double>(frame) /
                                     static_cast<double>(frame_count - 1);
      pose = kengo_fullbody::SyntheticSmpl24();
      pose.positions[18].z() -= 0.08 * std::sin(phase * M_PI);
      pose.positions[20].z() -= 0.12 * std::sin(phase * M_PI);
    } else {
      pose = replay[static_cast<std::size_t>(frame)];
    }
    result = retargeter.Solve(pose);
    durations.push_back(result.solve_duration_ms);
    if (options.emit_all) {
      PrintResult(result);
    }
  }
  if (!options.emit_all) {
    PrintResult(result);
  }
  const std::size_t warmup = std::min<std::size_t>(5, durations.size() - 1);
  std::vector<double> steady(durations.begin() +
                                 static_cast<std::ptrdiff_t>(warmup),
                             durations.end());
  std::sort(steady.begin(), steady.end());
  const double mean =
      std::accumulate(steady.begin(), steady.end(), 0.0) / steady.size();
  const std::size_t p95_index = static_cast<std::size_t>(
      std::floor(0.95 * static_cast<double>(steady.size() - 1)));
  std::cout << std::setprecision(8)
            << "{\"ok\":true,\"backend\":\"cpp-mujoco-eigen-boxqp\""
            << ",\"nq\":" << retargeter.nq()
            << ",\"nv\":" << retargeter.nv()
            << ",\"frames\":" << durations.size()
            << ",\"mean_ms\":" << mean
            << ",\"p95_ms\":" << steady[p95_index]
            << ",\"max_ms\":" << steady.back() << "}" << std::endl;
  return 0;
}

struct Sample {
  std::uint64_t sequence{0};
  std::int64_t source_stamp_ns{0};
  PoseFrame pose{};
};

struct UpperBodyTarget {
  std::array<double, kengo_fullbody::kPicoUpperBodyJointCount> positions{};
  std::chrono::steady_clock::time_point received_at{};
  std::uint64_t sequence{0};
};

class FullBodyNode final : public rclcpp::Node {
 public:
  FullBodyNode(const Options& options, const rclcpp::NodeOptions& node_options)
      : Node("kengo_pico_fullbody_retarget", node_options),
        retargeter_(options.model, options.config) {
    const auto& joint_names = retargeter_.joint_names();
    for (std::size_t upper = 0; upper < kFollowLimitedJointNames.size(); ++upper) {
      const auto match = std::find(joint_names.begin(), joint_names.end(),
                                   kFollowLimitedJointNames[upper]);
      if (match == joint_names.end()) {
        throw std::runtime_error("follow-limited upper joint is missing: " +
                                 kFollowLimitedJointNames[upper]);
      }
      follow_limited_indices_[upper] = static_cast<std::size_t>(
          std::distance(joint_names.begin(), match));
    }
    const double initial_follow_velocity = declare_parameter<double>(
        kUpperFollowVelocityParameter, kDefaultUpperFollowVelocityRadS);
    if (!ValidUpperFollowVelocity(initial_follow_velocity)) {
      throw std::runtime_error("initial upper follow velocity is out of range");
    }
    upper_follow_velocity_rad_s_.store(initial_follow_velocity,
                                       std::memory_order_relaxed);
    parameter_callback_ = add_on_set_parameters_callback(
        std::bind(&FullBodyNode::ConfigureParameters, this,
                  std::placeholders::_1));
    const auto input_qos =
        rclcpp::QoS(rclcpp::KeepAll()).best_effort().durability_volatile();
    const auto output_qos =
        rclcpp::QoS(rclcpp::KeepAll()).reliable().durability_volatile();
    const auto upper_qos =
        rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
    publisher_ = create_publisher<sensor_msgs::msg::JointState>(kOutputTopic,
                                                                output_qos);
    subscription_ = create_subscription<geometry_msgs::msg::PoseArray>(
        kInputTopic, input_qos,
        [this](geometry_msgs::msg::PoseArray::ConstSharedPtr message) {
          Receive(*message);
        });
    upper_subscription_ = create_subscription<sensor_msgs::msg::JointState>(
        kUpperBodyTopic, upper_qos,
        [this](sensor_msgs::msg::JointState::ConstSharedPtr message) {
          ReceiveUpperBody(*message);
        });
    worker_ = std::thread([this]() { WorkerMain(); });
  }

  ~FullBodyNode() override { Stop(); }

  void Stop() {
    bool expected = false;
    if (!stopping_.compare_exchange_strong(expected, true)) {
      return;
    }
    condition_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

 private:
  static bool ValidUpperFollowVelocity(double value) {
    return std::isfinite(value) && value >= kMinimumUpperFollowVelocityRadS &&
           value <= kMaximumUpperFollowVelocityRadS;
  }

  rcl_interfaces::msg::SetParametersResult ConfigureParameters(
      const std::vector<rclcpp::Parameter>& parameters) {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    for (const auto& parameter : parameters) {
      if (parameter.get_name() != kUpperFollowVelocityParameter) {
        continue;
      }
      if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE ||
          !ValidUpperFollowVelocity(parameter.as_double())) {
        result.successful = false;
        result.reason =
            "max_upper_follow_velocity_rad_s must be a finite double in "
            "[0.10, 2.00]";
        return result;
      }
      upper_follow_velocity_rad_s_.store(parameter.as_double(),
                                         std::memory_order_relaxed);
    }
    return result;
  }

  void SeedUpperFollowState(const RetargetResult& result,
                            std::chrono::steady_clock::time_point now) {
    if (has_previous_upper_output_) {
      return;
    }
    for (std::size_t upper = 0; upper < follow_limited_indices_.size(); ++upper) {
      previous_upper_output_[upper] =
          result.positions[follow_limited_indices_[upper]];
    }
    previous_upper_output_at_ =
        now - std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                  std::chrono::duration<double>(kNominalFollowPeriodSeconds));
    has_previous_upper_output_ = true;
  }

  void ApplyUpperFollowLimit(
      RetargetResult& result, std::chrono::steady_clock::time_point now) {
    SeedUpperFollowState(result, now);
    double elapsed = std::chrono::duration<double>(
                         now - previous_upper_output_at_)
                         .count();
    if (!std::isfinite(elapsed) || elapsed <= 0.0) {
      elapsed = kNominalFollowPeriodSeconds;
    }
    elapsed = std::min(elapsed, kMaximumFollowPeriodSeconds);
    const double maximum_step =
        upper_follow_velocity_rad_s_.load(std::memory_order_relaxed) * elapsed;
    for (std::size_t upper = 0; upper < follow_limited_indices_.size(); ++upper) {
      const std::size_t target_index = follow_limited_indices_[upper];
      const double base = previous_upper_output_[upper];
      const double desired = result.positions[target_index];
      const double limited =
          base + std::clamp(desired - base, -maximum_step, maximum_step);
      result.positions[target_index] = limited;
      previous_upper_output_[upper] = limited;
    }
    previous_upper_output_at_ = now;
  }

  void ReceiveUpperBody(const sensor_msgs::msg::JointState& message) {
    if (message.name.size() != kUpperBodyJointNames.size() ||
        message.position.size() != kUpperBodyJointNames.size()) {
      upper_rejected_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    UpperBodyTarget target;
    std::array<bool, kengo_fullbody::kPicoUpperBodyJointCount> seen{};
    for (std::size_t source = 0; source < message.name.size(); ++source) {
      const auto match = std::find(kUpperBodyJointNames.begin(),
                                   kUpperBodyJointNames.end(),
                                   message.name[source]);
      if (match == kUpperBodyJointNames.end()) {
        upper_rejected_.fetch_add(1, std::memory_order_relaxed);
        return;
      }
      const auto index = static_cast<std::size_t>(
          std::distance(kUpperBodyJointNames.begin(), match));
      const double value = message.position[source];
      if (seen[index] || !std::isfinite(value) || std::abs(value) > 32.0) {
        upper_rejected_.fetch_add(1, std::memory_order_relaxed);
        return;
      }
      seen[index] = true;
      target.positions[index] = value;
    }
    if (std::find(seen.begin(), seen.end(), false) != seen.end()) {
      upper_rejected_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    target.received_at = std::chrono::steady_clock::now();
    target.sequence =
        upper_accepted_.fetch_add(1, std::memory_order_relaxed) + 1;
    {
      std::lock_guard<std::mutex> lock(upper_mutex_);
      latest_upper_body_ = target;
    }
  }

  std::optional<UpperBodyTarget> FreshUpperBodyTarget() {
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(upper_mutex_);
    if (!latest_upper_body_.has_value() ||
        now - latest_upper_body_->received_at > kUpperBodyMaximumAge) {
      return std::nullopt;
    }
    return latest_upper_body_;
  }

  void Receive(const geometry_msgs::msg::PoseArray& message) {
    if (message.poses.size() != kengo_fullbody::kSmplJointCount) {
      rejected_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    Sample sample;
    sample.source_stamp_ns =
        static_cast<std::int64_t>(message.header.stamp.sec) * 1000000000LL +
        static_cast<std::int64_t>(message.header.stamp.nanosec);
    if (sample.source_stamp_ns < 0) {
      rejected_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    for (std::size_t index = 0; index < message.poses.size(); ++index) {
      const auto& source = message.poses[index];
      sample.pose.positions[index] = kengo_fullbody::Vec3(
          source.position.x, source.position.y, source.position.z);
      sample.pose.orientations[index] = kengo_fullbody::Quat(
          source.orientation.w, source.orientation.x, source.orientation.y,
          source.orientation.z);
    }
    try {
      kengo_fullbody::ValidatePose(sample.pose);
    } catch (const RetargetError&) {
      rejected_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    sample.sequence = accepted_.fetch_add(1, std::memory_order_relaxed) + 1;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.push_back(std::move(sample));
      const auto depth = static_cast<std::uint64_t>(queue_.size());
      std::uint64_t previous = maximum_queue_depth_.load(std::memory_order_relaxed);
      while (depth > previous &&
             !maximum_queue_depth_.compare_exchange_weak(
                 previous, depth, std::memory_order_relaxed)) {
      }
    }
    condition_.notify_one();
  }

  std::optional<Sample> WaitForSample() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this]() { return stopping_ || !queue_.empty(); });
    if (queue_.empty()) {
      return std::nullopt;
    }
    Sample sample = std::move(queue_.front());
    queue_.pop_front();
    return sample;
  }

  void WorkerMain() noexcept {
    auto next_status = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!stopping_.load(std::memory_order_relaxed)) {
      std::optional<Sample> sample = WaitForSample();
      if (!sample.has_value()) {
        continue;
      }
      try {
        RetargetResult result = retargeter_.Solve(sample->pose);
        const auto follow_time = std::chrono::steady_clock::now();
        SeedUpperFollowState(result, follow_time);
        const std::optional<UpperBodyTarget> upper = FreshUpperBodyTarget();
        std::string frame_role = "kengo_torso_target_camera_fullbody";
        if (upper.has_value()) {
          kengo_fullbody::GraftPicoUpperBody(
              retargeter_.joint_names(), upper->positions, result.positions);
          upper_grafted_.fetch_add(1, std::memory_order_relaxed);
          frame_role = "kengo_torso_target_upper_grafted";
        } else {
          upper_unavailable_.fetch_add(1, std::memory_order_relaxed);
        }
        ApplyUpperFollowLimit(result, follow_time);
        sensor_msgs::msg::JointState output;
        if (sample->source_stamp_ns > 0) {
          output.header.stamp.sec = static_cast<std::int32_t>(
              sample->source_stamp_ns / 1000000000LL);
          output.header.stamp.nanosec = static_cast<std::uint32_t>(
              sample->source_stamp_ns % 1000000000LL);
        } else {
          output.header.stamp = now();
        }
        output.header.frame_id = frame_role;
        output.name.assign(retargeter_.joint_names().begin(),
                           retargeter_.joint_names().end());
        output.position.assign(result.positions.begin(), result.positions.end());
        publisher_->publish(output);
        solved_.fetch_add(1, std::memory_order_relaxed);
      } catch (const std::exception& exception) {
        failed_.fetch_add(1, std::memory_order_relaxed);
        retargeter_.Reset();
        std::cerr << "fullbody_retarget_frame_failed=" << exception.what()
                  << std::endl;
      }
      if (std::chrono::steady_clock::now() >= next_status) {
        std::size_t depth = 0;
        {
          std::lock_guard<std::mutex> lock(mutex_);
          depth = queue_.size();
        }
        std::cout << "fullbody_retarget_status accepted=" << accepted_.load()
                  << " rejected=" << rejected_.load()
                  << " solved=" << solved_.load() << " failed=" << failed_.load()
                  << " queue=" << depth
                  << " max_queue=" << maximum_queue_depth_.load()
                  << " upper_accepted=" << upper_accepted_.load()
                  << " upper_rejected=" << upper_rejected_.load()
                  << " upper_grafted=" << upper_grafted_.load()
                  << " upper_unavailable=" << upper_unavailable_.load()
                  << " upper_follow_velocity_rad_s="
                  << upper_follow_velocity_rad_s_.load(
                         std::memory_order_relaxed)
                  << std::endl;
        next_status = std::chrono::steady_clock::now() + std::chrono::seconds(5);
      }
    }
  }

  RealtimeKengoRetargeter retargeter_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr publisher_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr subscription_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr
      upper_subscription_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<Sample> queue_;
  std::thread worker_;
  std::atomic<bool> stopping_{false};
  std::atomic<std::uint64_t> accepted_{0};
  std::atomic<std::uint64_t> rejected_{0};
  std::atomic<std::uint64_t> solved_{0};
  std::atomic<std::uint64_t> failed_{0};
  std::atomic<std::uint64_t> maximum_queue_depth_{0};
  std::mutex upper_mutex_;
  std::optional<UpperBodyTarget> latest_upper_body_;
  std::atomic<std::uint64_t> upper_accepted_{0};
  std::atomic<std::uint64_t> upper_rejected_{0};
  std::atomic<std::uint64_t> upper_grafted_{0};
  std::atomic<std::uint64_t> upper_unavailable_{0};
  std::array<std::size_t, kFollowLimitedJointCount> follow_limited_indices_{};
  std::array<double, kFollowLimitedJointCount> previous_upper_output_{};
  std::chrono::steady_clock::time_point previous_upper_output_at_{};
  bool has_previous_upper_output_{false};
  std::atomic<double> upper_follow_velocity_rad_s_{
      kDefaultUpperFollowVelocityRadS};
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr
      parameter_callback_;
};

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);
    if (options.self_test) {
      return RunSelfTest(options);
    }
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions node_options;
    node_options.enable_rosout(false)
        .start_parameter_services(true)
        .start_parameter_event_publisher(false);
    auto node = std::make_shared<FullBodyNode>(options, node_options);
    rclcpp::spin(node);
    node->Stop();
    rclcpp::shutdown();
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "kengo_fullbody_retarget_cpp_error=" << exception.what()
              << std::endl;
    return 1;
  }
}

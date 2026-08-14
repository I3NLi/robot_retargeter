# Full-body upper + measured lower target

`kengo_fullbody_composed_target_node` publishes a read-only 23-joint telemetry
pose. It does not depend on Walk and never publishes a motor command.

Inputs:

- `/pico4/retargeted/full_body_joint_targets`, `sensor_msgs/msg/JointState`:
  all ten upper-body target joints are used (both shoulders, elbows and wrist
  rolls).
- `/hybrid_body_controller/joint_states`, `hdas2/msg/HybridJointState`:
  `waist_yaw` and all twelve leg joints are used from the latest measured
  feedback.

Output:

- `/pico4/retargeted/composed_joint_targets`,
  `sensor_msgs/msg/JointState`, canonical 23-joint Kengo order.

Each accepted full-body target frame produces one output when measured
feedback exists and is no older than 50 ms. The target and output QoS use
reliable KEEP_ALL so the node does not add a target-frame drop queue; measured
feedback remains BEST_EFFORT KEEP_LAST(1) because only the most recent real
lower-body pose is meaningful. Missing, stale, duplicate, unknown, non-finite
or non-23-joint inputs cannot publish.

The process has no `/hybrid_body_controller/commands` publisher, no timer and
no service/action/control client. Walk may be stopped, idle or running without
changing this telemetry contract.

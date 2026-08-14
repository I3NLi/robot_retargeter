# Native Kengo full-body retarget service

The production service is a standalone C++17 `rclcpp` node. Python/Mink remains
the offline oracle only; the running process does not start a Python
interpreter.

## ROS contract

- Input: `/pico4/body_tracking/ik_poses`, `geometry_msgs/msg/PoseArray`
- Authoritative upper-body input: `/pico4/retargeted/joint_targets`,
  `sensor_msgs/msg/JointState`
- Output: `/pico4/retargeted/full_body_joint_targets`,
  `sensor_msgs/msg/JointState`
- Input QoS: `KEEP_ALL`, best effort, volatile
- Output QoS: `KEEP_ALL`, reliable, volatile
- Exactly two application subscriptions and one telemetry publisher. ROS 2
  Humble's standard `rclcpp::Node` also owns a read-only `/parameter_events`
  subscription for its time/parameter infrastructure; the node publishes no
  parameter events and starts no parameter services.
- No command publisher, service, client, action or timer

Every structurally valid frame is queued and solved in arrival order. There is
no rate gate, latest-frame replacement or age discard. The five-second status
line reports accepted, rejected, solved, failed, current queue depth and maximum
queue depth so a throughput regression is visible.

The native full-body solve owns waist, legs, wrists and the remaining whole-body
configuration. Its eight shoulder/elbow values are replaced by the latest
strictly validated output from the persistent upper-body IK service when that
latest-only mailbox is no older than 250 ms. Missing or stale upper-body data
never blocks camera full-body telemetry: the original full-body solve is
published with frame `kengo_torso_target_camera_fullbody`; a successfully fused
frame uses `kengo_torso_target_upper_grafted`. This service still never
publishes an HDAS command.

## Build

The target needs ROS 2 Humble development files, Eigen3, yaml-cpp and the
MuJoCo 3.3.4 headers/library. The deployment helper compiles and installs the
binary plus `libmujoco.so.3.3.4` into a self-contained release:

```bash
deployment/build_realtime_fullbody_cpp.sh \
  /path/to/robot_retargeter \
  /opt/kengo-fullbody-retarget/releases/CANDIDATE \
  /path/to/mujoco-python-package
```

The helper performs a real-model native self-test after installation. The
runtime launcher sources ROS/Galaxea only for message types and DDS settings,
then directly `exec`s `bin/kengo_fullbody_retarget_node`.

## Activation gates

Before replacing the service, all of the following are required:

1. Walk API reports `running=false` and `pid=null` twice, including immediately
   before the symlink switch.
2. C++/Python sequential replay matches all 23 joints, task error and iteration
   count within the release tolerance.
3. A remapped live probe sustains at least the PICO input rate with queue depth
   returning to zero and no failed frames.
4. The probe node exposes the declared input subscription, the standard
   read-only `/parameter_events` subscription, and the remapped telemetry
   publisher; no other endpoints are allowed.
5. Candidate startup and online verification pass; otherwise the previous
   release symlink and service are restored.

The service is visualization telemetry. It must never publish
`/hybrid_body_controller/commands` or change Walk/HDAS state.

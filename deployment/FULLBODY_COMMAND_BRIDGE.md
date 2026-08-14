# Full-body command bridge

`kengo_fullbody_command_bridge_node` is an independent, opt-in C++ ROS 2
publisher. It composes the upper-body target from the camera full-body
retargeter with the measured robot base and lower body.

## ROS contract

Subscriptions:

- `/pico4/retargeted/full_body_joint_targets`,
  `sensor_msgs/msg/JointState`, latest sample, best-effort/volatile.
- `/hybrid_body_controller/joint_states`,
  `hdas2/msg/HybridJointState`, latest sample, best-effort/volatile.

Publisher:

- `/hybrid_body_controller/commands`,
  `hdas2/msg/HybridJointCommand`, keep-last 1,
  reliable/transient-local.

The ten arm joints (both shoulder pitch/roll/yaw, elbow and wrist roll) use the
full-body target. `waist_yaw` and all twelve leg joints use the current measured
position. The output preserves the joint order reported by HDAS.

At 50 Hz, upper-body motion is limited to 0.60 rad/s (0.012 rad per command)
and to measured position +/-0.25 rad, then clipped to the Kengo joint limits.
Shoulders use KP/KD 22.21/1.41, elbows and wrists 30/1, and measured waist/legs
25/1. Velocity and effort targets are zero.

The target must be no older than 200 ms and feedback no older than 50 ms. A
missing, malformed, duplicate, non-finite or stale input cannot produce an
active command. If an already-active input becomes stale, the bridge emits one
release frame at measured positions with all gains zero and then remains
silent. If another `/hybrid_body_controller/commands` publisher appears, this
bridge stops publishing without racing it.

## Arming boundary

Installation must not arm or enable the unit. The runner exits with status 77
unless the service environment contains:

```text
KENGO_FULLBODY_COMMAND_ARMED=1
```

Before arming, verify all of the following:

1. Walk is stopped and no Walk process exists.
2. `/hybrid_body_controller/commands` has zero publishers.
3. HDAS is in the operator-approved mode and the robot is mechanically safe.
4. Both inputs are live and contain exactly the canonical 23 Kengo joints.
5. An operator is present with an immediate stop path.

The command bridge is deployed independently under
`/opt/kengo-fullbody-command-bridge`; rolling the monitor release backward does
not silently replace this control publisher.

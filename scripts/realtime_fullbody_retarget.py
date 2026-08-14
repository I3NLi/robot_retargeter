#!/usr/bin/env python3
"""Continuously retarget PICO SMPL-24 frames to Kengo full-body joints.

Input:  /pico4/body_tracking/ik_poses (geometry_msgs/PoseArray)
Output: /pico4/retargeted/full_body_joint_targets (sensor_msgs/JointState)

The output topic is telemetry only. This process contains no reference to the
HDAS command topic and creates no service, action, or control client.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Callable
import numpy as np

try:
    from scripts.realtime_fullbody_core import (
        KENGO_JOINT_NAMES,
        RetargetError,
        RetargetResult,
        RealtimeKengoRetargeter,
        synthetic_smpl24,
        validate_smpl24,
    )
except ModuleNotFoundError:
    from realtime_fullbody_core import (  # type: ignore[no-redef]
        KENGO_JOINT_NAMES,
        RetargetError,
        RetargetResult,
        RealtimeKengoRetargeter,
        synthetic_smpl24,
        validate_smpl24,
    )


INPUT_TOPIC = "/pico4/body_tracking/ik_poses"
OUTPUT_TOPIC = "/pico4/retargeted/full_body_joint_targets"
@dataclass(frozen=True)
class PoseSnapshot:
    sequence: int
    received_monotonic: float
    source_stamp_ns: int
    positions: np.ndarray
    quaternions_xyzw: np.ndarray


class LatestPoseMailbox:
    """Single-slot mailbox for a real-time visualization target.

    A solve already in progress is never interrupted.  While it runs, newer
    valid PICO samples atomically replace the one pending sample.  This keeps
    latency bounded when PICO publishes faster than the stateful Mink solver
    can run; an unbounded FIFO would make the card replay increasingly old
    motion and consume memory without bound.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._condition = threading.Condition()
        self._pending: PoseSnapshot | None = None
        self._sequence = 0
        self._closed = False
        self.accepted_frames = 0
        self.rejected_frames = 0
        self.replaced_frames = 0

    def offer(
        self,
        positions: np.ndarray,
        quaternions_xyzw: np.ndarray,
        source_stamp_ns: int = 0,
        received_monotonic: float | None = None,
    ) -> bool:
        try:
            valid_positions, valid_quaternions = validate_smpl24(
                positions, quaternions_xyzw
            )
        except (RetargetError, TypeError, ValueError, OverflowError):
            self.reject()
            return False
        received = self._clock() if received_monotonic is None else float(received_monotonic)
        if not math.isfinite(received) or source_stamp_ns < 0:
            self.reject()
            return False
        with self._condition:
            if self._closed:
                return False
            self._sequence += 1
            self.accepted_frames += 1
            if self._pending is not None:
                self.replaced_frames += 1
            self._pending = PoseSnapshot(
                self._sequence,
                received,
                int(source_stamp_ns),
                valid_positions.copy(),
                valid_quaternions.copy(),
            )
            self._condition.notify()
        return True

    def reject(self) -> None:
        with self._condition:
            self.rejected_frames += 1

    def get(self, timeout: float | None = None) -> PoseSnapshot | None:
        deadline = None if timeout is None else self._clock() + float(timeout)
        with self._condition:
            while self._pending is None and not self._closed:
                remaining = None if deadline is None else deadline - self._clock()
                if remaining is not None and remaining <= 0.0:
                    return None
                self._condition.wait(remaining)
            if self._pending is None:
                return None
            result = self._pending
            self._pending = None
            return result

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    @property
    def pending(self) -> int:
        with self._condition:
            return int(self._pending is not None)


class RealtimeWorker:
    def __init__(
        self,
        retargeter: RealtimeKengoRetargeter,
        frames: LatestPoseMailbox,
        publish: Callable[[RetargetResult, PoseSnapshot], None],
    ) -> None:
        self.retargeter = retargeter
        self.frames = frames
        self.publish = publish
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.failure: BaseException | None = None
        self.solved_frames = 0
        self.failed_frames = 0

    def start(self) -> None:
        if self.thread is not None:
            raise RuntimeError("worker already started")
        self.thread = threading.Thread(target=self._run, name="fullbody-retarget", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.frames.close()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                raise RuntimeError("retarget worker did not stop within 2 seconds")

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set():
                snapshot = self.frames.get(timeout=0.2)
                if snapshot is None:
                    continue
                try:
                    result = self.retargeter.solve(
                        snapshot.positions, snapshot.quaternions_xyzw
                    )
                    self.publish(result, snapshot)
                    self.solved_frames += 1
                except (RetargetError, ValueError, RuntimeError, FloatingPointError):
                    self.failed_frames += 1
                    self.retargeter.reset()
        except BaseException as exc:  # supervisor must see unexpected failures
            self.failure = exc
            self.stop_event.set()


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=root / "asset/robot/kengo_description/mjcf/kengo.xml")
    parser.add_argument("--config", type=Path, default=root / "config/robot/kengo.yaml")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def _self_test(args: argparse.Namespace) -> int:
    retargeter = RealtimeKengoRetargeter(args.model, args.config)
    positions, quaternions = synthetic_smpl24()
    durations: list[float] = []
    result: RetargetResult | None = None
    for phase in np.linspace(0.0, 1.0, 20):
        pose = positions.copy()
        pose[18, 2] -= 0.08 * math.sin(float(phase) * math.pi)
        pose[20, 2] -= 0.12 * math.sin(float(phase) * math.pi)
        result = retargeter.solve(pose, quaternions)
        durations.append(result.solve_duration_ms)
    assert result is not None
    if result.joint_names != KENGO_JOINT_NAMES or result.positions.shape != (23,):
        raise RuntimeError("unexpected output joint contract")
    print(
        {
            "ok": True,
            "input_topic": INPUT_TOPIC,
            "output_topic": OUTPUT_TOPIC,
            "joint_count": len(result.joint_names),
            "mean_solve_ms": float(np.mean(durations[5:])),
            "p95_solve_ms": float(np.percentile(durations[5:], 95)),
            "max_solve_ms": float(np.max(durations[5:])),
            "last_task_error": result.task_error,
        }
    )
    return 0


def _run_ros(args: argparse.Namespace) -> int:
    import rclpy
    from geometry_msgs.msg import PoseArray
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState

    rclpy.init(args=None)
    node = Node(
        "kengo_pico_fullbody_retarget",
        enable_rosout=False,
        start_parameter_services=False,
    )
    implicit_parameter_publisher = getattr(node, "_parameter_event_publisher", None)
    if implicit_parameter_publisher is not None:
        node.destroy_publisher(implicit_parameter_publisher)
        node._parameter_event_publisher = None

    frames = LatestPoseMailbox()
    input_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )
    output_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    publisher = node.create_publisher(JointState, OUTPUT_TOPIC, output_qos)

    def receive(message: PoseArray) -> None:
        if len(message.poses) != 24:
            frames.reject()
            return
        positions = np.asarray(
            [(pose.position.x, pose.position.y, pose.position.z) for pose in message.poses],
            dtype=np.float64,
        )
        quaternions = np.asarray(
            [
                (
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                )
                for pose in message.poses
            ],
            dtype=np.float64,
        )
        stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(
            message.header.stamp.nanosec
        )
        frames.offer(positions, quaternions, stamp_ns)

    subscription = node.create_subscription(PoseArray, INPUT_TOPIC, receive, input_qos)

    def publish(result: RetargetResult, snapshot: PoseSnapshot) -> None:
        message = JointState()
        if snapshot.source_stamp_ns > 0:
            message.header.stamp.sec = snapshot.source_stamp_ns // 1_000_000_000
            message.header.stamp.nanosec = snapshot.source_stamp_ns % 1_000_000_000
        else:
            message.header.stamp = node.get_clock().now().to_msg()
        message.header.frame_id = "kengo_torso_target"
        message.name = list(result.joint_names)
        message.position = result.positions.tolist()
        publisher.publish(message)

    retargeter = RealtimeKengoRetargeter(args.model, args.config)
    worker = RealtimeWorker(retargeter, frames, publish)
    worker.start()

    stop = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop.set()
        worker.stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        while rclpy.ok() and not stop.is_set() and worker.failure is None:
            rclpy.spin_once(node, timeout_sec=0.1)
        if worker.failure is not None:
            raise worker.failure
        return 0
    finally:
        worker.stop()
        del subscription
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "disable")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    args = _parse_args()
    if args.self_test:
        return _self_test(args)
    return _run_ros(args)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3

"""Preinsertion approach as a sequential state machine.

  LOOK_AT_BLOB     Rotate the gripper in place toward the blob until
                   ``taskboard_detected`` is available.
  MOVE_ABOVE_RAIL  Point the camera down and slide XY above the target NIC
                   rail, keeping the current height (rail 2 by default).
  MOVE_ABOVE_PORT  Once the NIC port TF appears, move to a standoff above it
                   (port 0, 4 cm by default).
  HOLD             Keep commanding the final pose.
"""

import sys
import time
from enum import Enum, auto

import numpy as np
import rclpy
from aic_control_interfaces.msg import MotionUpdate, TargetMode, TrajectoryGenerationMode
from aic_control_interfaces.srv import ChangeTargetMode
from aic_perinsertion_utils import quaternion_to_rotation_matrix, rotation_matrix_to_quaternion
from geometry_msgs.msg import Point, PointStamped, Pose, Quaternion, Twist, Vector3, Wrench
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from tf2_ros import Buffer, TransformException, TransformListener


class PreinsertionState(Enum):
    LOOK_AT_BLOB = auto()
    MOVE_ABOVE_RAIL = auto()
    MOVE_ABOVE_PORT = auto()
    HOLD = auto()
    FAILED = auto()


def rotation_angle(rotation_a, rotation_b):
    relative = np.asarray(rotation_a, dtype=np.float64).T @ np.asarray(
        rotation_b, dtype=np.float64
    )
    cosine = np.clip(0.5 * (np.trace(relative) - 1.0), -1.0, 1.0)
    return float(np.arccos(cosine))


def transform_from_msg(transform_msg):
    translation = np.array(
        [
            transform_msg.translation.x,
            transform_msg.translation.y,
            transform_msg.translation.z,
        ],
        dtype=np.float64,
    )
    quaternion = np.array(
        [
            transform_msg.rotation.x,
            transform_msg.rotation.y,
            transform_msg.rotation.z,
            transform_msg.rotation.w,
        ],
        dtype=np.float64,
    )
    return translation, quaternion_to_rotation_matrix(quaternion), quaternion


def nic_rail_center_in_taskboard(rail_index, short_edge_m, long_edge_m, rail_count):
    """Center of a NIC-zone rail in the taskboard XY plane (zone 1)."""
    x_min = -0.5 * float(short_edge_m)
    x_max = 0.0
    y_min = -0.5 * float(long_edge_m)
    y_max = 0.0
    count = max(1, int(rail_count))
    index = int(np.clip(int(rail_index), 0, count - 1))
    span = (y_max - y_min) / float(count)
    return np.array(
        [0.5 * (x_min + x_max), y_min + (index + 0.5) * span, 0.0],
        dtype=np.float64,
    )


def tool_down_rotation(board_rotation):
    """TCP rotation with Z pointing at the board and X along the board X axis."""
    z_tcp = -np.asarray(board_rotation, dtype=np.float64)[:, 2]
    z_norm = np.linalg.norm(z_tcp)
    z_tcp = z_tcp / z_norm if z_norm > 1e-9 else np.array([0.0, 0.0, -1.0])
    x_hint = np.asarray(board_rotation, dtype=np.float64)[:, 0]
    y_tcp = np.cross(z_tcp, x_hint)
    y_norm = np.linalg.norm(y_tcp)
    if y_norm < 1e-6:
        x_hint = np.asarray(board_rotation, dtype=np.float64)[:, 1]
        y_tcp = np.cross(z_tcp, x_hint)
        y_norm = np.linalg.norm(y_tcp)
    y_tcp = y_tcp / max(y_norm, 1e-9)
    x_tcp = np.cross(y_tcp, z_tcp)
    x_tcp = x_tcp / max(np.linalg.norm(x_tcp), 1e-9)
    y_tcp = np.cross(z_tcp, x_tcp)
    y_tcp = y_tcp / max(np.linalg.norm(y_tcp), 1e-9)
    return np.column_stack((x_tcp, y_tcp, z_tcp))


def board_up_axis(board_rotation):
    z_up = np.asarray(board_rotation, dtype=np.float64)[:, 2]
    z_norm = np.linalg.norm(z_up)
    return z_up / z_norm if z_norm > 1e-9 else np.array([0.0, 0.0, 1.0])


def pose_above_point(point_in_base, board_rotation, standoff_m):
    z_up = board_up_axis(board_rotation)
    position = np.asarray(point_in_base, dtype=np.float64) + float(standoff_m) * z_up
    quaternion = rotation_matrix_to_quaternion(tool_down_rotation(board_rotation))
    return position, quaternion


def pose_at_current_height_above_point(point_in_base, board_origin, board_rotation, tcp_in_base):
    """Same height above the board as the current TCP, XY over ``point_in_base``."""
    z_up = board_up_axis(board_rotation)
    height = float(np.dot(np.asarray(tcp_in_base, dtype=np.float64) - board_origin, z_up))
    position = np.asarray(point_in_base, dtype=np.float64) + height * z_up
    quaternion = rotation_matrix_to_quaternion(tool_down_rotation(board_rotation))
    return position, quaternion


class PreinsertionControl(Node):
    def __init__(self):
        super().__init__('preinsertion_control')

        self.controller_namespace = str(
            self.declare_parameter('controller_namespace', 'aic_controller').value
        )
        self.camera_name = str(self.declare_parameter('camera_name', 'center').value).strip()
        default_blob_topic = f'/{self.camera_name}_camera/blob_center'
        default_info_topic = f'/{self.camera_name}_camera/camera_info'
        default_camera_frame = f'{self.camera_name}_camera/optical'
        self.blob_center_topic = str(
            self.declare_parameter('blob_center_topic', default_blob_topic).value
        )
        self.camera_info_topic = str(
            self.declare_parameter('camera_info_topic', default_info_topic).value
        )
        self.camera_frame = str(
            self.declare_parameter('camera_frame', default_camera_frame).value
        )
        self.tcp_frame = str(self.declare_parameter('tcp_frame', 'gripper/tcp').value)
        self.base_frame = str(self.declare_parameter('base_frame', 'base_link').value)
        self.taskboard_frame = str(
            self.declare_parameter('taskboard_frame', 'taskboard_detected').value
        )
        self.control_rate_hz = max(
            1.0, float(self.declare_parameter('control_rate_hz', 25.0).value)
        )
        self.angular_gain = float(self.declare_parameter('angular_gain', 1.2).value)
        self.max_angular_vel = float(self.declare_parameter('max_angular_vel', 0.25).value)
        self.pixel_tolerance = float(self.declare_parameter('pixel_tolerance', 12.0).value)
        self.target_offset_x_px = float(self.declare_parameter('target_offset_x_px', 0.0).value)
        self.target_offset_y_px = float(self.declare_parameter('target_offset_y_px', 0.0).value)
        self.blob_timeout_sec = float(self.declare_parameter('blob_timeout_sec', 0.5).value)
        self.tf_stable_sec = float(self.declare_parameter('tf_stable_sec', 0.3).value)
        self.target_rail_index = int(self.declare_parameter('target_rail_index', 2).value)
        self.target_port_index = int(self.declare_parameter('target_port_index', 0).value)
        nic_port_tf_prefix = str(
            self.declare_parameter('nic_port_tf_prefix', 'nic_port').value
        ).strip() or 'nic_port'
        default_nic_port_frame = (
            f'{nic_port_tf_prefix}_r{self.target_rail_index}_p{self.target_port_index}'
        )
        self.nic_port_frame = str(
            self.declare_parameter('nic_port_frame', default_nic_port_frame).value
        )
        self.taskboard_short_edge_m = float(
            self.declare_parameter('taskboard_short_edge_m', 0.30).value
        )
        self.taskboard_long_edge_m = float(
            self.declare_parameter('taskboard_long_edge_m', 0.42).value
        )
        self.rail_count = max(1, int(self.declare_parameter('rail_count', 5).value))
        self.rail_standoff_m = float(self.declare_parameter('rail_standoff_m', 0.20).value)
        self.port_standoff_m = float(self.declare_parameter('port_standoff_m', 0.04).value)
        self.pose_position_tolerance_m = float(
            self.declare_parameter('pose_position_tolerance_m', 0.012).value
        )
        self.pose_orientation_tolerance_rad = float(
            self.declare_parameter('pose_orientation_tolerance_rad', 0.08).value
        )
        self.pose_timeout_sec = float(self.declare_parameter('pose_timeout_sec', 20.0).value)
        stiffness = float(self.declare_parameter('target_stiffness', 85.0).value)
        damping = float(self.declare_parameter('target_damping', 75.0).value)
        wait_for_controller = bool(self.declare_parameter('wait_for_controller', True).value)
        sensor_qos_depth = max(1, int(self.declare_parameter('sensor_qos_depth', 1).value))
        publisher_qos_depth = max(1, int(self.declare_parameter('publisher_qos_depth', 10).value))

        self.target_stiffness = np.diag([stiffness] * 6).flatten()
        self.target_damping = np.diag([damping] * 6).flatten()

        self.blob_uv = None
        self.last_blob_time = None
        self.image_width = None
        self.image_height = None
        self.fx = None
        self.fy = None
        self._logged_camera_info = False
        self._last_status = None

        self._state = PreinsertionState.LOOK_AT_BLOB
        self._tf_first_seen = None
        self._target_position = None
        self._target_quaternion = None
        self._pose_started = None
        self._rail_reached = False
        self._hold_pose = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.motion_update_publisher = self.create_publisher(
            MotionUpdate,
            f'/{self.controller_namespace}/pose_commands',
            publisher_qos_depth,
        )

        if wait_for_controller:
            while self.motion_update_publisher.get_subscription_count() == 0:
                self.get_logger().info(
                    f"Waiting for subscriber to '{self.controller_namespace}/pose_commands'..."
                )
                time.sleep(1.0)

        self.change_mode_client = self.create_client(
            ChangeTargetMode, f'/{self.controller_namespace}/change_target_mode'
        )
        if wait_for_controller:
            while not self.change_mode_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(
                    f"Waiting for service '{self.controller_namespace}/change_target_mode'..."
                )

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=sensor_qos_depth,
        )

        self.create_subscription(
            PointStamped, self.blob_center_topic, self.blob_center_callback, publisher_qos_depth
        )
        self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, sensor_qos
        )
        self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, publisher_qos_depth
        )

        self.create_timer(1.0 / self.control_rate_hz, self._tick)

        self.get_logger().info(
            f'Preinsertion state machine started. blob={self.blob_center_topic} '
            f'camera={self.camera_frame} rail={self.target_rail_index} '
            f'port_tf={self.nic_port_frame} port_standoff={self.port_standoff_m:.3f}m'
        )

    def blob_center_callback(self, msg):
        self.blob_uv = (float(msg.point.x), float(msg.point.y))
        self.last_blob_time = self.get_clock().now()

    def camera_info_callback(self, msg):
        self.image_width = int(msg.width)
        self.image_height = int(msg.height)
        k = msg.k
        if len(k) >= 6 and k[0] > 1.0 and k[4] > 1.0:
            self.fx = float(k[0])
            self.fy = float(k[4])
        elif self.image_width > 0 and self.image_height > 0:
            self.fx = float(self.image_width)
            self.fy = float(self.image_height)
        if not self._logged_camera_info and self.fx is not None:
            self._logged_camera_info = True
            target_u, target_v = self._image_target()
            self.get_logger().info(
                f'Camera info: {self.image_width}x{self.image_height} '
                f'fx={self.fx:.1f} fy={self.fy:.1f} '
                f'target=({target_u:.1f}, {target_v:.1f}) frame={self.camera_frame}'
            )

    def _image_target(self):
        target_u = 0.5 * self.image_width + self.target_offset_x_px
        target_v = 0.5 * self.image_height - self.target_offset_y_px
        return target_u, target_v

    def lookup_transform(self, target_frame, source_frame):
        try:
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=0.0),
            )
        except TransformException:
            return None

    def can_transform(self, target_frame, source_frame):
        try:
            return self.tf_buffer.can_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=0.0),
            )
        except TransformException:
            return False

    def _tf_is_stable(self, child_frame, parent_frame=None):
        parent = parent_frame or self.base_frame
        if not self.can_transform(parent, child_frame):
            self._tf_first_seen = None
            self._log_status('WAIT_TF', f'{parent} -> {child_frame} not available')
            return False

        now = self.get_clock().now()
        if self._tf_first_seen is None:
            self._tf_first_seen = now
            self.get_logger().info(f'TF seen: {parent} -> {child_frame}')
        seen_for = (now - self._tf_first_seen).nanoseconds * 1e-9
        if seen_for < self.tf_stable_sec:
            self._log_status(
                'WAIT_TF',
                f'{child_frame} seen for {seen_for:.2f}s / {self.tf_stable_sec:.2f}s',
            )
            return False
        return True

    def _set_state(self, new_state):
        if new_state == self._state:
            return
        self.get_logger().info(f'State: {self._state.name} -> {new_state.name}')
        self._state = new_state
        self._tf_first_seen = None
        self._target_position = None
        self._target_quaternion = None
        self._pose_started = None
        self._rail_reached = False
        self._last_status = None

    def _log_status(self, status, details=''):
        message = f'{status}: {details}' if details else status
        if status != self._last_status:
            self._last_status = status
            self.get_logger().info(message)
        else:
            self.get_logger().info(message, throttle_duration_sec=1.0)

    def _zero_twist(self):
        return Twist()

    def _look_at_blob_twist(self):
        if self.image_width is None or self.image_height is None or self.fx is None or self.fy is None:
            self._log_status('LOOK_AT_BLOB', f'no camera info on {self.camera_info_topic}')
            return self._zero_twist()

        if self.blob_uv is None or self.last_blob_time is None:
            self._log_status('LOOK_AT_BLOB', f'no blob center on {self.blob_center_topic}')
            return self._zero_twist()

        blob_age = (self.get_clock().now() - self.last_blob_time).nanoseconds * 1e-9
        if blob_age > self.blob_timeout_sec:
            self._log_status(
                'LOOK_AT_BLOB', f'blob older than {self.blob_timeout_sec:.2f}s'
            )
            return self._zero_twist()

        blob_u, blob_v = self.blob_uv
        target_u, target_v = self._image_target()
        error_u = blob_u - target_u
        error_v = blob_v - target_v
        pixel_error = float(np.hypot(error_u, error_v))
        if pixel_error <= self.pixel_tolerance:
            self._log_status(
                'LOOK_AT_BLOB',
                f'aligned pixel error {pixel_error:.1f}px <= {self.pixel_tolerance:.1f}px',
            )
            return self._zero_twist()

        # Optical frame: X right, Y down, Z forward. For a static world point,
        # ṗ = -ω × p, so ωx < 0 looks down and ωy > 0 looks right.
        omega_cam = np.array(
            [
                -self.angular_gain * (error_v / self.fy),
                self.angular_gain * (error_u / self.fx),
                0.0,
            ],
            dtype=np.float64,
        )
        omega_cam = np.clip(omega_cam, -self.max_angular_vel, self.max_angular_vel)

        cam_in_tcp = self.lookup_transform(self.tcp_frame, self.camera_frame)
        if cam_in_tcp is None:
            self._log_status(
                'LOOK_AT_BLOB',
                f'waiting for TF {self.tcp_frame} -> {self.camera_frame}',
            )
            return self._zero_twist()

        _, r_cam, _ = transform_from_msg(cam_in_tcp.transform)
        omega_tcp = r_cam @ omega_cam

        self._log_status(
            'LOOK_AT_BLOB',
            f'error=({error_u:.1f}, {error_v:.1f})px '
            f'w_cam=({omega_cam[0]:.3f}, {omega_cam[1]:.3f}) rad/s',
        )

        # Angular-only twist in gripper/tcp: Cartesian mode rotates about the
        # TCP origin, so the gripper position stays put.
        twist = Twist()
        twist.angular.x = float(omega_tcp[0])
        twist.angular.y = float(omega_tcp[1])
        twist.angular.z = float(omega_tcp[2])
        return twist

    def _rail_hover_pose(self):
        board = self.lookup_transform(self.base_frame, self.taskboard_frame)
        tcp = self.lookup_transform(self.base_frame, self.tcp_frame)
        if board is None or tcp is None:
            return None
        t_board, r_board, _ = transform_from_msg(board.transform)
        t_tcp, _, _ = transform_from_msg(tcp.transform)
        rail_in_board = nic_rail_center_in_taskboard(
            self.target_rail_index,
            self.taskboard_short_edge_m,
            self.taskboard_long_edge_m,
            self.rail_count,
        )
        rail_in_base = t_board + r_board @ rail_in_board
        return pose_at_current_height_above_point(rail_in_base, t_board, r_board, t_tcp)

    def _port_hover_pose(self):
        port = self.lookup_transform(self.base_frame, self.nic_port_frame)
        board = self.lookup_transform(self.base_frame, self.taskboard_frame)
        if port is None or board is None:
            return None
        t_port, _, _ = transform_from_msg(port.transform)
        _, r_board, _ = transform_from_msg(board.transform)
        return pose_above_point(t_port, r_board, self.port_standoff_m)

    def _ensure_target_pose(self, pose_fn, log_label):
        if self._target_position is not None:
            return True
        pose = pose_fn()
        if pose is None:
            self._log_status(log_label, 'waiting for pose')
            return False
        self._target_position, self._target_quaternion = pose
        self._pose_started = self.get_clock().now()
        self._hold_pose = pose
        self.get_logger().info(
            f'{log_label}: target '
            f'({self._target_position[0]:.3f}, {self._target_position[1]:.3f}, '
            f'{self._target_position[2]:.3f}) m in {self.base_frame}'
        )
        return True

    def _pose_reached(self, log_label):
        tcp = self.lookup_transform(self.base_frame, self.tcp_frame)
        if tcp is None:
            self._log_status(log_label, f'waiting for TF {self.tcp_frame}')
            return False

        position, rotation, _ = transform_from_msg(tcp.transform)
        position_error = float(np.linalg.norm(position - self._target_position))
        orientation_error = rotation_angle(
            rotation, quaternion_to_rotation_matrix(self._target_quaternion)
        )
        self._log_status(
            log_label,
            f'pos err {position_error * 1000:.1f} mm, '
            f'ori err {np.degrees(orientation_error):.1f} deg',
        )
        if (
            position_error <= self.pose_position_tolerance_m
            and orientation_error <= self.pose_orientation_tolerance_rad
        ):
            self.get_logger().info(f'{log_label}: reached target')
            return True

        elapsed = (self.get_clock().now() - self._pose_started).nanoseconds * 1e-9
        if elapsed > self.pose_timeout_sec:
            self.get_logger().warn(
                f'{log_label}: timed out after {elapsed:.1f}s '
                f'(pos err {position_error:.3f} m)'
            )
            self._set_state(PreinsertionState.FAILED)
        return False

    def _tick(self):
        if self._state == PreinsertionState.LOOK_AT_BLOB:
            self._tick_look_at_blob()
        elif self._state == PreinsertionState.MOVE_ABOVE_RAIL:
            self._tick_move_above_rail()
        elif self._state == PreinsertionState.MOVE_ABOVE_PORT:
            self._tick_move_above_port()
        elif self._state == PreinsertionState.HOLD:
            self._publish_hold_pose()
            self._log_status('HOLD', 'holding final pose')
        else:
            self._publish_hold_or_zero()
            self._log_status('FAILED', 'holding last command')

    def _tick_look_at_blob(self):
        if self._tf_is_stable(self.taskboard_frame, self.base_frame):
            self.get_logger().info(f'TF available: {self.base_frame} -> {self.taskboard_frame}')
            self._publish_velocity(self._zero_twist())
            self._set_state(PreinsertionState.MOVE_ABOVE_RAIL)
            return
        self._publish_velocity(self._look_at_blob_twist())

    def _tick_move_above_rail(self):
        if not self._rail_reached:
            if not self._ensure_target_pose(self._rail_hover_pose, 'ABOVE_RAIL'):
                self._publish_velocity(self._zero_twist())
                return
            self._publish_pose(self._target_position, self._target_quaternion)
            if self._pose_reached('ABOVE_RAIL'):
                self._rail_reached = True
                self._tf_first_seen = None
                self.get_logger().info(
                    f'Waiting for TF {self.base_frame} -> {self.nic_port_frame}'
                )
            return

        self._publish_pose(self._target_position, self._target_quaternion)
        if self._tf_is_stable(self.nic_port_frame, self.base_frame):
            self.get_logger().info(f'TF available: {self.base_frame} -> {self.nic_port_frame}')
            self._set_state(PreinsertionState.MOVE_ABOVE_PORT)

    def _tick_move_above_port(self):
        if not self._ensure_target_pose(self._port_hover_pose, 'ABOVE_PORT'):
            self._publish_hold_or_zero()
            return
        self._publish_pose(self._target_position, self._target_quaternion)
        if self._pose_reached('ABOVE_PORT'):
            self._set_state(PreinsertionState.HOLD)

    def _publish_hold_pose(self):
        if self._hold_pose is None:
            self._publish_velocity(self._zero_twist())
            return
        position, quaternion = self._hold_pose
        self._publish_pose(position, quaternion)

    def _publish_hold_or_zero(self):
        if self._hold_pose is not None:
            self._publish_hold_pose()
        else:
            self._publish_velocity(self._zero_twist())

    def _fill_common(self, msg, frame_id):
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.target_stiffness = self.target_stiffness
        msg.target_damping = self.target_damping
        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        msg.wrench_feedback_gains_at_tip = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        return msg

    def _publish_velocity(self, twist):
        msg = self._fill_common(MotionUpdate(), self.tcp_frame)
        msg.velocity = twist
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY
        self.motion_update_publisher.publish(msg)

    def _publish_pose(self, position, quaternion):
        msg = self._fill_common(MotionUpdate(), self.base_frame)
        msg.pose = Pose(
            position=Point(x=float(position[0]), y=float(position[1]), z=float(position[2])),
            orientation=Quaternion(
                x=float(quaternion[0]),
                y=float(quaternion[1]),
                z=float(quaternion[2]),
                w=float(quaternion[3]),
            ),
        )
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_POSITION
        self.motion_update_publisher.publish(msg)

    def send_change_control_mode_req(self, mode):
        if not self.change_mode_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(
                f"Service '{self.controller_namespace}/change_target_mode' not available"
            )
            return
        req = ChangeTargetMode.Request()
        req.target_mode.mode = mode
        self.get_logger().info(f'Sending request to change control mode to {mode}')
        future = self.change_mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        response = future.result()
        if response is not None and response.success:
            self.get_logger().info(f'Successfully changed control mode to {mode}')
        else:
            self.get_logger().warn(f'Failed to change control mode to {mode}')
        time.sleep(0.5)


def main(args=None):
    try:
        with rclpy.init(args=args):
            node = PreinsertionControl()
            try:
                node.send_change_control_mode_req(TargetMode.MODE_CARTESIAN)
                rclpy.spin(node)
            finally:
                node.destroy_node()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main(sys.argv)

#!/usr/bin/env python3

"""Center a detected blob in the wrist camera by commanding Cartesian TCP velocity.

Motion uses the same gripper/tcp velocity convention as cartesian_keyboard_teleop:
  a/d  -> -/+ linear.x  (camera left / right)
  w/s  -> -/+ linear.y  (camera up / down in the image)
"""

import sys
import time

import numpy as np
import rclpy
from aic_control_interfaces.msg import MotionUpdate, TargetMode, TrajectoryGenerationMode
from aic_control_interfaces.srv import ChangeTargetMode
from geometry_msgs.msg import PointStamped, Twist, Vector3, Wrench
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo


class PreinsertionControl(Node):
    def __init__(self):
        super().__init__('preinsertion_control')

        self.controller_namespace = str(
            self.declare_parameter('controller_namespace', 'aic_controller').value
        )
        self.camera_name = str(self.declare_parameter('camera_name', 'center').value).strip()
        default_blob_topic = f'/{self.camera_name}_camera/blob_center'
        default_info_topic = f'/{self.camera_name}_camera/camera_info'
        self.blob_center_topic = str(
            self.declare_parameter('blob_center_topic', default_blob_topic).value
        )
        self.camera_info_topic = str(
            self.declare_parameter('camera_info_topic', default_info_topic).value
        )
        self.frame_id = str(self.declare_parameter('frame_id', 'gripper/tcp').value)
        self.control_rate_hz = max(
            1.0, float(self.declare_parameter('control_rate_hz', 25.0).value)
        )
        self.linear_gain = float(self.declare_parameter('linear_gain', 0.35).value)
        self.max_linear_vel = float(self.declare_parameter('max_linear_vel', 0.1).value)
        self.pixel_tolerance = float(self.declare_parameter('pixel_tolerance', 12.0).value)
        # Positive y offset aims above image center to compensate for the gripper mask
        # cutting the bottom of the blob (image y increases downward).
        self.target_offset_x_px = float(self.declare_parameter('target_offset_x_px', 0.0).value)
        self.target_offset_y_px = float(self.declare_parameter('target_offset_y_px', 80.0).value)
        self.blob_timeout_sec = float(self.declare_parameter('blob_timeout_sec', 0.5).value)
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
        self._last_status = None
        self._logged_camera_info = False

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
        # Camera info may be published as sensor (best-effort) or default reliable QoS.
        self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, sensor_qos
        )
        self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, publisher_qos_depth
        )

        self.create_timer(1.0 / self.control_rate_hz, self.send_references)

        self.get_logger().info(
            f'Preinsertion control started. blob={self.blob_center_topic} '
            f'camera_info={self.camera_info_topic} frame={self.frame_id}'
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
                f'target=({target_u:.1f}, {target_v:.1f}) '
                f'(offset {self.target_offset_x_px:.1f}, {self.target_offset_y_px:.1f}px)'
            )

    def generate_velocity_motion_update(self, twist, frame_id):
        msg = MotionUpdate()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.velocity = twist
        msg.target_stiffness = self.target_stiffness
        msg.target_damping = self.target_damping
        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        msg.wrench_feedback_gains_at_tip = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY
        return msg

    def _image_target(self):
        target_u = 0.5 * self.image_width + self.target_offset_x_px
        target_v = 0.5 * self.image_height - self.target_offset_y_px
        return target_u, target_v

    def _zero_twist(self):
        return Twist()

    def _log_status(self, status, details=''):
        message = f'{status}: {details}' if details else status
        if status != self._last_status:
            self._last_status = status
            self.get_logger().info(message)
        elif status == 'TRACKING':
            self.get_logger().info(message, throttle_duration_sec=1.0)

    def _compute_tcp_velocity(self):
        if self.image_width is None or self.image_height is None or self.fx is None or self.fy is None:
            self._log_status('WAITING', f'no camera info on {self.camera_info_topic}')
            return self._zero_twist()

        if self.blob_uv is None or self.last_blob_time is None:
            self._log_status('WAITING', f'no blob center on {self.blob_center_topic}')
            return self._zero_twist()

        blob_age = (self.get_clock().now() - self.last_blob_time).nanoseconds * 1e-9
        if blob_age > self.blob_timeout_sec:
            self._log_status('LOST', f'blob older than {self.blob_timeout_sec:.2f}s')
            return self._zero_twist()

        blob_u, blob_v = self.blob_uv
        target_u, target_v = self._image_target()
        error_u = blob_u - target_u
        error_v = blob_v - target_v
        pixel_error = float(np.hypot(error_u, error_v))

        if pixel_error <= self.pixel_tolerance:
            self._log_status(
                'ALIGNED',
                f'pixel error {pixel_error:.1f}px <= {self.pixel_tolerance:.1f}px',
            )
            return self._zero_twist()

        # Normalized image error. Positive u is right in the image; positive v is down.
        # Matches cartesian_keyboard_teleop: d = +x (right), s = +y (down).
        vx = self.linear_gain * (error_u / self.fx)
        vy = self.linear_gain * (error_v / self.fy)
        vx = float(np.clip(vx, -self.max_linear_vel, self.max_linear_vel))
        vy = float(np.clip(vy, -self.max_linear_vel, self.max_linear_vel))

        self._log_status(
            'TRACKING',
            f'error=({error_u:.1f}, {error_v:.1f})px vel=({vx:.3f}, {vy:.3f}) m/s',
        )

        twist = Twist()
        twist.linear.x = vx
        twist.linear.y = vy
        return twist

    def send_references(self):
        twist = self._compute_tcp_velocity()
        self.motion_update_publisher.publish(
            self.generate_velocity_motion_update(twist=twist, frame_id=self.frame_id)
        )

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
